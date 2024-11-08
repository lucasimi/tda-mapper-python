from random import randrange

from tdamapper.utils.metrics import get_metric
from tdamapper.utils.heap import MaxHeap


def _swap(arr, i, j):
    arr[i], arr[j] = arr[j], arr[i]


def _mid(start, end):
    return (start + end) // 2


def _partition(data, start, end, p_ord):
    higher = start
    for j in range(start, end):
        j_ord, _, _ = data[j]
        if j_ord < p_ord:
            _swap(data, higher, j)
            higher += 1
    return higher


def _quickselect(data, start, end, k):
    if (k < start) or (k >= end):
        return
    start_, end_, higher = start, end, None
    while higher != k + 1:
        p, _, _ = data[k]
        _swap(data, start_, k)
        higher = _partition(data, start_ + 1, end_, p)
        _swap(data, start_, higher - 1)
        if k <= higher - 1:
            end_ = higher
        else:
            start_ = higher


class VPTree:

    def __init__(
        self,
        X,
        metric='euclidean',
        metric_params=None,
        leaf_capacity=1,
        leaf_radius=0.0,
        pivoting=None
    ):
        self.__metric = metric
        self.__metric_params = metric_params
        self.__leaf_capacity = leaf_capacity
        self.__leaf_radius = leaf_radius
        self.__pivoting = pivoting
        self.__dataset = self._Build(self, X).build()

    def get_metric(self):
        return self.__metric

    def get_metric_params(self):
        return self.__metric_params

    def get_leaf_capacity(self):
        return self.__leaf_capacity

    def get_leaf_radius(self):
        return self.__leaf_radius

    def get_pivoting(self):
        return self.__pivoting

    def _get_dataset(self):
        return self.__dataset

    def _get_distance(self):
        metric_params = self.__metric_params or {}
        return get_metric(self.__metric, **metric_params)

    class _Build:

        def __init__(self, vpt, X):
            self.__distance = vpt._get_distance()
            self.__dataset = [(0.0, x, False) for x in X]
            self.__leaf_capacity = vpt.get_leaf_capacity()
            self.__leaf_radius = vpt.get_leaf_radius()
            pivoting = vpt.get_pivoting()
            self.__pivoting = self._pivoting_disabled
            if pivoting == 'random':
                self.__pivoting = self._pivoting_random
            elif pivoting == 'furthest':
                self.__pivoting = self._pivoting_furthest

        def _pivoting_disabled(self, start, end):
            pass

        def _pivoting_random(self, start, end):
            if end <= start:
                return
            pivot = randrange(start, end)
            if pivot > start:
                _swap(self.__dataset, start, pivot)

        def _furthest(self, start, end, i):
            furthest_dist = 0.0
            furthest = start
            _, i_point, _ = self.__dataset[i]
            for j in range(start, end):
                _, j_point, _ = self.__dataset[j]
                j_dist = self.__distance(i_point, j_point)
                if j_dist > furthest_dist:
                    furthest = j
                    furthest_dist = j_dist
            return furthest

        def _pivoting_furthest(self, start, end):
            if end <= start:
                return
            rnd = randrange(start, end)
            furthest_rnd = self._furthest(start, end, rnd)
            furthest = self._furthest(start, end, furthest_rnd)
            if furthest > start:
                _swap(self.__dataset, start, furthest)

        def _update(self, start, end):
            self.__pivoting(start, end)
            _, v_point, is_terminal = self.__dataset[start]
            for i in range(start + 1, end):
                _, point, _ = self.__dataset[i]
                self.__dataset[i] = self.__distance(v_point, point), point, is_terminal

        def build(self):
            self._build_iter()
            return self.__dataset

        def _build_iter(self):
            stack = [(0, len(self.__dataset))]
            while stack:
                start, end = stack.pop()
                mid = _mid(start, end)
                self._update(start, end)
                _, v_point, _ = self.__dataset[start]
                _quickselect(self.__dataset, start + 1, end, mid)
                v_radius, _, _ = self.__dataset[mid]
                if (
                    (end - start > 2 * self.__leaf_capacity) and
                    (v_radius > self.__leaf_radius)
                ):
                    self.__dataset[start] = (v_radius, v_point, False)
                    stack.append((mid, end))
                    stack.append((start + 1, mid))
                else:
                    self.__dataset[start] = (v_radius, v_point, True)

    def ball_search(self, point, eps, inclusive=True):
        return self._BallSearch(self, point, eps, inclusive).search()

    class _BallSearch:

        def __init__(self, vpt, point, eps, inclusive=True):
            self.__dataset = vpt._get_dataset()
            self.__distance = vpt._get_distance()
            self.__point = point
            self.__eps = eps
            self.__inclusive = inclusive

        def search(self):
            return self._search_iter()

        def _inside(self, dist):
            if self.__inclusive:
                return dist <= self.__eps
            return dist < self.__eps

        def _search_iter(self):
            stack = [(0, len(self.__dataset))]
            result = []
            while stack:
                start, end = stack.pop()
                v_radius, v_point, is_terminal = self.__dataset[start]
                if is_terminal:
                    for _, x, _ in self.__dataset[start:end]:
                        dist = self.__distance(self.__point, x)
                        if self._inside(dist):
                            result.append(x)
                else:
                    dist = self.__distance(self.__point, v_point)
                    mid = _mid(start, end)
                    if self._inside(dist):
                        result.append(v_point)
                    if dist <= v_radius:
                        fst = (start + 1, mid)
                        snd = (mid, end)
                    else:
                        fst = (mid, end)
                        snd = (start + 1, mid)
                    if abs(dist - v_radius) <= self.__eps:
                        stack.append(snd)
                    stack.append(fst)
            return result

    def knn_search(self, point, k):
        return self._KnnSearch(self, point, k).search()

    class _KnnSearch:

        def __init__(self, vpt, point, neighbors):
            self.__dataset = vpt._get_dataset()
            self.__distance = vpt._get_distance()
            self.__point = point
            self.__neighbors = neighbors
            self.__radius = float('inf')
            self.__result = MaxHeap()

        def _get_items(self):
            while len(self.__result) > self.__neighbors:
                self.__result.pop()
            return [x for (_, x) in self.__result]

        def search(self):
            self._search_iter()
            return self._get_items()

        def _process(self, x):
            dist = self.__distance(self.__point, x)
            if dist >= self.__radius:
                return dist
            self.__result.add(dist, x)
            while len(self.__result) > self.__neighbors:
                self.__result.pop()
            if len(self.__result) == self.__neighbors:
                self.__radius, _ = self.__result.top()
            return dist

        def _search_iter(self):
            PRE, POST = 0, 1
            self.__result = MaxHeap()
            stack = [(0, len(self.__dataset), 0.0, PRE)]
            while stack:
                start, end, thr, action = stack.pop()
                v_radius, v_point, is_terminal = self.__dataset[start]
                if is_terminal:
                    for _, x, _ in self.__dataset[start:end]:
                        self._process(x)
                else:
                    if action == PRE:
                        mid = _mid(start, end)
                        dist = self._process(v_point)
                        if dist <= v_radius:
                            fst_start, fst_end = start + 1, mid
                            snd_start, snd_end = mid, end
                        else:
                            fst_start, fst_end = mid, end
                            snd_start, snd_end = start + 1, mid
                        stack.append((snd_start, snd_end, abs(v_radius - dist), POST))
                        stack.append((fst_start, fst_end, 0.0, PRE))
                    elif action == POST:
                        if self.__radius > thr:
                            stack.append((start, end, 0.0, PRE))
            return self._get_items()
