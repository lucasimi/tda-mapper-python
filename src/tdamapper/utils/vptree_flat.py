from random import randrange

from tdamapper.utils.metrics import get_metric
from tdamapper.utils.quickselect import quickselect
from tdamapper.utils.heap import MaxHeap


def _swap(arr, i, j):
    arr[i], arr[j] = arr[j], arr[i]


def _mid(start, end):
    return (start + end) // 2


class VPTree:

    def __init__(
        self,
        X,
        metric='euclidean',
        metric_params=None,
        leaf_capacity=1,
        leaf_radius=0.0,
        pivoting=None,
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
            self.__dataset = [(0.0, x) for x in X]
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
            if end - start < 2:
                return
            pivot = randrange(start, end)
            if pivot > start:
                _swap(self.__dataset, start, pivot)

        def _furthest(self, start, end, i):
            furthest_dist = 0.0
            furthest = start
            _, i_point = self.__dataset[i]
            for j in range(start, end):
                _, j_point = self.__dataset[j]
                j_dist = self.__distance(i_point, j_point)
                if j_dist > furthest_dist:
                    furthest = j
                    furthest_dist = j_dist
            return furthest

        def _pivoting_furthest(self, start, end):
            if end - start < 2:
                return
            rnd = randrange(start, end)
            furthest_rnd = self._furthest(start, end, rnd)
            furthest = self._furthest(start, end, furthest_rnd)
            if furthest > start:
                _swap(self.__dataset, start, furthest)

        def _update(self, start, end):
            self.__pivoting(start, end)
            _, v_point = self.__dataset[start]
            for i in range(start, end):
                _, point = self.__dataset[i]
                self.__dataset[i] = self.__distance(v_point, point), point

        def build(self):
            self._build_iter()
            return self.__dataset

        def _build_iter(self):
            stack = [(0, len(self.__dataset))]
            while stack:
                start, end = stack.pop()
                mid = _mid(start, end)
                self._update(start, end)
                _, v_point = self.__dataset[start]
                quickselect(self.__dataset, start + 1, end, mid)
                v_radius, _ = self.__dataset[mid]
                self.__dataset[start] = (v_radius, v_point)
                if (end - start > 2 * self.__leaf_capacity) and (v_radius > self.__leaf_radius):
                    stack.append((mid, end))
                    stack.append((start + 1, mid))

    def ball_search(self, point, eps, inclusive=True):
        return self._BallSearch(self, point, eps, inclusive).search()

    class _BallSearch:

        def __init__(self, vpt, center, radius, inclusive):
            self.__distance = vpt._get_distance()
            self.__dataset = vpt._get_dataset()
            self.__leaf_capacity = vpt.get_leaf_capacity()
            self.__leaf_radius = vpt.get_leaf_radius()
            self.__center = center
            self.__radius = radius
            self.__items = []
            self.__inclusive = inclusive

        class _BSVisit:

            def __init__(self, start, end, m_radius):
                self.__start = start
                self.__end = end
                self.__m_radius = m_radius

            def bounds(self):
                return self.__start, self.__end, self.__m_radius

        def search(self):
            stack = [self._BSVisit(0, len(self.__dataset), float('inf'))]
            while stack:
                visit = stack.pop()
                start, end, m_radius = visit.bounds()
                v_radius, v_point = self.__dataset[start]
                if (end - start <= 2 * self.__leaf_capacity) or (m_radius <= self.__leaf_radius) or (v_radius <= self.__leaf_radius):
                    for _, x in self.__dataset[start:end]:
                        dist = self.__distance(self.__center, x)
                        if self._inside(dist):
                            self.__items.append(x)
                else:
                    dist = self.__distance(self.__center, v_point)
                    if self._inside(dist):
                        self.__items.append(v_point)
                    mid = _mid(start, end)
                    if dist <= v_radius:
                        fst_start, fst_end, fst_radius = start + 1, mid, v_radius
                        snd_start, snd_end, snd_radius = mid, end, float('inf')
                    else:
                        fst_start, fst_end, fst_radius = mid, end, float('inf')
                        snd_start, snd_end, snd_radius = start + 1, mid, v_radius
                    if abs(dist - v_radius) <= self.__radius:
                        stack.append(self._BSVisit(snd_start, snd_end, snd_radius))
                    stack.append(self._BSVisit(fst_start, fst_end, fst_radius))
            return self.__items

        def _inside(self, dist):
            if self.__inclusive:
                return dist <= self.__radius
            return dist < self.__radius

    def knn_search(self, point, neighbors):
        return self._KNNSearch(self, point, neighbors).search()

    class _KNNSearch:

        def __init__(self, vpt, center, neighbors):
            self.__distance = vpt._get_distance()
            self.__dataset = vpt._get_dataset()
            self.__leaf_capacity = vpt.get_leaf_capacity()
            self.__leaf_radius = vpt.get_leaf_radius()
            self.__center = center
            self.__neighbors = neighbors
            self.__items = MaxHeap()

        def _get_items(self):
            while len(self.__items) > self.__neighbors:
                self.__items.pop()
            return [x for (_, x) in self.__items]

        def _get_radius(self):
            if len(self.__items) < self.__neighbors:
                return float('inf')
            furthest_dist, _ = self.__items.top()
            return furthest_dist

        def _process(self, x):
            dist = self.__distance(self.__center, x)
            if dist >= self._get_radius():
                return dist
            self.__items.add(dist, x)
            while len(self.__items) > self.__neighbors:
                self.__items.pop()
            return dist

        def pre(self, pre, stack):
            start, end, _ = pre.bounds()
            v_radius, v_point = self.__dataset[start]
            dist = self._process(v_point)
            mid = _mid(start, end)
            if dist <= v_radius:
                fst_start, fst_end, fst_radius = start + 1, mid, v_radius
                snd_start, snd_end, snd_radius = mid, end, float('inf')
            else:
                fst_start, fst_end, fst_radius = mid, end, float('inf')
                snd_start, snd_end, snd_radius = start + 1, mid, v_radius
            stack.append((self._KVPost(snd_start, snd_end, snd_radius, dist, v_radius), self.post))
            stack.append((self._KVPre(fst_start, fst_end, fst_radius), self.pre))

        def post(self, post, stack):
            start, end, _ = post.bounds()
            m_radius, dist, v_radius = post.rad()
            if abs(dist - v_radius) <= self._get_radius():
                stack.append((self._KVPre(start, end, m_radius), self.pre))

        def search(self):
            stack = [(self._KVPre(0, len(self.__dataset), float('inf')), self.pre)]
            while stack:
                visit, after = stack.pop()
                start, end, m_radius = visit.bounds()
                v_radius, _ = self.__dataset[start]
                if (end - start <= 2 * self.__leaf_capacity) or (m_radius <= self.__leaf_radius) or (v_radius <= self.__leaf_radius):
                    for _, x in self.__dataset[start:end]:
                        self._process(x)
                else:
                    after(visit, stack)
            return self._get_items()

        class _KVPre:

            def __init__(self, start, end, m_radius):
                self.__start = start
                self.__end = end
                self.__m_radius = m_radius

            def bounds(self):
                return self.__start, self.__end, self.__m_radius

        class _KVPost:

            def __init__(self, start, end, m_radius, dist, v_radius):
                self.__start = start
                self.__end = end
                self.__m_radius = m_radius
                self.__dist = dist
                self.__v_radius = v_radius

            def bounds(self):
                return self.__start, self.__end, self.__m_radius

            def rad(self):
                return self.__m_radius, self.__dist, self.__v_radius
