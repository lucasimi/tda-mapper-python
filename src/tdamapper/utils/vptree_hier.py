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
        pivoting=None
    ):
        self.__metric = metric
        self.__metric_params = metric_params
        self.__leaf_capacity = leaf_capacity
        self.__leaf_radius = leaf_radius
        self.__pivoting = pivoting
        tree, dataset = self._Build(self, X).build()
        self.__tree, self.__dataset = tree, dataset

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

    def _get_tree(self):
        return self.__tree

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
            if end <= start:
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
            if end <= start:
                return
            rnd = randrange(start, end)
            furthest_rnd = self._furthest(start, end, rnd)
            furthest = self._furthest(start, end, furthest_rnd)
            if furthest > start:
                _swap(self.__dataset, start, furthest)

        def _update(self, start, end):
            self.__pivoting(start, end)
            _, v_point = self.__dataset[start]
            for i in range(start + 1, end):
                _, point = self.__dataset[i]
                self.__dataset[i] = self.__distance(v_point, point), point

        def build(self):
            tree = self._build_rec(0, len(self.__dataset))
            return tree, self.__dataset

        def _build_rec(self, start, end):
            mid = _mid(start, end)
            self._update(start, end)
            _, v_point = self.__dataset[start]
            quickselect(self.__dataset, start + 1, end, mid)
            v_radius, _ = self.__dataset[mid]
            self.__dataset[start] = (v_radius, v_point)
            if (
                (end - start <= 2 * self.__leaf_capacity) or
                (v_radius <= self.__leaf_radius)
            ):
                left = _Leaf(start + 1, mid)
                right = _Leaf(mid, end)
            else:
                left = self._build_rec(start + 1, mid)
                right = self._build_rec(mid, end)
            return _Node(v_radius, v_point, left, right)

    def ball_search(self, point, eps, inclusive=True):
        return self._BallSearch(self, point, eps, inclusive).search()

    class _BallSearch:

        def __init__(self, vpt, point, eps, inclusive=True):
            self.__tree = vpt._get_tree()
            self.__dataset = vpt._get_dataset()
            self.__distance = vpt._get_distance()
            self.__point = point
            self.__eps = eps
            self.__inclusive = inclusive
            self.__result = []

        def search(self):
            self.__result.clear()
            self._search_rec(self.__tree)
            return self.__result

        def _inside(self, dist):
            if self.__inclusive:
                return dist <= self.__eps
            return dist < self.__eps

        def _search_rec(self, tree):
            if tree.is_terminal():
                start, end = tree.get_bounds()
                for _, x in self.__dataset[start:end]:
                    dist = self.__distance(self.__point, x)
                    if self._inside(dist):
                        self.__result.append(x)
            else:
                v_radius, v_point = tree.get_ball()
                dist = self.__distance(v_point, self.__point)
                if self._inside(dist):
                    self.__result.append(v_point)
                if dist <= v_radius:
                    fst, snd = tree.get_left(), tree.get_right()
                else:
                    fst, snd = tree.get_right(), tree.get_left()
                self._search_rec(fst)
                if abs(dist - v_radius) <= self.__eps:
                    self._search_rec(snd)

    def knn_search(self, point, k):
        return self._KnnSearch(self, point, k).search()

    class _KnnSearch:

        def __init__(self, vpt, point, neighbors):
            self.__tree = vpt._get_tree()
            self.__dataset = vpt._get_dataset()
            self.__distance = vpt._get_distance()
            self.__point = point
            self.__neighbors = neighbors
            self.__items = MaxHeap()

        def _add(self, dist, x):
            self.__items.add(dist, x)
            if len(self.__items) > self.__neighbors:
                self.__items.pop()

        def _get_items(self):
            while len(self.__items) > self.__neighbors:
                self.__items.pop()
            return [x for (_, x) in self.__items]

        def _get_radius(self):
            if len(self.__items) < self.__neighbors:
                return float('inf')
            furthest_dist, _ = self.__items.top()
            return furthest_dist

        def search(self):
            self._search_rec(self.__tree)
            return self._get_items()

        def _search_rec(self, tree):
            if tree.is_terminal():
                start, end = tree.get_bounds()
                for _, x in self.__dataset[start:end]:
                    dist = self.__distance(self.__point, x)
                    if dist < self._get_radius():
                        self._add(dist, x)
            else:
                v_radius, v_point = tree.get_ball()
                dist = self.__distance(v_point, self.__point)
                if dist < self._get_radius():
                    self._add(dist, v_point)
                if dist <= v_radius:
                    fst, snd = tree.get_left(), tree.get_right()
                else:
                    fst, snd = tree.get_right(), tree.get_left()
                self._search_rec(fst)
                if abs(dist - v_radius) <= self._get_radius():
                    self._search_rec(snd)


class _Node:

    def __init__(self, radius, center, left, right):
        self.__radius = radius
        self.__center = center
        self.__left = left
        self.__right = right

    def get_ball(self):
        return self.__radius, self.__center

    def is_terminal(self):
        return False

    def get_left(self):
        return self.__left

    def get_right(self):
        return self.__right


class _Leaf:

    def __init__(self, start, end):
        self.__start = start
        self.__end = end

    def get_bounds(self):
        return self.__start, self.__end

    def is_terminal(self):
        return True
