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
            tree = self._build_iter()
            return tree, self.__dataset

        def _build_iter(self):
            root = _Node(None, None, None, None)
            stack = [(0, len(self.__dataset), root, 0)]
            while stack:
                start, end, parent, side = stack.pop()
                mid = _mid(start, end)
                self._update(start, end)
                _, v_point = self.__dataset[start]
                quickselect(self.__dataset, start + 1, end, mid)
                v_radius, _ = self.__dataset[mid]
                self.__dataset[start] = (v_radius, v_point)
                if (end - start > 2 * self.__leaf_capacity) and (v_radius > self.__leaf_radius):
                    node = _Node(v_radius, v_point, None, None)
                    stack.append((mid, end, node, 1))
                    stack.append((start + 1, mid, node, 0))
                else:
                    node = _Leaf(start, end)
                if side == 0:
                    parent._set_left(node)
                elif side == 1:
                    parent._set_right(node)
            return root.get_left()

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

        def search(self):
            return self._search_iter()

        def _inside(self, dist):
            if self.__inclusive:
                return dist <= self.__eps
            return dist < self.__eps

        def _search_iter(self):
            stack = [self.__tree]
            result = []
            while stack:
                tree = stack.pop()
                if tree.is_terminal():
                    start, end = tree.get_bounds()
                    for _, x in self.__dataset[start:end]:
                        dist = self.__distance(self.__point, x)
                        if self._inside(dist):
                            result.append(x)
                else:
                    v_radius, v_point = tree.get_ball()
                    dist = self.__distance(self.__point, v_point)
                    if self._inside(dist):
                        result.append(v_point)
                    if dist <= v_radius:
                        fst, snd = tree.get_left(), tree.get_right()
                    else:
                        fst, snd = tree.get_right(), tree.get_left()
                    if abs(dist - v_radius) <= self.__eps:
                        stack.append(snd)
                    stack.append(fst)
            return result

    def knn_search(self, point, k):
        return self._KnnSearch(self, point, k).search()

    class _KnnSearch:

        def __init__(self, vpt, point, neighbors):
            self.__tree = vpt._get_tree()
            self.__dataset = vpt._get_dataset()
            self.__distance = vpt._get_distance()
            self.__point = point
            self.__neighbors = neighbors
            self.__result = MaxHeap()

        def _add(self, dist, x):
            self.__result.add(dist, x)
            if len(self.__result) > self.__neighbors:
                self.__result.pop()

        def _get_items(self):
            while len(self.__result) > self.__neighbors:
                self.__result.pop()
            return [x for (_, x) in self.__result]

        def _get_radius(self):
            if len(self.__result) < self.__neighbors:
                return float('inf')
            furthest_dist, _ = self.__result.top()
            return furthest_dist

        def search(self):
            self._search_iter()
            return self._get_items()

        def _process(self, x):
            dist = self.__distance(self.__point, x)
            if dist >= self._get_radius():
                return dist
            self.__result.add(dist, x)
            while len(self.__result) > self.__neighbors:
                self.__result.pop()
            return dist

        def _pre(self, tree, stack):
            v_radius, v_point = tree.get_ball()
            dist = self._process(v_point)
            if dist <= v_radius:
                fst, snd = tree.get_left(), tree.get_right()
            else:
                fst, snd = tree.get_right(), tree.get_left()
            stack.append((snd, dist, v_radius, 1))
            stack.append((fst, None, None, 0))

        def _post(self, tree, dist, v_radius, stack):
            if abs(dist - v_radius) <= self._get_radius():
                stack.append((tree, None, None, 0))

        def _search_iter(self):
            self.__result = MaxHeap()
            stack = [(self.__tree, None, None, 0)]
            while stack:
                tree, dist, v_radius, after = stack.pop()
                if tree.is_terminal():
                    start, end = tree.get_bounds()
                    for _, x in self.__dataset[start:end]:
                        self._process(x)
                else:
                    if after == 0:
                        self._pre(tree, stack)
                    elif after == 1:
                        self._post(tree, dist, v_radius, stack)
            return self._get_items()


class _Node:

    def __init__(self, radius, center, left, right):
        self.__radius = radius
        self.__center = center
        self.__left = left
        self.__right = right

    def _set_left(self, left):
        self.__left = left

    def _set_right(self, right):
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
