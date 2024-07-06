"""A class for fast knn and range searches, depending only on a given metric"""
from random import randrange

from tdamapper.utils.metrics import get_metric
from tdamapper.utils.quickselect import quickselect
from tdamapper.utils.heap import MaxHeap


class VPTree:

    def __init__(self, distance, dataset, leaf_capacity=1, leaf_radius=0.0, pivoting=None, **kwargs):
        self.__distance = get_metric(distance, **kwargs)
        self.__dataset = [(0.0, x) for x in dataset]
        self.__leaf_capacity = leaf_capacity
        self.__leaf_radius = leaf_radius
        self.__pivoting = self._pivoting_disabled
        if pivoting == 'random':
            self.__pivoting = self._pivoting_random
        elif pivoting == 'furthest':
            self.__pivoting = self._pivoting_furthest
        self.__tree = self._build_rec(0, len(self.__dataset), True)

    def _pivoting_disabled(self, start, end):
        pass

    def _pivoting_random(self, start, end):
        pivot = randrange(start, end)
        if pivot > start:
            self.__dataset[start], self.__dataset[pivot] = self.__dataset[pivot], self.__dataset[start]

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
        rnd = randrange(start, end)
        furthest_rnd = self._furthest(start, end, rnd)
        furthest = self._furthest(start, end, furthest_rnd)
        if furthest > start:
            self.__dataset[start], self.__dataset[furthest] = self.__dataset[furthest], self.__dataset[start]

    def _update(self, start, end):
        self.__pivoting(start, end)
        _, v_point = self.__dataset[start]
        for i in range(start + 1, end):
            _, point = self.__dataset[i]
            self.__dataset[i] = self.__distance(v_point, point), point

    def _build_rec(self, start, end, update):
        mid = (end + start) // 2
        if update:
            self._update(start, end)
        _, v_point = self.__dataset[start]
        quickselect(self.__dataset, start + 1, end, mid)
        v_radius, _ = self.__dataset[mid]
        if (end - start <= 2 * self.__leaf_capacity) or (v_radius <= self.__leaf_radius):
            left = _Leaf(start, mid)
            right = _Leaf(mid, end)
        else:
            left = self._build_rec(start, mid, False)
            right = self._build_rec(mid, end, True)
        return _Node(v_radius, v_point, left, right)

    def ball_search(self, point, eps, inclusive=True):
        search = _BallSearch(self.__distance, point, eps, inclusive)
        self._search_rec(self.__tree, search)
        return search.get_items()

    def knn_search(self, point, k):
        search = _KNNSearch(self.__distance, point, k)
        self._search_rec(self.__tree, search)
        return search.get_items()

    def _search_rec(self, tree, search):
        if tree.is_terminal():
            start, end = tree.get_bounds()
            search.process_all(self.__dataset, start, end)
        else:
            v_radius, v_point = tree.get_ball()
            point = search.get_center()
            dist = self.__distance(v_point, point)
            if dist <= v_radius:
                fst, snd = tree.get_left(), tree.get_right()
            else:
                fst, snd = tree.get_right(), tree.get_left()
            self._search_rec(fst, search)
            if abs(dist - v_radius) <= search.get_radius():
                self._search_rec(snd, search)


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


class _BallSearch:

    def __init__(self, distance, center, radius, inclusive):
        self.__distance = distance
        self.__center = center
        self.__radius = radius
        self.__items = []
        self.__inside = self._inside_inclusive if inclusive else self._inside_not_inclusive

    def get_items(self):
        return self.__items

    def get_radius(self):
        return self.__radius

    def get_center(self):
        return self.__center

    def process_all(self, data, start, end):
        for _, p in data[start:end]:
            if self.__inside(self._dist_from_center(p)):
                self.__items.append(p)

    def _dist_from_center(self, p):
        return self.__distance(self.__center, p)

    def _inside_inclusive(self, dist):
        return dist <= self.__radius

    def _inside_not_inclusive(self, dist):
        return dist < self.__radius


class _KNNSearch:

    def __init__(self, distance, center, neighbors):
        self.__distance = distance
        self.__center = center
        self.__neighbors = neighbors
        self.__items = MaxHeap()

    def get_items(self):
        while len(self.__items) > self.__neighbors:
            self.__items.pop()
        return [x for (_, x) in self.__items]

    def get_radius(self):
        if len(self.__items) < self.__neighbors:
            return float('inf')
        furthest_dist, _ = self.__items.top()
        return furthest_dist

    def get_center(self):
        return self.__center

    def _dist_from_center(self, value):
        return self.__distance(self.__center, value)

    def process_all(self, data, start, end):
        for _, p in data[start:end]:
            dist = self._dist_from_center(p)
            if dist < self.get_radius():
                self.__items.add(dist, p)
                if len(self.__items) > self.__neighbors:
                    self.__items.pop()
