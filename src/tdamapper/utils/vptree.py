"""A class for fast knn and range searches, depending only on a given metric"""
from random import randrange
from tdamapper.utils.quickselect import quickselect_tuple
from tdamapper.utils.heap import MaxHeap


class VPTree:

    def __init__(self, distance, dataset, leaf_capacity=1, leaf_radius=0.0, pivoting=None):
        self.__distance = distance
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
        if end - start <= self.__leaf_capacity:
            return _Tree([x for _, x in self.__dataset[start:end]])
        mid = (end + start) // 2
        if update:
            self._update(start, end)
        _, v_point = self.__dataset[start]
        quickselect_tuple(self.__dataset, start + 1, end, mid)
        v_radius, _ = self.__dataset[mid]
        if v_radius <= self.__leaf_radius:
            left = _Tree([x for _, x in self.__dataset[start:mid]])
        else:
            left = self._build_rec(start, mid, False)
        right = self._build_rec(mid, end, True)
        return _Tree(_Ball(v_point, v_radius), left, right)

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
            search.process_all(tree.get_data())
        else:
            v_ball = tree.get_data()
            v_radius, v_point = v_ball.get_radius(), v_ball.get_center()
            point = search.get_center()
            dist = self.__distance(v_point, point)
            if dist <= v_radius:
                fst, snd = tree.get_left(), tree.get_right()
            else:
                fst, snd = tree.get_right(), tree.get_left()
            self._search_rec(fst, search)
            if abs(dist - v_radius) <= search.get_radius():
                self._search_rec(snd, search)


class _Ball:

    def __init__(self, center, radius):
        self.__center = center
        self.__radius = radius

    def get_radius(self):
        return self.__radius

    def get_center(self):
        return self.__center


class _Tree:

    def __init__(self, data, left=None, right=None):
        self.__data = data
        self.__left = left
        self.__right = right

    def get_data(self):
        return self.__data

    def get_left(self):
        return self.__left

    def get_right(self):
        return self.__right

    def is_terminal(self):
        return (self.__left is None) and (self.__right is None)

    def get_height(self):
        if self.__left is None:
            if self.__right is None:
                return 0
            return self.__right.get_height() + 1
        if self.__right is None:
            return self.__left.get_height() + 1
        l_height = self.__left.get_height()
        r_height = self.__right.get_height()
        return max(l_height, r_height) + 1


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

    def process_all(self, values):
        inside = [x for x in values if self.__inside(self._from_center(x))]
        self.__items.extend(inside)

    def _from_center(self, value):
        return self.__distance(self.__center, value)

    def _inside_inclusive(self, dist):
        return dist <= self.__radius

    def _inside_not_inclusive(self, dist):
        return dist < self.__radius


class _KNNSearch:

    def __init__(self, dist, center, neighbors):
        self.__dist = dist
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

    def _process(self, value):
        dist = self.__dist(self.__center, value)
        if dist >= self.get_radius():
            return
        self.__items.add(dist, value)
        if len(self.__items) > self.__neighbors:
            self.__items.pop()

    def process_all(self, values):
        for val in values:
            self._process(val)
