"""A class for fast knn and range searches, depending only on a given metric"""
from random import randrange

from tdamapper.utils.quickselect import quickselect_tuple
from tdamapper.utils.heap import MaxHeap
from tdamapper.utils.metrics import get_metric


class VPTree:

    def __init__(self, 
            metric='euclidean',
            leaf_capacity=1,
            leaf_radius=0.0,
            strategy='random'):
        self.metric = metric
        self.leaf_capacity = leaf_capacity
        self.leaf_radius = leaf_radius
        self.strategy = strategy

    def fit(self, X):
        self.__metric = get_metric(self.metric)
        self.__arr = [(0.0, x) for x in X]
        self.__capacity = self.leaf_capacity
        self.__radius = self.leaf_radius
        self.__strategy = self._strategy()
        self.__tree = self._build_rec(0, len(self.__arr), True)

    def ball_search(self, point, eps=0.5, inclusive=True):
        search = _BallSearch(self.__metric, point, eps, inclusive)
        self._search_rec(self.__tree, search)
        return search.get_items()

    def knn_search(self, point, k=1):
        search = _KNNSearch(self.__metric, point, k)
        self._search_rec(self.__tree, search)
        return search.get_items()

    def _strategy(self):
        if self.strategy == 'random':
            return self._strategy_random
        elif self.strategy == 'furthest':
            return self._strategy_furthest
        elif self.strategy == 'fixed':
            return self._strategy_fixed

    def _strategy_fixed(self, start, end):
        pass

    def _strategy_random(self, start, end):
        pivot = randrange(start, end)
        if pivot > start:
            self.__arr[start], self.__arr[pivot] = self.__arr[pivot], self.__arr[start]

    def _strategy_furthest(self, start, end):
        rnd = randrange(start, end)
        furthest_rnd = self._furthest(start, end, rnd)
        furthest = self._furthest(start, end, furthest_rnd)
        if furthest > start:
            self.__arr[start], self.__arr[furthest] = self.__arr[furthest], self.__arr[start]

    def _furthest(self, start, end, i):
        furthest_dist = 0.0
        furthest = start
        _, i_point = self.__arr[i]
        for j in range(start, end):
            _, j_point = self.__arr[j]
            j_dist = self.__metric(i_point, j_point)
            if j_dist > furthest_dist:
                furthest = j
                furthest_dist = j_dist
        return furthest

    def _update_arr(self, start, end):
        self.__strategy(start, end)
        _, v_point = self.__arr[start]
        for i in range(start + 1, end):
            _, point = self.__arr[i]
            self.__arr[i] = self.__metric(v_point, point), point

    def _build_rec(self, start, end, update):
        if end - start <= self.__capacity:
            return _Tree([x for _, x in self.__arr[start:end]])
        mid = (end + start) // 2
        if update:
            self._update_arr(start, end)
        _, v_point = self.__arr[start]
        quickselect_tuple(self.__arr, start + 1, end, mid)
        v_radius, _ = self.__arr[mid]
        if v_radius <= self.__radius:
            left = _Tree([x for _, x in self.__arr[start:mid]])
        else:
            left = self._build_rec(start, mid, False)
        right = self._build_rec(mid, end, True)
        return _Tree(_Ball(v_point, v_radius), left, right)

    def _search_rec(self, tree, search):
        if tree.is_terminal():
            search.process_all(tree.get_data())
        else:
            v_ball = tree.get_data()
            v_radius, v_point = v_ball.get_radius(), v_ball.get_center()
            point = search.get_center()
            dist = self.__metric(v_point, point)
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
