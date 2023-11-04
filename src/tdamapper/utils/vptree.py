"""A class for fast knn and range searches, depending only on a given metric"""
from .quickselect import quickselect_tuple
from .heap import MaxHeap


class VPTree:

    def __init__(self, distance, dataset, leaf_size=None, leaf_radius=None):
        self.__distance = distance
        self.__leaf_size = 1 if leaf_size is None else leaf_size
        self.__leaf_radius = float('inf') if leaf_radius is None else leaf_radius
        self.__dataset = [(0.0, x) for x in dataset]
        self.__tree = self._build_rec(0, len(self.__dataset), True)

    def _update(self, v_point, start, end):
        for i in range(start + 1, end):
            _, point = self.__dataset[i]
            self.__dataset[i] = self.__distance(v_point, point), point

    def _build_rec(self, start, end, update):
        if end - start <= self.__leaf_size:
            return _Tree([x for _, x in self.__dataset[start:end]])
        mid = (end + start) // 2
        _, v_point = self.__dataset[start]
        if update:
            self._update(v_point, start, end)
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
        ballheap = search.get_heap()
        while len(ballheap) > k:
            ballheap.pop()
        return [x for (_, x) in ballheap]

    def _search_rec(self, tree, search):
        if tree.is_terminal():
            search.process_all(tree.get_data())
        else:
            v_ball = tree.get_data()
            v_radius, v_point = v_ball.get_radius(), v_ball.get_center()
            point = search.get_center()
            dist = self.__distance(v_point, point)
            if dist < v_radius:
                fst, snd = tree.get_left(), tree.get_right()
            else:
                fst, snd = tree.get_right(), tree.get_left()
            self._search_rec(fst, search)
            if abs(dist - v_radius) < search.get_radius():
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

    def _process(self, value):
        dist = self.__dist(self.__center, value)
        if dist >= self.get_radius():
            return
        self.__items.add(dist, value)
        if len(self.__items) > self.__neighbors:
            self.__items.pop()

    def get_radius(self):
        if len(self.__items) < self.__neighbors:
            return float('inf')
        furthest_dist, _ = self.__items.top()
        return furthest_dist

    def get_center(self):
        return self.__center

    def get_heap(self):
        return self.__items

    def process_all(self, values):
        for val in values:
            self._process(val)
