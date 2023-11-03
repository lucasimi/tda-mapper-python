"""A class for fast knn and range searches, depending only on a given metric"""
from .quickselect import quickselect_tuple
from .heap import MaxHeap


class KBall:

    def __init__(self, dist, center, k):
        self.__dist = dist
        self.__center = center
        self.__k = k
        self.__heap = MaxHeap()

    def insert(self, data):
        dist = self.__dist(self.__center, data)
        radius = self.get_radius()
        if dist >= radius:
            return
        self.__heap.add(dist, data)
        if len(self.__heap) > self.__k:
            self.__heap.pop()

    def get_radius(self):
        if len(self.__heap) < self.__k:
            return float('inf')
        _, furthest = self.__heap.top()
        return self.__dist(self.__center, furthest)

    def get_center(self):
        return self.__center

    def get_heap(self):
        return self.__heap

    def update(self, values):
        for val in values:
            self.insert(val)


class Ball:

    def __init__(self, center, radius):
        self.__center = center
        self.__radius = radius

    def get_radius(self):
        return self.__radius

    def get_center(self):
        return self.__center


class BallSearch:

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

    def process(self, value):
        dist = self._from_center(value)
        if self.__inside(dist):
            self.__items.append(value)
        return dist

    def _from_center(self, value):
        return self.__distance(self.__center, value)

    def _inside_inclusive(self, dist):
        return dist <= self.__radius

    def _inside_not_inclusive(self, dist):
        return dist < self.__radius


class Tree:

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


class VPTree:

    def __init__(self, distance, dataset, leaf_size=None, leaf_radius=None):
        self.__distance = distance
        self.__leaf_size = 1 if leaf_size is None else leaf_size
        self.__leaf_radius = float('inf') if leaf_radius is None else leaf_radius
        self.__dataset = [(0.0, x) for x in dataset]
        self.__tree = self._build_rec(0, len(self.__dataset), True)

    def get_tree(self):
        return self.__tree

    def get_height(self):
        return self.__tree.get_height()

    def _update(self, v_point, start, end):
        for i in range(start + 1, end):
            _, point = self.__dataset[i]
            self.__dataset[i] = self.__distance(v_point, point), point

    def _build_rec(self, start, end, update):
        if end - start <= self.__leaf_size:
            return Tree([x for _, x in self.__dataset[start:end]])
        mid = (end + start) // 2
        _, v_point = self.__dataset[start]
        if update:
            self._update(v_point, start, end)
        quickselect_tuple(self.__dataset, start + 1, end, mid)
        v_radius, _ = self.__dataset[mid]
        if v_radius <= self.__leaf_radius:
            left = Tree([x for _, x in self.__dataset[start:mid]])
        else:
            left = self._build_rec(start, mid, False)
        right = self._build_rec(mid, end, True)
        return Tree(Ball(v_point, v_radius), left, right)

    def ball_search(self, point, eps, inclusive=True):
        search = BallSearch(self.__distance, point, eps, inclusive)
        self._ball_search_rec(self.__tree, search)
        return search.get_items()

    def _ball_search_rec(self, tree, search):
        if tree.is_terminal():
            search.process_all(tree.get_data())
        else:
            left, right = tree.get_left(), tree.get_right()
            v_ball = tree.get_data()
            v_point, v_radius = v_ball.get_center(), v_ball.get_radius()
            point = search.get_center()
            eps = search.get_radius()
            dist = self.__distance(v_point, point)
            if left and (dist <= v_radius + eps): # search intersects B(center, radius)
                self._ball_search_rec(left, search)
            if right and (dist > v_radius - eps): # search is not contained in B(center, radius)
                self._ball_search_rec(right, search)

    def knn_search(self, point, k):
        kball = KBall(self.__distance, point, k)
        self._knn_search_rec(self.__tree, point, kball)
        ballheap = kball.get_heap()
        while len(ballheap) > k:
            ballheap.pop()
        return [x for (_, x) in ballheap]

    def _knn_search_rec(self, tree, point, kball):
        if tree.is_terminal():
            self._knn_search_all(tree, kball)
        else:
            ball = tree.get_data()
            center, radius = ball.get_center(), ball.get_radius()
            dist_center_point = self.__distance(center, point)
            if dist_center_point < radius:
                self._knn_search_inside(
                    tree, point, dist_center_point, radius, kball)
            else:
                self._knn_search_outside(
                    tree, point, dist_center_point, radius, kball)

    def _knn_search_all(self, tree, kball):
        ball = tree.get_data()
        kball.update(ball)

    def _knn_search_inside(self, tree, point, dist_center_point, radius, kball):
        self._knn_search_rec(tree.get_left(), point, kball)
        fst_dist = kball.get_radius()
        if dist_center_point + fst_dist <= radius:
            return
        self._knn_search_rec(tree.get_right(), point, kball)

    def _knn_search_outside(self, tree, point, dist_center_point, radius, kball):
        self._knn_search_rec(tree.get_right(), point, kball)
        fst_dist = kball.get_radius()
        if dist_center_point >= radius + fst_dist:
            return
        self._knn_search_rec(tree.get_left(), point, kball)
