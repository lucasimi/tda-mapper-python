"""A class for fast knn and range searches, depending only on a given metric"""
import random

from .heap import MaxHeap


class KBall:

    def __init__(self, dist, center, k):
        self.__dist = dist
        self.__center = center
        self.__k = k
        self.__heap = MaxHeap(fun=lambda x: self.__dist(x, self.__center))
    
    def insert(self, x):
        dist_x = self.__dist(self.__center, x)
        radius = self.get_radius()
        if dist_x >= radius:
            return
        else:
            self.__heap.insert(x)
            if len(self.__heap) > self.__k:
                self.__heap.extract_max()

    def get_radius(self):
        if len(self.__heap) < self.__k:
            return float('inf')
        else:
            furthest = self.__heap.max()
            return self.__dist(self.__center, furthest)

    def get_center(self):
        return self.__center

    def get_heap(self):
        return self.__heap

    def update(self, values):
        for x in values:
            self.insert(x)


class Ball:

    def __init__(self, center, radius):
        self.__center = center
        self.__radius = radius

    def get_radius(self):
        return self.__radius

    def get_center(self):
        return self.__center


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
            else:
                return self.__right.get_height() + 1
        else:
            if self.__right is None:
                return self.__left.get_height() + 1
            else:
                a = self.__left.get_height()
                b = self.__right.get_height()
                return max(a, b) + 1


class VPTree:

    def __init__(self, distance, dataset, leaf_size=1, leaf_radius=None):
        self.__distance = distance
        self.__leaf_size = leaf_size
        self.__leaf_radius = leaf_radius
        self.__dataset = [x for x in dataset]
        self.__tree = self._build(0, len(dataset))

    def get_tree(self):
        return self.__tree

    def get_height(self):
        return self.__tree.get_height()

    def _build(self, start, end):
        if end - start <= self.__leaf_size:
            return Tree(self.__dataset[start:end])
        else:
            center = random.choice(self.__dataset[start:end]) #improve this by removing the copy
            mid = (end + start) // 2
            _place_in_order(self.__dataset, start, end, mid, lambda x: self.__distance(center, x))
            radius = self.__distance(center, self.__dataset[mid])
            if self.__leaf_radius and radius <= self.__leaf_radius:
                left = Tree(self.__dataset[start:mid])
            else:
                left = self._build(start, mid)
            right = self._build(mid, end)
            return Tree(Ball(center, radius), left, right)
    
    def ball_search(self, point, eps):
        results = []
        self._ball_search(self.__tree, point, eps, results)
        return results

    def _ball_search(self, tree, point, eps, results):
        if tree.is_terminal():
            ball = tree.get_data()
            results.extend([x for x in ball if self.__distance(x, point) < eps])
        else:
            left, right = tree.get_left(), tree.get_right()
            ball = tree.get_data()
            center, radius = ball.get_center(), ball.get_radius()
            d = self.__distance(center, point)
            # the search ball B(point, eps) intersects B(center, radius) 
            if left and d <= radius + eps:
                self._ball_search(left, point, eps, results)
            # the search ball B(point, eps) is not contained in B(center, radius) 
            if right and (eps > radius or d > radius - eps):
                self._ball_search(right, point, eps, results)

    def knn_search(self, point, k):
        kball = KBall(self.__distance, point, k)
        self._knn_search(self.__tree, point, kball)
        ballheap = kball.get_heap()
        while len(ballheap) > k:
            ballheap.extract_max()
        return list(ballheap)

    def _knn_search(self, tree, point, kball):
        if tree.is_terminal():
            self._knn_search_all(tree, kball)
        else:
            ball = tree.get_data()
            center, radius = ball.get_center(), ball.get_radius()
            dist_center_point = self.__distance(center, point)
            if dist_center_point < radius:
                self._knn_search_inside(tree, point, dist_center_point, radius, kball)
            else:
                self._knn_search_outside(tree, point, dist_center_point, radius, kball)

    def _knn_search_all(self, tree, kball):
        ball = tree.get_data()
        kball.update(ball)

    def _knn_search_inside(self, tree, point, dist_center_point, radius, kball):
        self._knn_search(tree.get_left(), point, kball)
        fst_dist = kball.get_radius()
        if dist_center_point + fst_dist <= radius:
            return
        else:
            self._knn_search(tree.get_right(), point, kball)
    
    def _knn_search_outside(self, tree, point, dist_center_point, radius, kball):
        self._knn_search(tree.get_right(), point, kball)
        fst_dist = kball.get_radius()
        if dist_center_point >= radius + fst_dist:
            return
        else:
            self._knn_search(tree.get_left(), point, kball)


def _pivot_higher(data, start, end, i, fun=lambda x: x):
    """Move the elements less or equal than data[i] on the left, and higher on the right"""
    data[start], data[i] = data[i], data[start]
    higher = start + 1
    for j in range(start + 1, end):
        if fun(data[j]) <= fun(data[start]):
            data[higher], data[j] = data[j], data[higher]
            higher += 1
    data[start], data[higher - 1] = data[higher - 1], data[start]
    return higher - 1


def _place_in_order(data, start, end, k, fun=lambda x: x):
    """Return the element which should fall at place k among [start:end]"""
    s_current, e_current = start, end
    higher = None
    while higher != k:
        higher = _pivot_higher(data, s_current, e_current, k, fun)
        if k < higher:
            e_current = higher
        else:
            s_current = higher

