"""A class for fast knn and range searches, depending only on a given metric"""
import random

from .quickselect import quickselect
from .heap import MaxHeap


class KBall:

    def __init__(self, dist, center, k):
        self.__dist = dist
        self.__center = center
        self.__k = k
        self.__heap = MaxHeap(fun=lambda x: self.__dist(x, self.__center))

    def insert(self, data):
        dist_x = self.__dist(self.__center, data)
        radius = self.get_radius()
        if dist_x >= radius:
            return
        self.__heap.insert(data)
        if len(self.__heap) > self.__k:
            self.__heap.extract_max()

    def get_radius(self):
        if len(self.__heap) < self.__k:
            return float('inf')
        furthest = self.__heap.max()
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

    def __init__(self, distance, dataset, leaf_size=1, leaf_radius=None):
        self.__distance = distance
        self.__leaf_size = leaf_size
        self.__leaf_radius = leaf_radius
        self.__dataset = [(0.0, x) for x in dataset]
        if not leaf_radius:
            self.__tree = self._build_update_no_radius(0, len(dataset))
        else:
            self.__tree = self._build_update_radius(0, len(dataset))

    def get_tree(self):
        return self.__tree

    def get_height(self):
        return self.__tree.get_height()

    def _build_update_no_radius(self, start, end):
        if end - start <= self.__leaf_size:
            return Tree([x for _, x in self.__dataset[start:end]])
        mid = (end + start) // 2
        _, vp = self.__dataset[start]
        for i in range(start + 1, end):
            _, x = self.__dataset[i]
            self.__dataset[i] = self.__distance(vp, x), x
        quickselect(self.__dataset, start + 1, end, mid,
                    lambda x: x[0])
        radius, _ = self.__dataset[mid]
        left = self._build_no_update_no_radius(start, mid)
        right = self._build_update_no_radius(mid, end)
        return Tree(Ball(vp, radius), left, right)

    def _build_no_update_no_radius(self, start, end):
        if end - start <= self.__leaf_size:
            return Tree([x for _, x in self.__dataset[start:end]])
        mid = (end + start) // 2
        _, vp = self.__dataset[start]
        quickselect(self.__dataset, start + 1, end, mid, lambda x: x[0])
        radius, _ = self.__dataset[mid]
        left = self._build_no_update_no_radius(start, mid)
        right = self._build_update_no_radius(mid, end)
        return Tree(Ball(vp, radius), left, right)

    def _build_update_radius(self, start, end):
        if end - start <= self.__leaf_size:
            return Tree([x for _, x in self.__dataset[start:end]])
        mid = (end + start) // 2
        _, vp = self.__dataset[start]
        for i in range(start + 1, end):
            _, x = self.__dataset[i]
            self.__dataset[i] = self.__distance(vp, x), x
        quickselect(self.__dataset, start + 1, end, mid, lambda x: x[0])
        radius, _ = self.__dataset[mid]
        if radius <= self.__leaf_radius:
            left = Tree([x for _, x in self.__dataset[start:mid]])
        else:
            left = self._build_no_update_radius(start, mid)
        right = self._build_update_radius(mid, end)
        return Tree(Ball(vp, radius), left, right)

    def _build_no_update_radius(self, start, end):
        if end - start <= self.__leaf_size:
            return Tree([x for _, x in self.__dataset[start:end]])
        mid = (end + start) // 2
        _, vp = self.__dataset[start]
        quickselect(self.__dataset, start + 1, end, mid, lambda x: x[0])
        radius, _ = self.__dataset[mid]
        if radius <= self.__leaf_radius:
            left = Tree([x for _, x in self.__dataset[start:mid]])
        else:
            left = self._build_no_update_radius(start, mid)
        right = self._build_update_radius(mid, end)
        return Tree(Ball(vp, radius), left, right)

    def ball_search(self, point, eps):
        results = []
        self._ball_search(self.__tree, point, eps, results)
        return results

    def _ball_search(self, tree, point, eps, results):
        if tree.is_terminal():
            ball = tree.get_data()
            results.extend(
                [x for x in ball if self.__distance(x, point) < eps])
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
                self._knn_search_inside(
                    tree, point, dist_center_point, radius, kball)
            else:
                self._knn_search_outside(
                    tree, point, dist_center_point, radius, kball)

    def _knn_search_all(self, tree, kball):
        ball = tree.get_data()
        kball.update(ball)

    def _knn_search_inside(self, tree, point, dist_center_point, radius, kball):
        self._knn_search(tree.get_left(), point, kball)
        fst_dist = kball.get_radius()
        if dist_center_point + fst_dist <= radius:
            return
        self._knn_search(tree.get_right(), point, kball)

    def _knn_search_outside(self, tree, point, dist_center_point, radius, kball):
        self._knn_search(tree.get_right(), point, kball)
        fst_dist = kball.get_radius()
        if dist_center_point >= radius + fst_dist:
            return
        self._knn_search(tree.get_left(), point, kball)
