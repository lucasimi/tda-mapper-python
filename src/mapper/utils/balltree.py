"""A class for fast knn-like searches, depending only on a given metric"""
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

    def __init__(self, center, radius, elements=None):
        self.__center = center
        self.__radius = radius
        self.__elements = elements

    def get_radius(self):
        return self.__radius

    def get_center(self):
        return self.__center

    def get_elements(self):
        return self.__elements


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

class SearchResult:

    def __init__(self, point, error):
        self._point = point
        self._error = error

    def get_point(self):
        return self._point

    def get_error(self):
        return self._error

class BallTree:

    def __init__(self, distance, data, max_count=1, min_radius=None):
        self.__distance = distance
        dataset = [x for x in data]
        self.__tree = self._build(distance, dataset, 0, len(dataset), max_count, min_radius)

    def get_tree(self):
        return self.__tree

    def get_height(self):
        return self.__tree.get_height()

    def _build(self, dist, data, start, end, max_count=1, min_radius=None):
        if end - start <= max_count:
            return Tree(Ball(None, None, data[start:end]))
        else:
            center = random.choice(data[start:end]) #improve this by removing the copy
            mid = (end + start) // 2
            place_in_order(data, start, end, mid, lambda x: dist(center, x))
            radius = dist(center, data[mid])
            if min_radius and radius <= min_radius:
                left = Tree(Ball(None, None, data[start:mid]))
            else:
                left = self._build(dist, data, start, mid, max_count)
            right = self._build(dist, data, mid, end, max_count)
            return Tree(Ball(center, radius), left, right)
    

    def ball_search(self, point, eps):
        results = []
        self._ball_search(self.__distance, self.__tree, point, eps, results)
        return results

    def _ball_search(self, dist, tree, point, eps, results):
        if tree.is_terminal():
            ball = tree.get_data()
            #results.update([x for x in ball.get_elements() if dist(x, point) < eps])
            results.extend([x for x in ball.get_elements() if dist(x, point) < eps])
        else:
            left, right = tree.get_left(), tree.get_right()
            ball = tree.get_data()
            center, radius = ball.get_center(), ball.get_radius()
            d = dist(center, point)
            # the search ball B(point, eps) intersects B(center, radius) 
            if left and d <= radius + eps:
                self._ball_search(dist, left, point, eps, results)
            # the search ball B(point, eps) is not contained in B(center, radius) 
            if right and (eps > radius or d > radius - eps):
                self._ball_search(dist, right, point, eps, results)


    def nn_search(self, point):
        res = self._nn_search(self.__distance, self.__tree, point)
        return res.get_point()

    def _nn_search_inside(self, dist, tree, point, dist_center_point, radius):
        fst_res = self._nn_search(dist, tree.get_left(), point)
        fst_dist = fst_res.get_error()
        if dist_center_point + fst_dist <= radius:
            return fst_res
        else:
            snd_res = self._nn_search(dist, tree.get_right(), point)
            snd_dist = snd_res.get_error()
            return fst_res if fst_dist < snd_dist else snd_res
        
    def _nn_search_outside(self, dist, tree, point, dist_center_point, radius):
        fst_res = self._nn_search(dist, tree.get_right(), point)
        fst_dist = fst_res.get_error()
        if dist_center_point >= radius + fst_dist:
            return fst_res
        else:
            snd_res = self._nn_search(dist, tree.get_left(), point)
            snd_dist = snd_res.get_error()
            return fst_res if fst_dist < snd_dist else snd_res

    def _nn_search_all(self, dist, tree, point):
        ball = tree.get_data()
        min_dist = float('inf')
        best_fit = None
        for x in ball.get_elements():
            x_dist = dist(point, x)
            if x_dist < min_dist:
                min_dist = x_dist
                best_fit = x
        return SearchResult(best_fit, min_dist)

    def _nn_search(self, dist, tree, point):
        if tree.is_terminal():
            return self._nn_search_all(dist, tree, point)
        else:
            ball = tree.get_data()
            center, radius = ball.get_center(), ball.get_radius()
            dist_center_point = dist(center, point)
            if dist_center_point < radius:
                return self._nn_search_inside(dist, tree, point, dist_center_point, radius)
            else:
                return self._nn_search_outside(dist, tree, point, dist_center_point, radius)


    def knn_search(self, point, k):
        kball = KBall(self.__distance, point, k)
        self._knn_search(self.__distance, self.__tree, point, kball)
        ballheap = kball.get_heap()
        while len(ballheap) > k:
            ballheap.extract_max()
        return list(ballheap)

    def _knn_search(self, dist, tree, point, kball):
        if tree.is_terminal():
            self._knn_search_all(tree, kball)
        else:
            ball = tree.get_data()
            center, radius = ball.get_center(), ball.get_radius()
            dist_center_point = dist(center, point)
            if dist_center_point < radius:
                self._knn_search_inside(dist, tree, point, dist_center_point, radius, kball)
            else:
                self._knn_search_outside(dist, tree, point, dist_center_point, radius, kball)

    def _knn_search_all(self, tree, kball):
        ball = tree.get_data()
        kball.update(ball.get_elements())

    def _knn_search_inside(self, dist, tree, point, dist_center_point, radius, kball):
        self._knn_search(dist, tree.get_left(), point, kball)
        fst_dist = kball.get_radius()
        if dist_center_point + fst_dist <= radius:
            return
        else:
            self._knn_search(dist, tree.get_right(), point, kball)
    
    def _knn_search_outside(self, dist, tree, point, dist_center_point, radius, kball):
        self._knn_search(dist, tree.get_right(), point, kball)
        fst_dist = kball.get_radius()
        if dist_center_point >= radius + fst_dist:
            return
        else:
            self._knn_search(dist, tree.get_left(), point, kball)

def pivot_higher(data, start, end, i, fun=lambda x: x):
    """Move the elements less or equal than data[i] on the left, and higher on the right"""
    data[start], data[i] = data[i], data[start]
    higher = start + 1
    for j in range(start + 1, end):
        if fun(data[j]) <= fun(data[start]):
            data[higher], data[j] = data[j], data[higher]
            higher += 1
    data[start], data[higher - 1] = data[higher - 1], data[start]
    return higher - 1

def place_in_order(data, start, end, k, fun=lambda x: x):
    """Return the element which should fall at place k among [start:end]"""
    s_current, e_current = start, end
    higher = None
    while higher != k:
        higher = pivot_higher(data, s_current, e_current, k, fun)
        if k < higher:
            e_current = higher
        else:
            s_current = higher
