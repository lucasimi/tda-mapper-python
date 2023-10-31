"""A class for fast knn and range searches, depending only on a given metric"""
from .quickselect import quickselect
from .heap import MaxHeap


class KNNResults:

    def __init__(self, dist, center, neighbors):
        self.__dist = dist
        self.__center = center
        self.__neighbors = neighbors
        self.__items = MaxHeap()

    def __iter__(self):
        return iter(self.__items)

    def __next__(self):
        return next(self.__items)

    def extract(self):
        results = []
        while len(self.__items) > 0:
            results.append(self.__items.pop())
        return results

    def get_radius(self):
        if len(self.__items) < self.__neighbors:
            return float('inf')
        _, furthest = self.__items.top()
        return self.__dist(furthest, self.__center)

    def add(self, value):
        dist = self.__dist(self.__center, value)
        self.__items.add((dist, value))
        while len(self.__items) > self.__neighbors:
            self.__items.pop()
        return dist


class BallResults:

    def __init__(self, dist, center, radius):
        self.__dist = dist
        self.__center = center
        self.__radius = radius
        self.__items = []

    def extract(self):
        return self.__items

    def get_radius(self):
        return self.__radius

    def add(self, value):
        dist = self.__dist(self.__center, value)
        if dist <= self.__radius: #TODO: inclusive or not?
            self.__items.append(value)
        return dist


class VPTree:

    def __init__(self, distance, dataset, leaf_size=None, leaf_radius=None):
        self.__distance = distance
        self.__leaf_size = 1 if leaf_size is None else leaf_size
        self.__leaf_radius = float('inf') if leaf_radius is None else leaf_radius
        self.__dataset = [(0.0, x) for x in dataset]
        self._build_update_iter()

    def _build_update_iter(self):
        stack = []
        if len(self.__dataset) > self.__leaf_size:
            stack.append((0, len(self.__dataset)))
        while stack:
            (start, end) = stack.pop()
            mid = (end + start) // 2
            _, vp = self.__dataset[start]
            for i in range(start + 1, end):
                _, x = self.__dataset[i]
                self.__dataset[i] = self.__distance(vp, x), x
            quickselect(self.__dataset, start + 1, end, mid,
                        lambda x: x[0]) #TODO: optimized version of quickselect
            radius, _ = self.__dataset[mid]
            self.__dataset[start] = (radius, vp)
            if end - mid > self.__leaf_size:
                stack.append((mid, end))
            if (mid - start - 1 > self.__leaf_size) and (radius > self.__leaf_radius):
                stack.append((start + 1, mid))

    def knn_search(self, point, neighbors):
        results = KNNResults(self.__distance, point, neighbors)
        return self._knn_search(point, neighbors, 0, len(self.__dataset), results)

    def _knn_search(self, point, neighbors, start, end, results):
        if end - start <= self.__leaf_size:
            for _, itm in self.__dataset[start:end]:
                results.add(itm)
        else:
            radius, center = self.__dataset[start]
            mid = (end + start) // 2
            dist = results.add(center)
            self._knn_search(point, neighbors, start + 1, mid, results)
            neigh = 0
            for res in results:
                if self.__distance(res, center) < abs(radius - dist):
                    neigh += 1
            if neigh < neighbors:
                self._knn_search(point, neighbors - neigh, mid, end, results)
        return results

    def ball_search(self, point, eps):
        results = []
        stack = [(0, len(self.__dataset))]
        while stack:
            (start, end) = stack.pop()
            if end - start <= self.__leaf_size:
                partial = [x for _, x in self.__dataset[start:end]
                    if self.__distance(x, point) < eps] # TODO: inclusive?
                results.extend(partial)
            else:
                radius, center = self.__dataset[start]
                dist = self.__distance(center, point)
                mid = (end + start) // 2
                if dist <= eps: #TODO: inclusive?
                    results.append(center)
                if dist <= radius + eps:    # results intersects B(center, radius)
                    stack.append((start + 1, mid))
                if dist > radius - eps:     # results is not contained in B(center, radius)
                    stack.append((mid, end))
        return results
