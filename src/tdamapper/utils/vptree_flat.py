"""A class for fast knn and range searches, depending only on a given metric"""
from .quickselect import quickselect

class SearchResults:

    def __init__(self, dist, center, max_radius, max_neighbors):
        self.__dist = dist
        self.__center = center
        self.__max_radius = max_radius
        self.__max_neighbors = max_neighbors
        self.__heap = MaxHeap()

    def extract():
        results = []
        while len(self.__heap) > 0:
            results.append(self.__heap.pop())
        return results

    def get_radius(self):
        if len(self.__heap) < self.__max_neighbors:
            return self.__max_radius
        _, furthest = self.__heap.top()
        return self.__dist(furthest, self.__center)

    def add(self, value):
        dist = self.__dist(self.__center, value)
        if dist <= self.__max_radius: # TODO: inclusive or not?
            self.__heap.add((dist, value))
            while len(self.__heap) > self.__max_neighbors:
                self.__heap.pop()

    


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
                        lambda x: x[0]) # TODO: optimized version of quickselect
            radius, _ = self.__dataset[mid]
            self.__dataset[start] = (radius, vp)
            if end - mid > self.__leaf_size:
                stack.append((mid, end))
            if (mid - start - 1 > self.__leaf_size) and (radius > self.__leaf_radius):
                stack.append((start + 1, mid))

    def ball_search(self, point, eps):
        return self._ball_search(point, eps)

    def _ball_search(self, point, eps):
        results = []
        stack = [(0, len(self.__dataset))]
        while stack:
            (start, end) = stack.pop()
            if end - start <= self.__leaf_size:
                sliced = [x for (_, x) in self.__dataset[start:end]]
                partial = [x for x in sliced if self.__distance(x, point) < eps] # TODO: inclusive?
                results.extend(partial)
            else:
                radius, center = self.__dataset[start]
                d = self.__distance(center, point)
                mid = (end + start) // 2
                if d < eps: 
                    results.append(center)
                # the search ball B(point, eps) intersects B(center, radius)
                if d <= radius + eps:
                    stack.append((start + 1, mid))
                # the search ball B(point, eps) is not contained in B(center, radius)
                if eps > radius or d > radius - eps:
                    stack.append((mid, end))
        return results
