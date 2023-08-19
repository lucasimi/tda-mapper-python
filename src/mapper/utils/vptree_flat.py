"""A class for fast knn and range searches, depending only on a given metric"""
from .quickselect import quickselect


class Ball:

    def __init__(self, center, radius):
        self.__center = center
        self.__radius = radius

    def get_radius(self):
        return self.__radius

    def get_center(self):
        return self.__center


class VPTree:

    def __init__(self, distance, dataset, leaf_size=1, leaf_radius=None):
        self.__distance = distance
        self.__leaf_size = leaf_size
        self.__leaf_radius = leaf_radius
        self.__dataset = [(0.0, x) for x in dataset]
        if not leaf_radius:
            self._build_update_no_radius(0, len(dataset))
        else:
            self._build_update_radius(0, len(dataset))

    def _build_update_no_radius(self, start, end):
        if end - start <= self.__leaf_size:
            return
        mid = (end + start) // 2
        _, vp = self.__dataset[start]
        for i in range(start + 1, end):
            _, x = self.__dataset[i]
            self.__dataset[i] = self.__distance(vp, x), x
        quickselect(self.__dataset, start + 1, end, mid,
                    lambda x: x[0])
        radius, _ = self.__dataset[mid]
        self.__dataset[start] = (radius, vp)
        self._build_update_no_radius(start + 1, mid)
        self._build_update_no_radius(mid, end)

    def _build_update_radius(self, start, end):
        if end - start <= self.__leaf_size:
            return
        mid = (end + start) // 2
        _, vp = self.__dataset[start]
        for i in range(start + 1, end):
            _, x = self.__dataset[i]
            self.__dataset[i] = self.__distance(vp, x), x
        quickselect(self.__dataset, start + 1, end, mid, lambda x: x[0])
        radius, _ = self.__dataset[mid]
        self.__dataset[start] = (radius, vp)
        if radius > self.__leaf_radius:
            self._build_update_radius(start + 1, mid)
        self._build_update_radius(mid, end)

    def ball_search(self, point, eps):
        results = []
        self._ball_search(point, eps, results, 0, len(self.__dataset))
        return results

    def _ball_search(self, point, eps, results, start, end):
        if end - start <= self.__leaf_size:
            sliced = [x for (_, x) in self.__dataset[start:end]]
            partial = [x for x in sliced if self.__distance(x, point) < eps]
            results.extend(partial)
        else:
            radius, center = self.__dataset[start]
            d = self.__distance(center, point)
            mid = (end + start) // 2
            if d < eps: 
                results.append(center)
            # the search ball B(point, eps) intersects B(center, radius)
            if d <= radius + eps:
                self._ball_search(point, eps, results, start + 1, mid)
            # the search ball B(point, eps) is not contained in B(center, radius)
            if eps > radius or d > radius - eps:
                self._ball_search(point, eps, results, mid, end)
