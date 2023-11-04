"""A class for fast knn and range searches, depending only on a given metric"""
from .quickselect import quickselect_tuple
from .heap import MaxHeap


class VPTree:

    def __init__(self, distance, dataset, leaf_size=None, leaf_radius=None):
        self.__distance = distance
        self.__leaf_size = 1 if leaf_size is None else leaf_size
        self.__leaf_radius = float('inf') if leaf_radius is None else leaf_radius
        self.__dataset = [(0.0, x) for x in dataset]
        self._build_iter()

    def _update(self, v_point, start, end):
        for i in range(start + 1, end):
            _, point = self.__dataset[i]
            self.__dataset[i] = self.__distance(v_point, point), point

    def _build_iter(self):
        stack = []
        if len(self.__dataset) > self.__leaf_size:
            stack.append((0, len(self.__dataset)))
        while stack:
            start, end = stack.pop()
            mid = (end + start) // 2
            _, v_point = self.__dataset[start]
            self._update(v_point, start, end)
            quickselect_tuple(self.__dataset, start + 1, end, mid)
            v_radius, _ = self.__dataset[mid]
            self.__dataset[start] = (v_radius, v_point)
            if end - mid > self.__leaf_size:
                stack.append((mid, end))
            if (mid - start - 1 > self.__leaf_size) and (v_radius > self.__leaf_radius):
                stack.append((start + 1, mid))

    def ball_search(self, point, eps, inclusive=True):
        search = _BallSearch(self.__distance, point, eps, inclusive)
        stack = [_BallSearchVisit(0, len(self.__dataset))]
        return self._search_iter(search, stack)

    def knn_search(self, point, neighbors):
        search = _KNNSearch(self.__distance, point, neighbors)
        stack = [_KNNSearchVisitPre(0, len(self.__dataset))]
        return self._search_iter(search, stack)

    def _search_iter(self, search, stack):
        while stack:
            visit = stack.pop()
            start, end = visit.bounds()
            if end - start <= self.__leaf_size:
                search.process_all([x for _, x in self.__dataset[start:end]])
            else:
                visit.after(self.__dataset, stack, search)
        return search.get_items()


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


class _KNNSearch:

    def __init__(self, dist, center, neighbors):
        self.__dist = dist
        self.__center = center
        self.__neighbors = neighbors
        self.__items = MaxHeap()

    def __iter__(self):
        return iter(self.__items)

    def __next__(self):
        return next(self.__items)

    def get_items(self):
        while len(self.__items) > self.__neighbors:
            self.__items.pop()
        return [x for (_, x) in self.__items]

    def get_radius(self):
        if len(self.__items) < self.__neighbors:
            return float('inf')
        furthest_dist, _ = self.__items.top()
        return furthest_dist

    def process(self, value):
        dist = self.__dist(self.__center, value)
        if dist >= self.get_radius():
            return dist
        self.__items.add(dist, value)
        while len(self.__items) > self.__neighbors:
            self.__items.pop()
        return dist

    def process_all(self, values):
        for value in values:
            self.process(value)


class _BallSearchVisit:

    def __init__(self, start, end):
        self.__start = start
        self.__end = end

    def bounds(self):
        return self.__start, self.__end

    def after(self, dataset, stack, search):
        v_radius, v_point = dataset[self.__start]
        dist = search.process(v_point)
        mid = (self.__end + self.__start) // 2
        if dist < v_radius:
            fst_start, fst_end = self.__start + 1, mid
            snd_start, snd_end = mid, self.__end
        else:
            fst_start, fst_end = mid, self.__end
            snd_start, snd_end = self.__start + 1, mid
        if abs(dist - v_radius) <= search.get_radius():
            stack.append(_BallSearchVisit(snd_start, snd_end))
        stack.append(_BallSearchVisit(fst_start, fst_end))


class _KNNSearchVisitPre:

    def __init__(self, start, end):
        self.__start = start
        self.__end = end

    def bounds(self):
        return self.__start, self.__end

    def after(self, dataset, stack, search):
        v_radius, v_point = dataset[self.__start]
        dist = search.process(v_point)
        mid = (self.__end + self.__start) // 2
        if dist < v_radius:
            fst_start, fst_end = self.__start + 1, mid
            snd_start, snd_end = mid, self.__end
        else:
            fst_start, fst_end = mid, self.__end
            snd_start, snd_end = self.__start + 1, mid
        stack.append(_KNNSearchVisitPost(snd_start, snd_end, dist))
        stack.append(_KNNSearchVisitPre(fst_start, fst_end))


class _KNNSearchVisitPost:

    def __init__(self, start, end, dist):
        self.__start = start
        self.__end = end
        self.__dist = dist

    def bounds(self):
        return self.__start, self.__end

    def after(self, dataset, stack, search):
        v_radius, _ = dataset[self.__start]
        if abs(self.__dist - v_radius) <= search.get_radius():
            stack.append(_KNNSearchVisitPre(self.__start, self.__end))
