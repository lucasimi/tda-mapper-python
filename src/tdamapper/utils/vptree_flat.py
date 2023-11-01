"""A class for fast knn and range searches, depending only on a given metric"""
from .quickselect import quickselect_tuple
from .heap import MaxHeap


class KNNSearch:

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
        self.__items.add((dist, value))
        while len(self.__items) > self.__neighbors:
            self.__items.pop()
        return dist

    def process_all(self, values):
        for value in values:
            self.process(value)


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

    def knn_search(self, point, neighbors):
        s = KNNSearch(self.__distance, point, neighbors)
        stack = [VisitLeft(0, len(self.__dataset))]
        return self._search(s, stack)

    def ball_search(self, point, eps, inclusive=True):
        s = BallSearch(self.__distance, point, eps, inclusive)
        stack = [Visit(0, len(self.__dataset))]
        return self._search(s, stack)

    def ball_search_old(self, point, eps, inclusive=True):
        search = BallSearch(self.__distance, point, eps, inclusive)
        stack = [(0, len(self.__dataset))]
        while stack:
            (start, end) = stack.pop()
            if end - start <= self.__leaf_size:
                search.process_all([x for _, x in self.__dataset[start:end]])
            else:
                v_radius, v_point = self.__dataset[start]
                mid = (end + start) // 2
                dist = search.process(v_point)
                if dist <= v_radius + eps:    # results intersects B(center, radius)
                    stack.append((start + 1, mid))
                if dist > v_radius - eps:     # results is not contained in B(center, radius)
                    stack.append((mid, end))
        return search.get_items()

    def _search(self, search, stack):
        while stack:
            visit = stack.pop()
            start, end = visit.start(), visit.end()
            if end - start <= self.__leaf_size:
                search.process_all([x for _, x in self.__dataset[start:end]])
            else:
                visit.after(self.__dataset, stack, search)
        return search.get_items()


class Visit:

    def __init__(self, start, end):
        self.__start = start
        self.__end = end

    def start(self):
        return self.__start

    def end(self):
        return self.__end

    def after(self, dataset, stack, search):
        v_radius, v_point = dataset[self.__start]
        dist = search.process(v_point)
        mid = (self.__end + self.__start) // 2
        if dist > v_radius - search.get_radius():     # results is not contained in B(center, radius)
            stack.append(Visit(mid, self.__end))
        if dist <= v_radius + search.get_radius():    # results intersects B(center, radius)
            stack.append(Visit(self.__start + 1, mid))


class VisitLeft:

    def __init__(self, start, end):
        self.__start = start
        self.__end = end

    def start(self):
        return self.__start

    def end(self):
        return self.__end

    def after(self, dataset, stack, search):
        v_radius, v_point = dataset[self.__start]
        dist = search.process(v_point)
        mid = (self.__end + self.__start) // 2
        stack.append(VisitRight(self.__start, self.__end, dist))
        if dist <= v_radius + search.get_radius():    # results intersects B(center, radius)
            stack.append(VisitLeft(self.__start + 1, mid))


class VisitRight:

    def __init__(self, start, end, dist):
        self.__start = start
        self.__end = end
        self.__dist = dist

    def start(self):
        return self.__start

    def end(self):
        return self.__end

    def after(self, dataset, stack, search):
        v_radius, _ = dataset[self.__start]
        mid = (self.__end + self.__start) // 2
        if self.__dist > v_radius - search.get_radius():     # results is not contained in B(center, radius)
            stack.append(VisitLeft(mid, self.__end))
