"""A class for fast knn and range searches, depending only on a given metric"""
from random import randrange

from tdamapper.utils.cython.metrics import get_metric
from tdamapper.utils.quickselect import quickselect_tuple
from tdamapper.utils.heap import MaxHeap


def _swap(arr, i, j):
    arr[i], arr[j] = arr[j], arr[i]


class VPTree:

    def __init__(self, distance, dataset, leaf_capacity=1, leaf_radius=0.0, pivoting=None):
        self.__distance = get_metric(distance)
        self.__arr_data = list(dataset)
        self.__arr_ord = [0.0 for _ in dataset]
        self.__leaf_capacity = leaf_capacity
        self.__leaf_radius = leaf_radius
        self.__pivoting = self._pivoting_disabled
        if pivoting == 'random':
            self.__pivoting = self._pivoting_random
        elif pivoting == 'furthest':
            self.__pivoting = self._pivoting_furthest
        self._build_iter()

    def __getitem__(self, k):
        x_ord = self.__arr_ord[k]
        x_data = self.__arr_data[k]
        return x_ord, x_data

    def __len__(self):
        return len(self.__arr_data)

    def _pivoting_disabled(self, start, end):
        pass

    def _pivoting_random(self, start, end):
        pivot = randrange(start, end)
        if pivot > start:
            _swap(self.__arr_data, start, pivot)
            _swap(self.__arr_ord, start, pivot)

    def _furthest(self, start, end, i):
        furthest_dist = 0.0
        furthest = start
        i_point = self.__arr_data[i]
        for j in range(start, end):
            j_point = self.__arr_data[j]
            j_dist = self.__distance(i_point, j_point)
            if j_dist > furthest_dist:
                furthest = j
                furthest_dist = j_dist
        return furthest

    def _pivoting_furthest(self, start, end):
        rnd = randrange(start, end)
        furthest_rnd = self._furthest(start, end, rnd)
        furthest = self._furthest(start, end, furthest_rnd)
        if furthest > start:
            _swap(self.__arr_data, start, furthest)
            _swap(self.__arr_ord, start, furthest)

    def _update(self, start, end):
        self.__pivoting(start, end)
        v_point = self.__arr_data[start]
        for i in range(start, end):
            point = self.__arr_data[i]
            self.__arr_ord[i] = self.__distance(v_point, point)

    def _build_iter(self):
        stack = [(0, len(self.__arr_data))]
        while stack:
            start, end = stack.pop()
            if end - start <= self.__leaf_capacity:
                continue
            mid = (end + start) // 2
            self._update(start, end)
            quickselect_tuple(self.__arr_ord, self.__arr_data, start + 1, end, mid)
            v_radius = self.__arr_ord[mid]
            self.__arr_ord[start] = v_radius
            if end - mid > self.__leaf_capacity:
                stack.append((mid, end))
            if (mid - start - 1 > self.__leaf_capacity) and (v_radius > self.__leaf_radius):
                stack.append((start + 1, mid))

    def ball_search(self, point, eps, inclusive=True):
        search = _BallSearch(self.__distance, point, eps, inclusive)
        stack = [_BallSearchVisit(0, len(self.__arr_data), float('inf'))]
        return self._search_iter(search, stack)

    def knn_search(self, point, neighbors):
        search = _KNNSearch(self.__distance, point, neighbors)
        stack = [_KNNSearchVisitPre(0, len(self.__arr_data), float('inf'))]
        return self._search_iter(search, stack)

    def _search_iter(self, search, stack):
        while stack:
            visit = stack.pop()
            start, end, m_radius = visit.bounds()
            if (end - start <= self.__leaf_capacity) or (m_radius <= self.__leaf_radius):
                search.process_all(self.__arr_data[start:end])
            else:
                visit.after(self.__arr_ord, self.__arr_data, stack, search)
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

    def __init__(self, start, end, m_radius):
        self.__start = start
        self.__end = end
        self.__m_radius = m_radius

    def bounds(self):
        return self.__start, self.__end, self.__m_radius

    def after(self, arr_ord, arr_data, stack, search):
        v_radius = arr_ord[self.__start]
        v_point = arr_data[self.__start]
        dist = search.process(v_point)
        mid = (self.__end + self.__start) // 2
        if dist <= v_radius:
            fst_start, fst_end, fst_radius = self.__start + 1, mid, v_radius
            snd_start, snd_end, snd_radius = mid, self.__end, float('inf')
        else:
            fst_start, fst_end, fst_radius = mid, self.__end, float('inf')
            snd_start, snd_end, snd_radius = self.__start + 1, mid, v_radius
        if abs(dist - v_radius) <= search.get_radius():
            stack.append(_BallSearchVisit(snd_start, snd_end, snd_radius))
        stack.append(_BallSearchVisit(fst_start, fst_end, fst_radius))


class _KNNSearchVisitPre:

    def __init__(self, start, end, m_radius):
        self.__start = start
        self.__end = end
        self.__m_radius = m_radius

    def bounds(self):
        return self.__start, self.__end, self.__m_radius

    def after(self, arr_ord, arr_data, stack, search):
        v_radius = arr_ord[self.__start]
        v_point = arr_data[self.__start]
        dist = search.process(v_point)
        mid = (self.__end + self.__start) // 2
        if dist <= v_radius:
            fst_start, fst_end, fst_radius = self.__start + 1, mid, v_radius
            snd_start, snd_end, snd_radius = mid, self.__end, float('inf')
        else:
            fst_start, fst_end, fst_radius = mid, self.__end, float('inf')
            snd_start, snd_end, snd_radius = self.__start + 1, mid, v_radius
        stack.append(_KNNSearchVisitPost(snd_start, snd_end, snd_radius, dist, v_radius))
        stack.append(_KNNSearchVisitPre(fst_start, fst_end, fst_radius))


class _KNNSearchVisitPost:

    def __init__(self, start, end, m_radius, dist, v_radius):
        self.__start = start
        self.__end = end
        self.__m_radius = m_radius
        self.__dist = dist
        self.__v_radius = v_radius

    def bounds(self):
        return self.__start, self.__end, self.__m_radius

    def after(self, _, __, stack, search):
        if abs(self.__dist - self.__v_radius) <= search.get_radius():
            stack.append(_KNNSearchVisitPre(self.__start, self.__end, self.__m_radius))
