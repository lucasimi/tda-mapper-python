"""A class for fast knn and range searches, depending only on a given metric"""
from random import randrange

from tdamapper.utils.metrics import get_metric
from tdamapper.utils.quickselect import quickselect
from tdamapper.utils.heap import MaxHeap


def _swap(arr, i, j):
    arr[i], arr[j] = arr[j], arr[i]


class VPTree:

    def __init__(self, distance, dataset, leaf_capacity=1, leaf_radius=0.0, pivoting=None, **kwargs):
        self.__distance = get_metric(distance, **kwargs)
        self.__dataset = [(0.0, x) for x in dataset]
        self.__leaf_capacity = leaf_capacity
        self.__leaf_radius = leaf_radius
        self.__pivoting = self._pivoting_disabled
        if pivoting == 'random':
            self.__pivoting = self._pivoting_random
        elif pivoting == 'furthest':
            self.__pivoting = self._pivoting_furthest
        self._build_iter()

    def _pivoting_disabled(self, start, end):
        pass

    def _pivoting_random(self, start, end):
        pivot = randrange(start, end)
        if pivot > start:
            _swap(self.__dataset, start, pivot)

    def _furthest(self, start, end, i):
        furthest_dist = 0.0
        furthest = start
        _, i_point = self.__dataset[i]
        for j in range(start, end):
            _, j_point = self.__dataset[j]
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
            _swap(self.__dataset, start, furthest)

    def _update(self, start, end):
        self.__pivoting(start, end)
        _, v_point = self.__dataset[start]
        for i in range(start, end):
            _, point = self.__dataset[i]
            self.__dataset[i] = self.__distance(v_point, point), point

    def _build_iter(self):
        stack = [(0, len(self.__dataset))]
        while stack:
            start, end = stack.pop()
            mid = (end + start) // 2
            self._update(start, end)
            _, v_point = self.__dataset[start]
            quickselect(self.__dataset, start + 1, end, mid)
            v_radius, _ = self.__dataset[mid]
            self.__dataset[start] = (v_radius, v_point)
            if (end - start > 2 * self.__leaf_capacity) and (v_radius > self.__leaf_radius):
                stack.append((mid, end))
                stack.append((start + 1, mid))

    def ball_search(self, point, eps, inclusive=True):
        search = _BallSearch(self.__distance, point, eps, inclusive)
        stack = [_BallSearchVisit(0, len(self.__dataset), float('inf'))]
        return self._search_iter(search, stack)

    def knn_search(self, point, neighbors):
        search = _KNNSearch(self.__distance, point, neighbors)
        stack = [_KNNSearchVisitPre(0, len(self.__dataset), float('inf'))]
        return self._search_iter(search, stack)

    def _search_iter(self, search, stack):
        while stack:
            visit = stack.pop()
            start, end, m_radius = visit.bounds()
            v_radius, _ = self.__dataset[start]
            if (end - start <= 2 * self.__leaf_capacity) or (m_radius <= self.__leaf_radius) or (v_radius <= self.__leaf_radius):
                search.process_all(self.__dataset, start, end)
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

    def process_all(self, data, start, end):
        for _, x in data[start:end]:
            if self.__inside(self._dist_from_center(x)):
                self.__items.append(x)

    def process(self, p):
        dist = self._dist_from_center(p)
        if self.__inside(dist):
            self.__items.append(p)
        return dist

    def _dist_from_center(self, p):
        return self.__distance(self.__center, p)

    def _inside_inclusive(self, dist):
        return dist <= self.__radius

    def _inside_not_inclusive(self, dist):
        return dist < self.__radius


class _KNNSearch:

    def __init__(self, distance, center, neighbors):
        self.__distance = distance
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

    def _dist_from_center(self, p):
        return self.__distance(self.__center, p)

    def process(self, x):
        dist = self._dist_from_center(x)
        if dist >= self.get_radius():
            return dist
        self.__items.add(dist, x)
        while len(self.__items) > self.__neighbors:
            self.__items.pop()
        return dist

    def process_all(self, data, start, end):
        for _, x in data[start:end]:
            self.process(x)


class _BallSearchVisit:

    def __init__(self, start, end, m_radius):
        self.__start = start
        self.__end = end
        self.__m_radius = m_radius

    def bounds(self):
        return self.__start, self.__end, self.__m_radius

    def after(self, dataset, stack, search):
        v_radius, v_point = dataset[self.__start]
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

    def after(self, dataset, stack, search):
        v_radius, v_point = dataset[self.__start]
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

    def after(self, _, stack, search):
        if abs(self.__dist - self.__v_radius) <= search.get_radius():
            stack.append(_KNNSearchVisitPre(self.__start, self.__end, self.__m_radius))
