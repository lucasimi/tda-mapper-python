"""A class for fast knn and range searches, depending only on a given metric"""
from random import randrange

import numpy as np

from tdamapper.utils.heap import MaxHeap


#@numba.njit
def partition_tuple(dord, data, start, end, p_ord):
    higher = start
    for j in range(start, end):
        j_ord = dord[j]
        if j_ord < p_ord:
            data[higher], data[j] = data[j], data[higher]
            dord[higher], dord[j] = dord[j], dord[higher]
            higher += 1
    return higher


#@numba.njit
def quickselect_tuple(dord, data, start, end, k):
    start_, end_, higher = start, end, None
    while higher != k + 1:
        p_ord = dord[k]
        data[start_], data[k] = data[k], data[start_]
        dord[start_], dord[k] = dord[k], dord[start_]
        higher = partition_tuple(dord, data, start_ + 1, end_, p_ord)
        data[start_], data[higher - 1] = data[higher - 1], data[start_]
        dord[start_], dord[higher - 1] = dord[higher - 1], dord[start_]
        if k <= higher - 1:
            end_ = higher
        else:
            start_ = higher


#@numba.njit
def _build_iter(dord, data, dist, leaf_capacity, leaf_radius):
    stack = [(0, len(data))]
    while stack:
        start, end = stack.pop()
        if end - start <= leaf_capacity:
            continue
        mid = (end + start) // 2
        _update(dord, data, dist, start, end)
        v_point = data[start]
        quickselect_tuple(dord, data, start + 1, end, mid)
        v_radius = dord[mid]
        dord[start] = v_radius
        data[start] = v_point
        if end - mid > leaf_capacity:
            stack.append((mid, end))
        if (mid - start - 1 > leaf_capacity) and (v_radius > leaf_radius):
            stack.append((start + 1, mid))

#@numba.njit
def _update(dord, data, dist, start, end):
    #self.__pivoting(dord, data, start, end)
    v_point = data[start]
    for i in range(start, end):
        point = data[i]
        dord[i] = dist(v_point, point)

def _pivoting_disabled(dord, data, dist, start, end):
    pass

def _pivoting_random(dord, data, dist, start, end):
    pivot = randrange(start, end)
    if pivot > start:
        data[start], data[pivot] = data[pivot], data[start]
        dord[start], dord[pivot] = dord[pivot], dord[start]

def _furthest(dord, data, dist, start, end, i):
    furthest_dist = 0.0
    furthest = start
    i_point = data[i]
    for j in range(start, end):
        j_point = data[j]
        j_dist = dist(i_point, j_point)
        if j_dist > furthest_dist:
            furthest = j
            furthest_dist = j_dist
    return furthest

def _pivoting_furthest(dord, data, dist, start, end):
    rnd = randrange(start, end)
    furthest_rnd = _furthest(dord, data, dist, start, end, rnd)
    furthest = _furthest(dord, data, dist, start, end, furthest_rnd)
    if furthest > start:
        data[start], data[furthest] = data[furthest], data[start]
        dord[start], dord[furthest] = dord[furthest], dord[start]


class VPTree:

    def __init__(self, distance, dataset, leaf_capacity=1, leaf_radius=0.0, pivoting=None):
        #self.__distance = numba.njit(distance)
        self.__distance = distance
        self.__leaf_capacity = leaf_capacity
        self.__leaf_radius = leaf_radius
        #data = np.array([x for x in dataset])
        data = [x for x in dataset]
        #dord = np.zeros(len(dataset))
        dord = [0.0 for _ in dataset]
        _build_iter(dord, data, self.__distance, leaf_capacity, leaf_radius)
        self.__dord = np.array(dord)
        self.__data = np.array(data)

    def ball_search(self, point, eps, inclusive=True):
        search = _BallSearch(self.__distance, point, eps, inclusive)
        stack = [_BallSearchVisit(0, len(self.__data), float('inf'))]
        return self._search_iter(search, stack)

    def knn_search(self, point, neighbors):
        search = _KNNSearch(self.__distance, point, neighbors)
        stack = [_KNNSearchVisitPre(0, len(self.__data), float('inf'))]
        return self._search_iter(search, stack)

    def _search_iter(self, search, stack):
        while stack:
            visit = stack.pop()
            start, end, m_radius = visit.bounds()
            if (end - start <= self.__leaf_capacity) or (m_radius <= self.__leaf_radius):
                search.process_all([x for x in self.__data[start:end]])
            else:
                visit.after(self.__dord, self.__data, stack, search)
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

    def after(self, dord, data, stack, search):
        v_radius, v_point = dord[self.__start], data[self.__start]
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

    def after(self, dord, data, stack, search):
        v_radius, v_point = dord[self.__start], data[self.__start]
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
