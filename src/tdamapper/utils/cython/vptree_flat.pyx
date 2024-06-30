from random import randrange

cimport cython
cimport numpy as np

import tdamapper.utils.cython.metrics as metrics

cimport tdamapper.utils.cython.quickselect as quickselect
cimport tdamapper.utils.cython.heap as heap


cdef class VPTree:

    cdef object __distance
    cdef list __dataset
    cdef int __leaf_capacity
    cdef double __leaf_radius
    cdef object __pivoting

    def __init__(self, object distance, list dataset, int leaf_capacity=1, double leaf_radius=0.0, pivoting=None):
        self.__distance = metrics.get_metric(distance)
        self.__dataset = [(0.0, x) for x in dataset]
        self.__leaf_capacity = leaf_capacity
        self.__leaf_radius = leaf_radius
        self.__pivoting = self._pivoting_disabled
        if pivoting == 'random':
            self.__pivoting = self._pivoting_random
        elif pivoting == 'furthest':
            self.__pivoting = self._pivoting_furthest
        self._build_iter()

    cdef void _pivoting_disabled(self, int start, int end):
        pass

    cdef void _pivoting_random(self, int start, int end):
        cdef int pivot = randrange(start, end)
        if pivot > start:
            self.__dataset[start], self.__dataset[pivot] = self.__dataset[pivot], self.__dataset[start]

    cdef int _furthest(self, int start, int end, int i):
        cdef double furthest_dist = 0.0
        cdef int furthest = start
        cdef np.ndarray[np.float64_t, ndim=1] i_point = self.__dataset[i][1]
        cdef np.ndarray[np.float64_t, ndim=1] j_point
        cdef double j_dist
        for j in range(start, end):
            j_point = self.__dataset[j][1]
            j_dist = self.__distance(i_point, j_point)
            if j_dist > furthest_dist:
                furthest = j
                furthest_dist = j_dist
        return furthest

    cdef void _pivoting_furthest(self, int start, int end):
        cdef int rnd = randrange(start, end)
        cdef int furthest_rnd = self._furthest(start, end, rnd)
        cdef int furthest = self._furthest(start, end, furthest_rnd)
        if furthest > start:
            self.__dataset[start], self.__dataset[furthest] = self.__dataset[furthest], self.__dataset[start]

    cdef void _update(self, int start, int end):
        self.__pivoting(start, end)
        cdef np.ndarray[np.float64_t, ndim=1] v_point = self.__dataset[start][1]
        cdef int i
        cdef np.ndarray[np.float64_t, ndim=1] point
        for i in range(start, end):
            point = self.__dataset[i][1]
            self.__dataset[i] = (self.__distance(v_point, point), point)

    cdef void _build_iter(self):
        cdef list stack = [(0, len(self.__dataset))]
        cdef int start, end, mid
        cdef np.ndarray[np.float64_t, ndim=1] v_point
        cdef double v_radius
        while stack:
            start, end = stack.pop()
            if end - start <= self.__leaf_capacity:
                continue
            mid = (end + start) // 2
            self._update(start, end)
            v_point = self.__dataset[start][1]
            quickselect.quickselect_tuple(self.__dataset, start + 1, end, mid)
            v_radius = self.__dataset[mid][0]
            self.__dataset[start] = (v_radius, v_point)
            if end - mid > self.__leaf_capacity:
                stack.append((mid, end))
            if (mid - start - 1 > self.__leaf_capacity) and (v_radius > self.__leaf_radius):
                stack.append((start + 1, mid))

    cpdef list ball_search(self, np.ndarray[np.float64_t, ndim=1] point, double eps):
        cdef BallSearch search
        cdef list stack
        search = BallSearch(self.__distance, point, eps, True)
        stack = [BallSearchVisit(0, len(self.__dataset), float('inf'))]
        return self._ball_search_iter(search, stack)

    cpdef list knn_search(self, np.ndarray[np.float64_t, ndim=1] point, int neighbors):
        cdef KNNSearch search
        cdef list stack
        search = KNNSearch(self.__distance, point, neighbors)
        stack = [KNNSearchVisit(0, len(self.__dataset), float('inf'), 0.0, 0.0, 'pre')]
        return self._knn_search_iter(search, stack)

    cdef list _ball_search_iter(self, BallSearch search, list stack):
        cdef int start, end
        cdef double m_radius
        cdef BallSearchVisit visit
        while stack:
            visit = stack.pop()
            start, end, m_radius = visit.get_bounds()
            if (end - start <= self.__leaf_capacity) or (m_radius <= self.__leaf_radius):
                search.process_all([x for _, x in self.__dataset[start:end]])
            else:
                visit.after(self.__dataset, stack, search)
        return search.get_items()

    cdef list _knn_search_iter(self, KNNSearch search, list stack):
        cdef int start, end
        cdef double m_radius
        cdef KNNSearchVisit visit
        while stack:
            visit = stack.pop()
            start, end, m_radius = visit.get_bounds()
            if (end - start <= self.__leaf_capacity) or (m_radius <= self.__leaf_radius):
                search.process_all([x for _, x in self.__dataset[start:end]])
            else:
                visit.after(self.__dataset, stack, search)
        return search.get_items()


cdef class BallSearch:

    cdef object __distance
    cdef double[:] __center
    cdef double __radius
    cdef list __items
    cdef object __inside

    def __init__(self, object distance, np.ndarray[np.float64_t, ndim=1] center, double radius, bint inclusive):
        self.__distance = distance
        self.__center = center
        self.__radius = radius
        self.__items = []
        self.__inside = self._inside_inclusive if inclusive else self._inside_not_inclusive

    cdef list get_items(self):
        return self.__items

    cdef double get_radius(self):
        return self.__radius

    cdef void process_all(self, list values):
        cdef list inside
        inside = [x for x in values if self.__inside(self._from_center(x))]
        self.__items.extend(inside)

    cdef double process(self, np.ndarray[np.float64_t, ndim=1] value):
        cdef double dist = self._from_center(value)
        if self.__inside(dist):
            self.__items.append(value)
        return dist

    cdef double _from_center(self, np.ndarray[np.float64_t, ndim=1] value):
        return self.__distance(self.__center, value)

    cdef bint _inside_inclusive(self, double dist):
        return dist <= self.__radius

    cdef bint _inside_not_inclusive(self, double dist):
        return dist < self.__radius


cdef class KNNSearch:

    cdef object __dist
    cdef double[:] __center
    cdef int __neighbors
    cdef heap.MaxHeap __items

    def __init__(self, object dist, np.ndarray[np.float64_t, ndim=1] center, int neighbors):
        self.__dist = dist
        self.__center = center
        self.__neighbors = neighbors
        self.__items = heap.MaxHeap()

    def __iter__(self):
        return iter(self.__items)

    def __next__(self):
        return next(self.__items)

    cdef list get_items(self):
        while len(self.__items) > self.__neighbors:
            self.__items.pop()
        return [x for (_, x) in self.__items]

    cdef double get_radius(self):
        if len(self.__items) < self.__neighbors:
            return float('inf')
        cdef double furthest_dist = self.__items.top()[0]
        return furthest_dist

    cdef double process(self, np.ndarray[np.float64_t, ndim=1] value):
        cdef double dist = self.__dist(self.__center, value)
        if dist >= self.get_radius():
            return dist
        self.__items.add(dist, value)
        while len(self.__items) > self.__neighbors:
            self.__items.pop()
        return dist

    cdef void process_all(self, list values):
        cdef np.ndarray[np.float64_t, ndim=1] value
        for value in values:
            self.process(value)


cdef class BallSearchVisit:

    cdef int __start
    cdef int __end
    cdef double __m_radius

    def __init__(self, int start, int end, double m_radius):
        self.__start = start
        self.__end = end
        self.__m_radius = m_radius

    cdef tuple get_bounds(self):
        return self.__start, self.__end, self.__m_radius

    cdef void after(self, list dataset, list stack, BallSearch search):
        cdef double v_radius = dataset[self.__start][0]
        cdef object v_point = dataset[self.__start][1]
        cdef double dist = search.process(v_point)
        cdef int mid = (self.__end + self.__start) // 2
        cdef int fst_start, fst_end, snd_start, snd_end
        cdef double fst_radius, snd_radius
        if dist <= v_radius:
            fst_start, fst_end, fst_radius = self.__start + 1, mid, v_radius
            snd_start, snd_end, snd_radius = mid, self.__end, float('inf')
        else:
            fst_start, fst_end, fst_radius = mid, self.__end, float('inf')
            snd_start, snd_end, snd_radius = self.__start + 1, mid, v_radius
        if abs(dist - v_radius) <= search.get_radius():
            stack.append(BallSearchVisit(snd_start, snd_end, snd_radius))
        stack.append(BallSearchVisit(fst_start, fst_end, fst_radius))


cdef class KNNSearchVisit:

    cdef int __start
    cdef int __end
    cdef double __m_radius

    cdef double __dist
    cdef double __v_radius

    cdef str __type

    def __init__(self, int start, int end, double m_radius, double dist, double v_radius, str type):
        self.__start = start
        self.__end = end
        self.__m_radius = m_radius
        self.__dist = dist
        self.__v_radius = v_radius
        self.__type = type

    cdef tuple get_bounds(self):
        return self.__start, self.__end, self.__m_radius

    cdef void after_pre(self, list dataset, list stack, KNNSearch search):
        cdef double v_radius = dataset[self.__start][0]
        cdef object v_point = dataset[self.__start][1]
        cdef double dist = search.process(v_point)
        cdef int mid = (self.__end + self.__start) // 2
        cdef int fst_start, fst_end, snd_start, snd_end
        cdef double fst_radius, snd_radius
        if dist <= v_radius:
            fst_start, fst_end, fst_radius = self.__start + 1, mid, v_radius
            snd_start, snd_end, snd_radius = mid, self.__end, float('inf')
        else:
            fst_start, fst_end, fst_radius = mid, self.__end, float('inf')
            snd_start, snd_end, snd_radius = self.__start + 1, mid, v_radius
        stack.append(KNNSearchVisit(snd_start, snd_end, snd_radius, dist, v_radius, 'post'))
        stack.append(KNNSearchVisit(fst_start, fst_end, fst_radius, 0.0, 0.0, 'pre'))

    cdef void after_post(self, list dataset, list stack, KNNSearch search):
        if abs(self.__dist - self.__v_radius) <= search.get_radius():
            stack.append(KNNSearchVisit(self.__start, self.__end, self.__m_radius, 0.0, 0.0, 'pre'))

    cdef void after(self, list dataset, list stack, KNNSearch search):
        if self.__type == 'pre':
            self.after_pre(dataset, stack, search)
        elif self.__type == 'post':
            self.after_post(dataset, stack, search)
