cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int partition_tuple(list data, int start, int end, tuple p):
    cdef int higher = start
    cdef double p_ord = p[0]
    cdef double j_ord
    for j in range(start, end):
        j_ord = data[j][0]
        if j_ord < p_ord:
            data[higher], data[j] = data[j], data[higher]
            higher += 1
    return higher


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void quickselect_tuple(list data, int start, int end, int k):
    cdef int start_ = start
    cdef int end_ = end
    cdef int higher = k
    cdef tuple p
    while higher != k + 1:
        p = data[k]
        data[start_], data[k] = data[k], data[start_]
        higher = partition_tuple(data, start_ + 1, end_, p)
        data[start_], data[higher - 1] = data[higher - 1], data[start_]
        if k <= higher - 1:
            end_ = higher
        else:
            start_ = higher
