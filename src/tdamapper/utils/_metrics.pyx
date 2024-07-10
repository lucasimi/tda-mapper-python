from libc.math cimport sqrt, pow, fabs


cpdef inline double chebyshev(double[:] x, double[:] y) nogil:
    cdef double max_diff = 0.0
    cdef Py_ssize_t i, n = x.shape[0]
    for i in range(n):
        max_diff = max(max_diff, fabs(x[i] - y[i]))
    return max_diff


cpdef inline double euclidean(double[:] x, double[:] y) nogil:
    cdef double norm_squared = 0.0
    cdef double diff
    cdef Py_ssize_t i, n = x.shape[0]
    for i in range(n):
        diff = x[i] - y[i]
        norm_squared += diff * diff
    return sqrt(norm_squared)


cpdef inline double manhattan(double[:] x, double[:] y) nogil:
    cdef double norm = 0.0
    cdef Py_ssize_t i, n = x.shape[0]
    for i in range(n):
        norm += fabs(x[i] - y[i])
    return norm


cpdef inline double minkowski(int p, double[:] x, double[:] y) nogil:
    cdef double norm_p = 0.0
    cdef Py_ssize_t i, n = x.shape[0]
    for i in range(n):
        norm_p += pow(fabs(x[i] - y[i]), p)
    return pow(norm_p, 1.0 / p)


cpdef inline double cosine(double[:] x, double[:] y) nogil:
    cdef double dot_product = 0.0
    cdef double norm_x = 0.0
    cdef double norm_y = 0.0
    cdef Py_ssize_t i, n = x.shape[0]
    cdef double similarity = 0.0
    for i in range(n):
        dot_product += x[i] * y[i]
        norm_x += x[i] * x[i]
        norm_y += y[i] * y[i]
    similarity = dot_product / sqrt(norm_x * norm_y)
    return sqrt(2.0 * (1.0 - similarity))
