from libc.math cimport sqrt, pow, fabs


cpdef double chebyshev(double[:] x, double[:] y) nogil:
    cdef double max_diff = 0.0
    cdef Py_ssize_t i, n = x.shape[0]
    for i in range(n):
        max_diff = max(max_diff, fabs(x[i] - y[i]))
    return max_diff


cpdef double euclidean(double[:] x, double[:] y) nogil:
    cdef double norm_squared = 0.0
    cdef Py_ssize_t i, n = x.shape[0]
    for i in range(n):
        norm_squared += pow(fabs(x[i] - y[i]), 2)
    return sqrt(norm_squared)


cdef double minkowski(int p, double[:] x, double[:] y) nogil:
    cdef double norm_p = 0.0
    cdef Py_ssize_t i, n = x.shape[0]
    for i in range(n):
        norm_p += pow(fabs(x[i] - y[i]), p)
    return pow(norm_p, 1.0 / p)


cpdef double cosine(double[:] x, double[:] y) nogil:
    cdef double dot_product = 0.0
    cdef double norm_x = 0.0
    cdef double norm_y = 0.0
    cdef Py_ssize_t i, n = x.shape[0]
    for i in range(n):
        dot_product += x[i] * y[i]
        norm_x += pow(x[i], 2)
        norm_y += pow(y[i], 2)
    return 1.0 - (dot_product / sqrt(norm_x * norm_y))
