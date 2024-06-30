from libc.math cimport sqrt, pow, fabs


cpdef double euclidean(double[:] x, double[:] y) nogil:
    cdef double norm_squared = 0.0
    cdef Py_ssize_t i, n = x.shape[0]
    for i in range(n):
        norm_squared += pow(fabs(x[i] - y[i]), 2)
    return sqrt(norm_squared)


cpdef double minkowski(int p, double[:] x, double[:] y) nogil:
    cdef double norm_p = 0.0
    cdef Py_ssize_t i, n = x.shape[0]
    for i in range(n):
        norm_p += pow(fabs(x[i] - y[i]), p)
    return pow(norm_p, 1.0 / p)


cpdef double chebyshev(double[:] x, double[:] y) nogil:
    cdef double max_diff = 0.0
    cdef Py_ssize_t i, n = x.shape[0]
    for i in range(n):
        max_diff = max(max_diff, fabs(x[i] - y[i]))
    return max_diff


def get_metric(metric, **kwargs):
    if callable(metric):
        return metric
    elif metric == 'euclidean':
        return euclidean
    elif metric == 'minkowski':
        p = kwargs.get('p', 2)
        return lambda x, y: minkowski(p, x, y)
    elif metric == 'chebyshev':
        return chebyshev
    else:
        raise ValueError('metric must be a string or callable')
