from libc.math cimport sqrt, pow, fabs


_EUCLIDEAN = 'euclidean'
_MINKOWSKI = 'minkowski'
_MINKOWSKI_P = 'p'
_CHEBYSHEV = 'chebyshev'


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


cpdef double minkowski(int p, double[:] x, double[:] y) nogil:
    cdef double norm_p = 0.0
    cdef Py_ssize_t i, n = x.shape[0]
    for i in range(n):
        norm_p += pow(fabs(x[i] - y[i]), p)
    return pow(norm_p, 1.0 / p)


def get_metric(metric, **kwargs):
    if callable(metric):
        return metric
    elif metric == _EUCLIDEAN:
        return euclidean
    elif metric == _MINKOWSKI:
        p = kwargs.get(_MINKOWSKI_P, 2)
        return lambda x, y: minkowski(p, x, y)
    elif metric == _CHEBYSHEV:
        return chebyshev
    else:
        raise ValueError('metric must be a string or callable')