from libc.math cimport sqrt, pow, fabs


_EUCLIDEAN = 'euclidean'
_MINKOWSKI = 'minkowski'
_MINKOWSKI_P = 'p'
_CHEBYSHEV = 'chebyshev'
_COSINE = 'cosine'


cpdef double chebyshev(double[:] x, double[:] y) nogil:
    """
    Compute the Chebyshev distance between two vectors.

    The Chebyshev distance is defined as the maximum absolute difference
    between the components of the vectors.

    :param x: The first vector.
    :type x: array-like
    :param y: The second vector.
    :type y: array-like

    :return: The Chebyshev distance between x and y.
    :rtype: double
    """

    cdef double max_diff = 0.0
    cdef Py_ssize_t i, n = x.shape[0]
    for i in range(n):
        max_diff = max(max_diff, fabs(x[i] - y[i]))
    return max_diff


cpdef double euclidean(double[:] x, double[:] y) nogil:
    """
    Compute the Euclidean distance between two vectors.

    The Euclidean distance is defined as the square root of the sum of
    the squared differences between the components of the vectors.

    :param x: The first vector.
    :type x: array-like
    :param y: The second vector.
    :type y: array-like

    :return: The Euclidean distance between x and y.
    :rtype: double
    """

    cdef double norm_squared = 0.0
    cdef Py_ssize_t i, n = x.shape[0]
    for i in range(n):
        norm_squared += pow(fabs(x[i] - y[i]), 2)
    return sqrt(norm_squared)


cdef double _minkowski(int p, double[:] x, double[:] y) nogil:
    cdef double norm_p = 0.0
    cdef Py_ssize_t i, n = x.shape[0]
    for i in range(n):
        norm_p += pow(fabs(x[i] - y[i]), p)
    return pow(norm_p, 1.0 / p)


def minkowski(p):
    """
    Return a function that computes the Minkowski distance for a given order p.

    The Minkowski distance is a generalization of the Euclidean and
    Chebyshev distances. When p = 1, it is equivalent to the Manhattan
    distance, and when p = 2, it is equivalent to the Euclidean distance.

    :param p: The order of the Minkowski distance.
    :type p: int

    :return: A function that computes the Minkowski distance between two vectors.
    :rtype: callable
    """
    return lambda x, y: _minkowski(p, x, y)


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


def get_supported_metrics():
    """
    Return a list of supported metric names.

    :return: A list of supported metric names.
    :rtype: list of str
    """
    
    return [
        _EUCLIDEAN,
        _MINKOWSKI,
        _CHEBYSHEV,
        _COSINE,
    ]


def get_metric(metric, **kwargs):
    """
    Returns a distance metric function based on the specified metric string or callable.

    :param metric: The metric to use. If a callable function is provided, it is returned directly.
        Otherwise, predefined metric returned by `tdamapper.utils.cython.metrics.get_supported_metrics`
        are supported.

    :param kwargs: Additional keyword arguments (e.g., 'p' for Minkowski distance).
    :type kwargs: dict

    :return: The selected distance metric function.
    :rtype: callable

    :raises ValueError: If an invalid metric string is provided.
    """

    if callable(metric):
        return metric
    elif metric == _EUCLIDEAN:
        return euclidean
    elif metric == _MINKOWSKI:
        p = kwargs.get(_MINKOWSKI_P, 2)
        return lambda x, y: minkowski(x, y, p=p)
    elif metric == _CHEBYSHEV:
        return chebyshev
    elif metric == _COSINE:
        return cosine
    else:
        raise ValueError('metric must be a string or callable')
