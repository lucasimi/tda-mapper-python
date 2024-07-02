"""
Utilities for computing distance metrics.

A distance metric is any function  is a function that maps to points into a double value.
It's required for a distance metric to be symmetric, positive, and satisfy the triangle-inequality,
i.e. :math:`d(x, z) \leq d(x, y) + d(y, z)` for every x, y, z in the dataset.
"""

import tdamapper.utils._metrics as _metrics


_EUCLIDEAN = 'euclidean'
_MINKOWSKI = 'minkowski'
_MINKOWSKI_P = 'p'
_CHEBYSHEV = 'chebyshev'
_COSINE = 'cosine'


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


def euclidean(x, y):
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

    return _metrics.euclidean(x, y)


def chebyshev(x, y):
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

    return _metrics.chebyshev(x, y)


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
    return lambda x, y: _metrics.minkowski(p, x, y)


def cosine(x, y):
    """
    Return a function that computes the cosine distance between two vectors.

    
    :param p: The order of the Minkowski distance.
    :type p: int

    :return: The cosine similarity between the input vectors, which ranges from -1.0 to 1.0.
        A value of 1.0 indicates that the vectors are identical, 0.0 indicates orthogonality,
        and -1.0 indicates they are diametrically opposed.

    :rtype: callable
    """
    return _metrics.cosine(x, y)


def get_metric(metric, **kwargs):
    """
    Returns a distance metric function based on the specified metric string or callable.

    :param metric: The metric to use. If a callable function is provided, it is returned directly.
        Otherwise, predefined metric returned by `tdamapper.utils.metrics.get_supported_metrics`
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
        return minkowski(p)
    elif metric == _CHEBYSHEV:
        return chebyshev
    elif metric == _COSINE:
        return cosine
    else:
        raise ValueError('metric must be a string or callable')