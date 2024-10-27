"""
Utilities for computing metrics.

This module provides functions to calculate various distance metrics. A metric,
or distance function, is a function that maps two points to a double value,
representing the "distance" between them. For a function to qualify as a valid
metric, it must satisfy the following properties:

1. Symmetry: The distance between two points is the same regardless of the
    order, i.e.:
    :math:`d(x, y) = d(y, x)` for all x and y.
2. Positivity: The distance between two distinct points is always positive,
    i.e.:
    :math:`d(x, y) > 0` for all distinct x and y, and :math:`d(x, x) = 0`
    for every x.
3. Triangle inequality: The distance between two points is less than or equal
    to the sum of the distances from a third point, i.e.:
    :math:`d(x, z) \leq d(x, y) + d(y, z)` for all points x, y, z.

Supported distance metrics include:
- Euclidean: The square root of the sum of squared differences between the
components of vectors.
- Minkowski: A generalization of the Euclidean and Chebyshev distances,
parameterized by an order `p`.
- Chebyshev: The maximum absolute difference between the components of vectors.
- Cosine: A distance on unit vectors based on cosine similarity.
"""

import numpy as np
import tdamapper.utils._metrics as _metrics


_EUCLIDEAN = 'euclidean'
_MANHATTAN = 'manhattan'
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
        _MANHATTAN,
        _MINKOWSKI,
        _CHEBYSHEV,
        _COSINE,
    ]


def euclidean():
    """
    Return the Euclidean distance function for vectors.

    The Euclidean distance is defined as the square root of the sum of
    the squared differences between the components of the vectors.

    :return: The Euclidean distance function.
    :rtype: callable
    """
    return _metrics.euclidean


def manhattan():
    """
    Return the Manhattan distance function for vectors.

    The Manhattan distance is defined as the sum of the absolute differences
    between the components of the vectors.

    :return: The Manhattan distance function.
    :rtype: callable
    """
    return _metrics.manhattan


def chebyshev():
    """
    Return the Chebyshev distance function for vectors.

    The Chebyshev distance is defined as the maximum absolute difference
    between the components of the vectors.

    :return: The Chebyshev distance function.
    :rtype: callable
    """
    return _metrics.chebyshev


def minkowski(p):
    """
    Return the Minkowski distance function for order p on vectors.

    The Minkowski distance is a generalization of the Euclidean and Chebyshev
    distances. When p = 1, it is equivalent to the Manhattan distance, and
    when p = 2, it is equivalent to the Euclidean distance. When p is infinite,
    it is equivalent to the Chebyshev distance.

    :param p: The order of the Minkowski distance.
    :type p: int

    :return: The Minkowski distance function.
    :rtype: callable
    """
    if p == 1:
        return manhattan()
    elif p == 2:
        return euclidean()
    elif np.isinf(p):
        return chebyshev()
    return lambda x, y: _metrics.minkowski(p, x, y)


def cosine():
    """
    Return the cosine distance function for vectors.

    The cosine similarity between the input vectors ranges from -1.0 to 1.0.
    - A value of 1.0 indicates that the vectors are in the same direction.
    - A value of 0.0 indicates orthogonality (the vectors are perpendicular).
    - A value of -1.0 indicates that the vectors are diametrically opposed.

    The cosine distance is derived from the cosine similarity :math:`s` and
    is defined as:
    :math:`d(x, y) = \sqrt{2 \cdot (1 - s(x, y))}`

    This definition ensures that the cosine distance satisfies the triangle
    inequality on unit vectors.

    :return: The cosine distance function.
    :rtype: callable
    """
    return _metrics.cosine


def get_metric(metric, **kwargs):
    """
    Return a distance function based on the specified string or callable.

    :param metric: The metric to use. If a callable function is provided, it
        is returned directly. Otherwise, predefined metric names returned by
        `get_supported_metrics()` are supported.
    :type metric: str or callable

    :param kwargs: Additional keyword arguments (e.g., 'p' for Minkowski
        distance).
    :type kwargs: dict

    :return: The selected distance metric function.
    :rtype: callable

    :raises ValueError: If an invalid metric string is provided.
    """
    if callable(metric):
        return metric
    elif metric == _EUCLIDEAN:
        return euclidean()
    elif metric == _MANHATTAN:
        return manhattan()
    elif metric == _MINKOWSKI:
        p = kwargs.get(_MINKOWSKI_P, 2)
        return minkowski(p)
    elif metric == _CHEBYSHEV:
        return chebyshev()
    elif metric == _COSINE:
        return cosine()
    else:
        raise ValueError('metric must be a string or callable')
