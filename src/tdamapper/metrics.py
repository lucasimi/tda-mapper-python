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
    :math:`d(x, z) \\leq d(x, y) + d(y, z)` for all points x, y, z.

Supported distance metrics include:
- Euclidean: The square root of the sum of squared differences between the
components of vectors.
- Minkowski: A generalization of the Euclidean and Chebyshev distances,
parameterized by an order `p`.
- Chebyshev: The maximum absolute difference between the components of vectors.
- Cosine: A distance on unit vectors based on cosine similarity.
"""

from enum import Enum
from typing import Callable, List, Union

import numpy as np

import tdamapper._metrics as _metrics

_MINKOWSKI_P = "p"


class Metric(str, Enum):
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    MINKOWSKI = "minkowski"
    CHEBYSHEV = "chebyshev"
    COSINE = "cosine"


def euclidean(*args, **kwargs) -> Callable:
    """
    Return the Euclidean distance function for vectors.

    The Euclidean distance is defined as the square root of the sum of
    the squared differences between the components of the vectors.

    :return: The Euclidean distance function.
    :rtype: callable
    """
    return _metrics.euclidean


def manhattan(*args, **kwargs) -> Callable:
    """
    Return the Manhattan distance function for vectors.

    The Manhattan distance is defined as the sum of the absolute differences
    between the components of the vectors.

    :return: The Manhattan distance function.
    :rtype: callable
    """
    return _metrics.manhattan


def chebyshev(*args, **kwargs) -> Callable:
    """
    Return the Chebyshev distance function for vectors.

    The Chebyshev distance is defined as the maximum absolute difference
    between the components of the vectors.

    :return: The Chebyshev distance function.
    :rtype: callable
    """
    return _metrics.chebyshev


def minkowski(*args, **kwargs) -> Callable:
    """
    Return the Minkowski distance function for order p on vectors.

    The Minkowski distance is a generalization of the Euclidean and Chebyshev
    distances. When p = 1, it is equivalent to the Manhattan distance, and
    when p = 2, it is equivalent to the Euclidean distance. When p is infinite,
    it is equivalent to the Chebyshev distance.

    :return: The Minkowski distance function.
    :rtype: callable
    """
    p = kwargs.get(_MINKOWSKI_P, 2)

    if p == 1:
        return manhattan()
    elif p == 2:
        return euclidean()
    elif np.isinf(p):
        return chebyshev()

    def dist(x, y):
        return _metrics.minkowski(p, x, y)

    return dist


def cosine(*args, **kwargs) -> Callable:
    """
    Return the cosine distance function for vectors.

    The cosine similarity between the input vectors ranges from -1.0 to 1.0.
    - A value of 1.0 indicates that the vectors are in the same direction.
    - A value of 0.0 indicates orthogonality (the vectors are perpendicular).
    - A value of -1.0 indicates that the vectors are diametrically opposed.

    The cosine distance is derived from the cosine similarity :math:`s` and
    is defined as:
    :math:`d(x, y) = \\sqrt{2 \\cdot (1 - s(x, y))}`

    This definition ensures that the cosine distance satisfies the triangle
    inequality on unit vectors.

    :return: The cosine distance function.
    :rtype: callable
    """
    return _metrics.cosine


def _get_supported_metrics() -> List[str]:
    """
    Return a list of supported metric names.

    :return: A list of supported metric names.
    :rtype: list of str
    """
    return [m.value for m in Metric]

    
def get_metric_function(metric: Metric, *args, **kwargs) -> Callable:
    """
    Return the distance function for the specified metric.

    :param metric: The metric to use, as a string from the supported metrics.
    :type metric: Metric

    :return: The selected distance metric function.
    :rtype: callable

    :raises ValueError: If an invalid metric string is provided.
    """
    match metric:
        case Metric.EUCLIDEAN:
            return euclidean(*args, **kwargs)
        case Metric.MANHATTAN:
            return manhattan(*args, **kwargs)
        case Metric.MINKOWSKI:
            return minkowski(*args, **kwargs)
        case Metric.CHEBYSHEV:
            return chebyshev(*args, **kwargs)
        case Metric.COSINE:
            return cosine(*args, **kwargs)


def get_metric(metric: Union[str, Metric, Callable], *args, **kwargs) -> Callable:
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
    elif isinstance(metric, str):
        metric_enum = Metric(metric)
        if metric_enum not in _get_supported_metrics():
            raise ValueError(
                f"Unsupported metric: {metric}. "
                f"Supported metrics are: {', '.join(_get_supported_metrics())}"
            )
        return get_metric_function(metric_enum, *args, **kwargs)
    elif isinstance(metric, Metric):
        return get_metric_function(metric, *args, **kwargs)
    else:
        raise ValueError("metric must be a string or callable")


def _first_run() -> None:
    """
    Ensure that the metric functions are compiled with Numba on the first run.
    """
    a = np.array([0.0, 1.0])
    b = np.array([1.0, 0.0])
    for metric in Metric:
        f = get_metric_function(metric)
        f(a, b)  # Trigger the function to ensure it compiles with Numba


_first_run()  # Ensure the functions are compiled on first import
