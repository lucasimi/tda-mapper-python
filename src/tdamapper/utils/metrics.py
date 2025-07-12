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

from typing import Any, Callable, Literal, Union, get_args

import numpy as np

import tdamapper.utils._metrics as _metrics

Metric = Callable[[Any, Any], float]

MetricLiteral = Literal[
    "euclidean",
    "manhattan",
    "minkowski",
    "chebyshev",
    "cosine",
]


def get_supported_metrics() -> list[MetricLiteral]:
    """
    Return a list of supported metric names.

    :return: A list of supported metric names.
    """
    return list(get_args(MetricLiteral))


def euclidean(**_kwargs: dict[str, Any]) -> Metric:
    """
    Return the Euclidean distance function for vectors.

    The Euclidean distance is defined as the square root of the sum of
    the squared differences between the components of the vectors.

    :return: The Euclidean distance function.
    """
    return _metrics.euclidean


def manhattan(**_kwargs: dict[str, Any]) -> Metric:
    """
    Return the Manhattan distance function for vectors.

    The Manhattan distance is defined as the sum of the absolute differences
    between the components of the vectors.

    :return: The Manhattan distance function.
    """
    return _metrics.manhattan


def chebyshev(**_kwargs: dict[str, Any]) -> Metric:
    """
    Return the Chebyshev distance function for vectors.

    The Chebyshev distance is defined as the maximum absolute difference
    between the components of the vectors.

    :return: The Chebyshev distance function.
    """
    return _metrics.chebyshev


def minkowski(**kwargs: dict[str, Any]) -> Metric:
    """
    Return the Minkowski distance function for order p on vectors.

    The Minkowski distance is a generalization of the Euclidean and Chebyshev
    distances. When p = 1, it is equivalent to the Manhattan distance, and
    when p = 2, it is equivalent to the Euclidean distance. When p is infinite,
    it is equivalent to the Chebyshev distance.

    :param p: The order of the Minkowski distance.
    :return: The Minkowski distance function.
    """
    p = kwargs.get("p", 2)
    if not isinstance(p, (int, float)):
        raise TypeError("p must be an integer or a float")
    if p == 1:
        return manhattan()
    if p == 2:
        return euclidean()
    if np.isinf(p):
        return chebyshev()

    def dist(x: Any, y: Any) -> float:
        return _metrics.minkowski(p, x, y)

    return dist


def cosine(**_kwargs: dict[str, Any]) -> Metric:
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
    """
    return _metrics.cosine


def get_metric(
    metric: Union[MetricLiteral, Metric], **kwargs: dict[str, Any]
) -> Metric:
    """
    Return a distance function based on the specified string or callable.

    :param metric: The metric to use. If a callable function is provided, it
        is returned directly. Otherwise, predefined metric names returned by
        `get_supported_metrics()` are supported.
    :param kwargs: Additional keyword arguments (e.g., 'p' for Minkowski
        distance).
    :return: The selected distance metric function.
    :raises ValueError: If an invalid metric string is provided.
    """
    if callable(metric):
        return metric
    if metric == "euclidean":
        return euclidean(**kwargs)
    if metric == "manhattan":
        return manhattan(**kwargs)
    if metric == "minkowski":
        return minkowski(**kwargs)
    if metric == "chebyshev":
        return chebyshev(**kwargs)
    if metric == "cosine":
        return cosine(**kwargs)
    raise ValueError("metric must be a known string or callable")
