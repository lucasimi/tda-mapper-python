import math

import numpy as np
import pytest

from tdamapper.utils.metrics import (
    chebyshev,
    cosine,
    euclidean,
    get_metric,
    get_supported_metrics,
    manhattan,
    minkowski,
)
from tests.test_utils import (
    dataset_empty,
    dataset_grid,
    dataset_random,
    dataset_simple,
    dataset_two_lines,
)

EMPTY = dataset_empty()

SIMPLE = dataset_simple()

TWO_LINES = dataset_two_lines(100)

GRID = dataset_grid(10)

RANDOM = dataset_random(2, 100)


def _check_values(m1, m2, a, b):
    m1_div_by_zero = False
    m1_is_nan = False

    m2_div_by_zero = False
    m2_is_nan = False

    try:
        m1_value = m1(a, b)
        if np.isnan(m1_value):
            m1_is_nan = True
    except ZeroDivisionError:
        m1_div_by_zero = True
    try:
        m2_value = m2(a, b)
        if np.isnan(m2_value):
            m2_is_nan = True
    except ZeroDivisionError:
        m2_div_by_zero = True
    assert m1_div_by_zero == m2_div_by_zero
    assert m1_is_nan == m2_is_nan
    if m1_div_by_zero or m2_div_by_zero:
        return True
    if m1_is_nan or m2_is_nan:
        return True
    return math.isclose(m1_value, m2_value)


@pytest.mark.parametrize("data", [SIMPLE, TWO_LINES, GRID, RANDOM])
@pytest.mark.parametrize(
    "m1, m2",
    [
        (euclidean(), get_metric("euclidean")),
        (manhattan(), get_metric("manhattan")),
        (chebyshev(), get_metric("chebyshev")),
        (minkowski(p=3), get_metric("minkowski", p=3)),
        (minkowski(p=2.5), get_metric("minkowski", p=2.5)),
        (cosine(), get_metric("cosine")),
    ],
)
def test_metrics(m1, m2, data):
    for a in data:
        for b in data:
            assert _check_values(m1, m2, a, b)


def test_supported_metrics():
    expected_metrics = [
        "euclidean",
        "manhattan",
        "chebyshev",
        "cosine",
        "minkowski",
    ]
    supported_metrics = get_supported_metrics()
    assert set(supported_metrics) == set(expected_metrics)
