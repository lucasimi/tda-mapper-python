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


@pytest.mark.parametrize("data", [SIMPLE, TWO_LINES, GRID, RANDOM])
@pytest.mark.parametrize(
    "m1, m2",
    [
        (euclidean(), get_metric("euclidean")),
        (manhattan(), get_metric("manhattan")),
        (chebyshev(), get_metric("chebyshev")),
        (minkowski(p=3), get_metric("minkowski", p=3)),
        (cosine(), get_metric("cosine")),
    ],
)
def test_metrics(m1, m2, data):
    for a in data:
        for b in data:
            m1_fail = False
            m2_fail = False
            m1_value = 0.0
            m2_value = 0.0
            try:
                m1_value = m1(a, b)
            except Exception:
                m1_fail = True
            try:
                m2_value = m2(a, b)
            except Exception:
                m2_fail = True
            assert m1_fail == m2_fail
            if np.isnan(m1_value) and np.isnan(m2_value):
                return
            assert math.isclose(m1_value, m2_value)


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
