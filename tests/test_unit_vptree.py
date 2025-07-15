"""
Unit tests for the vp-tree implementations.
"""

import random

import numpy as np
import pytest

from tdamapper.utils.metrics import get_metric
from tdamapper.utils.vptree_flat.vptree import VPTree as FVPT
from tdamapper.utils.vptree_hier.vptree import VPTree as HVPT
from tests.ball_tree import SkBallTree
from tests.test_utils import (
    dataset_grid,
    dataset_random,
    dataset_simple,
    dataset_two_lines,
)


def distance(metric):
    """
    Get the distance function for the specified metric.
    """
    return get_metric(metric)


def distance_refs(metric, data):
    """
    Get the distance function for the specified metric, using data references.
    This is useful for testing with datasets that are not numpy arrays.
    """
    d = get_metric(metric)

    def dist_refs(i, j):
        return d(data[i, :], data[j, :])

    return dist_refs


SIMPLE = dataset_simple()
SIMPLE_REFS = np.array(list(range(len(SIMPLE))))

TWO_LINES = dataset_two_lines(100)
TWO_LINES_REFS = np.array(list(range(len(TWO_LINES))))

GRID = dataset_grid(10)
GRID_REFS = np.array(list(range(len(GRID))))

RANDOM = dataset_random(2, 100)
RANDOM_REFS = np.array(list(range(len(RANDOM))))


def _test_ball_search(data, dist, vpt, eps):
    for _ in range(len(data) // 10):
        point = random.choice(data)
        ball = vpt.ball_search(point, eps)
        d = get_metric(dist)
        near = [y for y in data if d(point, y) < eps]
        for x in ball:
            assert any(d(x, y) == 0.0 for y in near)
        for x in near:
            assert any(d(x, y) == 0.0 for y in ball)


def _test_knn_search(data, dist, vpt, neighbors):
    for _ in range(len(data) // 10):
        point = random.choice(data)
        neigh = vpt.knn_search(point, neighbors)
        assert neighbors == len(neigh)
        d = get_metric(dist)
        dist_neigh = [d(point, y) for y in neigh]
        dist_data = [d(point, y) for y in data]
        dist_data.sort()
        dist_neigh.sort()
        assert 0.0 == dist_data[0]
        assert 0.0 == dist_neigh[0]
        assert dist_neigh == dist_data[:neighbors]
        assert set(dist_neigh) == set(dist_data[:neighbors])


def _test_nn_search(data, dist, vpt):
    d = get_metric(dist)
    for val in data:
        neigh = vpt.knn_search(val, 1)
        assert 0.0 == d(val, neigh[0])


def _test_vptree(builder, data, dist, eps, neighbors, pivoting):
    vpt = builder(
        data,
        metric=dist,
        leaf_radius=eps,
        leaf_capacity=neighbors,
        pivoting=pivoting,
    )
    _test_ball_search(data, dist, vpt, eps)
    _test_knn_search(data, dist, vpt, neighbors)
    _test_nn_search(data, dist, vpt)


@pytest.mark.parametrize("pivoting", ["disabled", "random", "furthest"])
@pytest.mark.parametrize("eps", [0.1, 0.25, 0.5])
@pytest.mark.parametrize("neighbors", [1, 5, 10])
@pytest.mark.parametrize("builder", [HVPT, FVPT])
@pytest.mark.parametrize("metric", ["euclidean"])
@pytest.mark.parametrize("dataset", [SIMPLE, TWO_LINES, GRID, RANDOM])
def test_vptree(builder, dataset, metric, eps, neighbors, pivoting):
    """
    Test the vp-tree implementations with various datasets and metrics.
    """
    metric = get_metric(metric)
    _test_vptree(builder, dataset, metric, eps, neighbors, pivoting)
