"""
Unit tests for the vp-tree implementations.
"""

import math

import pytest

from tdamapper.utils.metrics import get_metric
from tdamapper.utils.vptree import VPTree
from tdamapper.utils.vptree_flat.vptree import VPTree as FVPT
from tdamapper.utils.vptree_hier.vptree import VPTree as HVPT
from tests.ball_tree import SkBallTree
from tests.test_utils import (
    dataset_empty,
    dataset_grid,
    dataset_random,
    dataset_simple,
    dataset_two_lines,
)

EMPTY = dataset_empty()

SIMPLE = dataset_simple()

TWO_LINES = dataset_two_lines(10)

GRID = dataset_grid(10)

RANDOM = dataset_random(2, 10)


def _test_ball_search(data, dist, vpt, eps):
    d = get_metric(dist)
    for point in data:
        neigh = vpt.ball_search(point, eps, inclusive=False)
        neigh_naive = [y for y in data if d(point, y) < eps]
        dist_neigh = [d(point, y) for y in neigh]
        dist_neigh_naive = [d(point, y) for y in neigh_naive]
        dist_neigh.sort()
        dist_neigh_naive.sort()
        assert len(dist_neigh) == len(dist_neigh_naive)
        assert len(neigh) == len(neigh_naive)
        for x, y in zip(dist_neigh, dist_neigh_naive):
            assert math.isclose(x, y)


def _test_knn_search(data, dist, vpt, neighbors):
    d = get_metric(dist)
    for point in data:
        neigh = vpt.knn_search(point, neighbors)
        neigh_len = len(neigh)
        assert min(neighbors, len(data)) == neigh_len
        dist_neigh = [d(point, y) for y in neigh]
        dist_data = [d(point, y) for y in data]
        dist_data.sort()
        dist_neigh.sort()
        assert math.isclose(dist_data[0], 0.0)
        assert math.isclose(dist_neigh[0], 0.0)
        for x, y in zip(dist_neigh, dist_data[:neigh_len]):
            assert math.isclose(x, y)


def _test_nn_search(data, dist, vpt):
    d = get_metric(dist)
    for val in data:
        neigh = vpt.knn_search(val, 1)
        assert math.isclose(d(val, neigh[0]), 0.0)


def _test_vptree(builder, data, dist, eps, neighbors, pivoting):
    vpt = builder(
        data,
        metric=dist,
        leaf_radius=eps,
        leaf_capacity=neighbors,
        pivoting=pivoting,
    )
    _check_vptree_property(vpt)
    _test_ball_search(data, dist, vpt, eps)
    _test_knn_search(data, dist, vpt, neighbors)
    _test_nn_search(data, dist, vpt)


def _check_vptree_property(vpt):
    arr = vpt.array
    data = arr._dataset
    distances = arr._distances
    indices = arr._indices

    dist = vpt.metric
    leaf_capacity = vpt.leaf_capacity
    leaf_radius = vpt.leaf_radius

    def _check_sub(start, end):
        v_radius = distances[start]
        v_point_index = indices[start]
        v_point = data[v_point_index]

        mid = (start + end) // 2
        for i in range(start + 1, mid):
            y_index = indices[i]
            y = data[y_index]
            assert dist(v_point, y) <= v_radius
        for i in range(mid, end):
            y_index = indices[i]
            y = data[y_index]
            assert dist(v_point, y) >= v_radius

    def _check_rec(start, end):
        v_radius = distances[start]
        if (end - start > leaf_capacity) and (v_radius > leaf_radius):
            _check_sub(start, end)
            mid = (start + end) // 2
            _check_rec(start + 1, mid)
            _check_rec(mid, end)

    _check_rec(0, len(data))


@pytest.mark.parametrize("builder", [HVPT, FVPT])
@pytest.mark.parametrize("dataset", [[], [1], [1, 2]])
def test_vptree_small_dataset(builder, dataset):
    """
    Test the vp-tree implementations with an empty dataset.
    """
    vpt = builder(dataset, metric=lambda x, y: abs(x - y))
    array = vpt.array
    assert array.size() == len(dataset)


@pytest.mark.parametrize("pivoting", ["disabled", "random", "furthest"])
@pytest.mark.parametrize("eps", [0.1, 0.5])
@pytest.mark.parametrize("neighbors", [2, 10])
@pytest.mark.parametrize("builder", [HVPT, FVPT])
@pytest.mark.parametrize("metric", ["euclidean", "manhattan"])
@pytest.mark.parametrize("dataset", [TWO_LINES, GRID, RANDOM])
def test_vptree(builder, dataset, metric, eps, neighbors, pivoting):
    """
    Test the vp-tree implementations with various datasets and metrics.
    """
    metric = get_metric(metric)
    vpt = builder(
        dataset,
        metric=metric,
        leaf_radius=eps,
        leaf_capacity=neighbors,
        pivoting=pivoting,
    )
    _check_vptree_property(vpt)
    _test_ball_search(dataset, metric, vpt, eps)
    _test_knn_search(dataset, metric, vpt, neighbors)
    _test_nn_search(dataset, metric, vpt)


@pytest.mark.parametrize("pivoting", ["disabled", "random", "furthest"])
@pytest.mark.parametrize("eps", [0.1, 0.5])
@pytest.mark.parametrize("neighbors", [2, 10])
@pytest.mark.parametrize("kind", ["flat", "hierarchical"])
@pytest.mark.parametrize("metric", ["euclidean", "manhattan"])
@pytest.mark.parametrize("dataset", [SIMPLE, TWO_LINES])
def test_vptree_public(kind, dataset, metric, eps, neighbors, pivoting):
    """
    Test the vp-tree implementations with various datasets and metrics.
    """
    metric = get_metric(metric)
    vpt = VPTree(
        dataset,
        kind=kind,
        metric=metric,
        leaf_radius=eps,
        leaf_capacity=neighbors,
        pivoting=pivoting,
    )
    _test_ball_search(dataset, metric, vpt, eps)
    _test_knn_search(dataset, metric, vpt, neighbors)
    _test_nn_search(dataset, metric, vpt)


@pytest.mark.parametrize("pivoting", ["disabled", "random", "furthest"])
@pytest.mark.parametrize("eps", [0.1, 0.5])
@pytest.mark.parametrize("neighbors", [2, 10])
@pytest.mark.parametrize("metric", ["euclidean", "manhattan"])
@pytest.mark.parametrize("dataset", [TWO_LINES, GRID, RANDOM])
def test_vptree_sklearn(dataset, metric, eps, neighbors, pivoting):
    """
    Test the baseline vp-tree implementation with various datasets and metrics.
    """
    metric = get_metric(metric)
    vpt = SkBallTree(
        dataset,
        metric=metric,
        leaf_radius=eps,
        leaf_capacity=neighbors,
        pivoting=pivoting,
    )
    _test_ball_search(dataset, metric, vpt, eps)
    _test_knn_search(dataset, metric, vpt, neighbors)
    _test_nn_search(dataset, metric, vpt)
