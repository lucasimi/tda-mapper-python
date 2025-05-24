import random

import numpy as np

from tdamapper.utils.metrics import get_metric
from tdamapper.utils.vptree_flat.vptree import VPTree as FVPT
from tdamapper.utils.vptree_hier.vptree import VPTree as HVPT
from tests.ball_tree import SkBallTree

distance = "euclidean"


def dataset(dim=10, num=1000):
    return [np.random.rand(dim) for _ in range(num)]


eps = 0.25

neighbors = 5


def _test_ball_search(data, dist, vpt):
    for _ in range(len(data) // 10):
        point = random.choice(data)
        ball = vpt.ball_search(point, eps)
        d = get_metric(dist)
        near = [y for y in data if d(point, y) < eps]
        for x in ball:
            assert any(d(x, y) == 0.0 for y in near)
        for x in near:
            assert any(d(x, y) == 0.0 for y in ball)


def _test_knn_search(data, dist, vpt):
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


def _test_vptree(builder, data, dist):
    vpt = builder(data, metric=dist, leaf_radius=eps, leaf_capacity=neighbors)
    _test_ball_search(data, dist, vpt)
    _test_knn_search(data, dist, vpt)
    _test_nn_search(data, dist, vpt)
    vpt = builder(
        data,
        metric=dist,
        leaf_radius=eps,
        leaf_capacity=neighbors,
        pivoting="random",
    )
    _test_ball_search(data, dist, vpt)
    _test_knn_search(data, dist, vpt)
    _test_nn_search(data, dist, vpt)
    vpt = builder(
        data,
        metric=dist,
        leaf_radius=eps,
        leaf_capacity=neighbors,
        pivoting="furthest",
    )
    _test_ball_search(data, dist, vpt)
    _test_knn_search(data, dist, vpt)
    _test_nn_search(data, dist, vpt)


def test_vptree_hier_refs():
    data = dataset()
    data_refs = list(range(len(data)))
    d = get_metric(distance)

    def dist_refs(i, j):
        return d(data[i], data[j])

    _test_vptree(HVPT, data_refs, dist_refs)


def test_vptree_hier_data():
    data = dataset()
    _test_vptree(HVPT, data, distance)


def test_vptree_flat_refs():
    data = dataset()
    data_refs = list(range(len(data)))
    d = get_metric(distance)

    def dist_refs(i, j):
        return d(data[i], data[j])

    _test_vptree(FVPT, data_refs, dist_refs)


def test_vptree_flat_data():
    data = dataset()
    _test_vptree(FVPT, data, distance)


def test_ball_tree_data():
    data = dataset()
    _test_vptree(SkBallTree, data, distance)
