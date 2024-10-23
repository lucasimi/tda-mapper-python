import unittest
import random

import numpy as np

from tdamapper.utils.metrics import get_metric
from tdamapper.utils.vptree_hier import VPTree as HVPT
from tdamapper.utils.vptree_flat import VPTree as FVPT
from tests.ball_tree import SkBallTree


distance = 'euclidean'


def dataset(dim=10, num=1000):
    return [np.random.rand(dim) for _ in range(num)]


class TestVPTree(unittest.TestCase):

    eps = 0.25

    neighbors = 5

    def _test_ball_search(self, data, dist, vpt):
        for _ in range(len(data) // 10):
            point = random.choice(data)
            ball = vpt.ball_search(point, self.eps)
            d = get_metric(dist)
            near = [y for y in data if d(point, y) < self.eps]
            for x in ball:
                self.assertTrue(any(d(x, y) == 0.0 for y in near))
            for x in near:
                self.assertTrue(any(d(x, y) == 0.0 for y in ball))

    def _test_knn_search(self, data, dist, vpt):
        for _ in range(len(data) // 10):
            point = random.choice(data)
            neigh = vpt.knn_search(point, self.neighbors)
            self.assertEqual(self.neighbors, len(neigh))
            d = get_metric(dist)
            dist_neigh = [d(point, y) for y in neigh]
            dist_data = [d(point, y) for y in data]
            dist_data.sort()
            dist_neigh.sort()
            self.assertEqual(0.0, dist_data[0])
            self.assertEqual(0.0, dist_neigh[0])
            self.assertEqual(dist_neigh, dist_data[:self.neighbors])
            self.assertEqual(set(dist_neigh), set(dist_data[:self.neighbors]))

    def _test_nn_search(self, data, dist, vpt):
        d = get_metric(dist)
        for val in data:
            neigh = vpt.knn_search(val, 1)
            self.assertEqual(0.0, d(val, neigh[0]))

    def _test_vptree(self, builder, data, dist):
        vpt = builder(data, metric=dist, leaf_radius=self.eps, leaf_capacity=self.neighbors)
        self._test_ball_search(data, dist, vpt)
        self._test_knn_search(data, dist, vpt)
        self._test_nn_search(data, dist, vpt)
        vpt = builder(data, metric=dist, leaf_radius=self.eps, leaf_capacity=self.neighbors, pivoting='random')
        self._test_ball_search(data, dist, vpt)
        self._test_knn_search(data, dist, vpt)
        self._test_nn_search(data, dist, vpt)
        vpt = builder(data, metric=dist, leaf_radius=self.eps, leaf_capacity=self.neighbors, pivoting='furthest')
        self._test_ball_search(data, dist, vpt)
        self._test_knn_search(data, dist, vpt)
        self._test_nn_search(data, dist, vpt)

    def test_vptree_hier_refs(self):
        data = dataset()
        data_refs = list(range(len(data)))
        d = get_metric(distance)

        def dist_refs(i, j):
            return d(data[i], data[j])
        self._test_vptree(HVPT, data_refs, dist_refs)

    def test_vptree_hier_data(self):
        data = dataset()
        self._test_vptree(HVPT, data, distance)

    def test_vptree_flat_refs(self):
        data = dataset()
        data_refs = list(range(len(data)))
        d = get_metric(distance)

        def dist_refs(i, j):
            return d(data[i], data[j])
        self._test_vptree(FVPT, data_refs, dist_refs)

    def test_vptree_flat_data(self):
        data = dataset()
        self._test_vptree(FVPT, data, distance)

    def test_ball_tree_data(self):
        data = dataset()
        self._test_vptree(SkBallTree, data, distance)
