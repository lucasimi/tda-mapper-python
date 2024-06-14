import unittest
import random

import numpy as np

from tdamapper.utils.vptree_hier import VPTree as HVPT
from tdamapper.utils.vptree_flat import VPTree as FVPT
from tdamapper.utils.metrics import euclidean


def dataset(dim=10, num=1000):
    return [np.random.rand(dim) for _ in range(num)]


class TestVPTree(unittest.TestCase):

    eps = 0.25

    neighbors = 5

    def _test_ball_search(self, data, dist, vpt):
        for _ in range(len(data) // 10):
            point = random.choice(data)
            ball = vpt.ball_search(point, self.eps)
            near = [y for y in data if dist(point, y) < self.eps]
            for x in ball:
                self.assertTrue(any(dist(x, y) == 0.0 for y in near))
            for x in near:
                self.assertTrue(any(dist(x, y) == 0.0 for y in ball))

    def _test_knn_search(self, data, dist, vpt):
        for _ in range(len(data) // 10):
            point = random.choice(data)
            neigh = vpt.knn_search(point, self.neighbors)
            self.assertEqual(self.neighbors, len(neigh))
            dist_neigh = [dist(point, y) for y in neigh]
            dist_data = [dist(point, y) for y in data]
            dist_data.sort()
            dist_neigh.sort()
            self.assertEqual(0.0, dist_data[0])
            self.assertEqual(0.0, dist_neigh[0])
            self.assertEqual(dist_neigh, dist_data[:self.neighbors])
            self.assertEqual(set(dist_neigh), set(dist_data[:self.neighbors]))

    def _test_nn_search(self, data, dist, vpt):
        for val in data:
            neigh = vpt.knn_search(val, 1)
            self.assertEqual(0.0, dist(val, neigh[0]))

    def _test_vptree(self, builder, data, dist):
        vpt = builder(
            metric=dist,
            leaf_radius=self.eps,
            leaf_capacity=self.neighbors,
            strategy='fixed')
        vpt.fit(data)
        self._test_ball_search(data, dist, vpt)
        self._test_knn_search(data, dist, vpt)
        self._test_nn_search(data, dist, vpt)
        vpt = builder(
            metric=dist,
            leaf_radius=self.eps,
            leaf_capacity=self.neighbors,
            strategy='random')
        vpt.fit(data)
        self._test_ball_search(data, dist, vpt)
        self._test_knn_search(data, dist, vpt)
        self._test_nn_search(data, dist, vpt)
        vpt = builder(
            metric=dist,
            leaf_radius=self.eps,
            leaf_capacity=self.neighbors,
            strategy='furthest')
        vpt.fit(data)
        self._test_ball_search(data, dist, vpt)
        self._test_knn_search(data, dist, vpt)
        self._test_nn_search(data, dist, vpt)

    def test_hier_vptree_refs(self):
        data = dataset()
        data_refs = list(range(len(data)))
        dist = euclidean()
        def dist_refs(i, j):
            return dist(data[i], data[j])
        self._test_vptree(HVPT, data_refs, dist_refs)

    def test_hier_vptree_data(self):
        data = dataset()
        self._test_vptree(HVPT, data, euclidean())

    def test_flat_vptree_refs(self):
        data = dataset()
        data_refs = list(range(len(data)))
        dist = euclidean()
        def dist_refs(i, j):
            return dist(data[i], data[j])
        self._test_vptree(FVPT, data_refs, dist_refs)

    def test_flat_vptree_data(self):
        data = dataset()
        self._test_vptree(FVPT, data, euclidean())
