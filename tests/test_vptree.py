import unittest
import random
import numpy as np
from tdamapper.utils.vptree import VPTree
from tdamapper.utils.vptree_flat import VPTree as FlatVPTree


def distance(x, y):
    return np.linalg.norm(x - y)


def dataset(dim=10, num=1000):
    return [np.random.rand(dim) for _ in range(num)]


class TestVPTree(unittest.TestCase):

    eps = 0.25

    neighbors = 5

    def _testBallSearch(self, data, dist, vpt):
        for _ in range(len(data) // 10):
            point = random.choice(data)
            ball = vpt.ball_search(point, self.eps)
            near = [y for y in data if dist(point, y) < self.eps]
            for x in ball:
                self.assertTrue(any(dist(x, y) == 0.0 for y in near))
            for x in near:
                self.assertTrue(any(dist(x, y) == 0.0 for y in ball))

    def _testKNNSearch(self, data, dist, vpt):
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

    def _testNNSearch(self, data, dist, vpt):
        for val in data:
            neigh = vpt.knn_search(val, 1)
            self.assertEqual(0.0, dist(val, neigh[0]))

    def _testVPTree(self, builder, data, dist):
        vpt = builder(dist, data, leaf_radius=self.eps, leaf_capacity=self.neighbors)
        self._testBallSearch(data, dist, vpt)
        self._testKNNSearch(data, dist, vpt)
        self._testNNSearch(data, dist, vpt)
        vpt = builder(dist, data, leaf_radius=self.eps, leaf_capacity=self.neighbors, pivoting='random')
        self._testBallSearch(data, dist, vpt)
        self._testKNNSearch(data, dist, vpt)
        self._testNNSearch(data, dist, vpt)
        vpt = builder(dist, data, leaf_radius=self.eps, leaf_capacity=self.neighbors, pivoting='furthest')
        self._testBallSearch(data, dist, vpt)
        self._testKNNSearch(data, dist, vpt)
        self._testNNSearch(data, dist, vpt)

    def testVPTreeRefs(self):
        data = dataset()
        data_refs = list(range(len(data)))
        def dist_refs(i, j):
            return distance(data[i], data[j])
        self._testVPTree(VPTree, data_refs, dist_refs)

    def testVPTreeData(self):
        data = dataset()
        self._testVPTree(VPTree, data, distance)

    def testFlatVPTreeRefs(self):
        data = dataset()
        data_refs = list(range(len(data)))
        def dist_refs(i, j):
            return distance(data[i], data[j])
        self._testVPTree(FlatVPTree, data_refs, dist_refs)

    def testFlatVPTreeData(self):
        data = dataset()
        self._testVPTree(FlatVPTree, data, distance)
