import unittest
import random
import numpy as np

from mapper.utils.vptree_flat import VPTree

def dist(x, y):
    return np.linalg.norm(x - y)


def dataset(dim=10, num=1000):
    return [np.random.rand(dim) for _ in range(num)]


class TestVPTree(unittest.TestCase):

    def testBallSearchDataRefs(self):
        eps = 0.25
        data = dataset()
        data_refs = list(range(len(data)))
        dist_refs = lambda i, j: dist(data[i], data[j])
        vpt = VPTree(dist_refs, data_refs, leaf_radius=eps)
        for val in data_refs:
            neigh = vpt.ball_search(val, eps)
            self.assertTrue(val in neigh)
        point = random.choice(data_refs)
        ball = vpt.ball_search(point, eps)
        near = [y for y in data_refs if dist_refs(point, y) < eps]
        self.assertEqual(set(near), set(ball))

    def testBallSearchData(self):
        eps = 0.25
        data = dataset()
        vpt = VPTree(dist, data, leaf_radius=eps)
        for val in data:
            neigh = vpt.ball_search(val, eps)
            self.assertTrue(any((val == y).all() for y in neigh))
        point = random.choice(data)
        ball = vpt.ball_search(point, eps)
        near = [y for y in data if dist(point, y) < eps]
        for val in ball:
            self.assertTrue(any((val == y).all() for y in near))
        for y in near:
            self.assertTrue(any((x == y).all() for x in ball))
