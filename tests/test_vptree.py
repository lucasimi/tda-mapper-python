import unittest
import random
import numpy as np

from mapper.utils.vptree import VPTree

def dist(x, y):
    return np.linalg.norm(x - y)

def dataset(dim=10, num=1000):
    return [np.random.rand(dim) for _ in range(num)]

class TestVPTree(unittest.TestCase):

    def testBallSearchDataRefs(self):
        eps = 0.25
        data = dataset()
        data_refs = [i for i in range(len(data))]
        dist_refs = lambda i, j: dist(data[i], data[j])
        vpt = VPTree(dist_refs, data_refs, min_radius=eps)
        for x in data_refs:
            neigh = vpt.ball_search(x, eps)
            self.assertTrue(x in neigh)
        point = random.choice(data_refs)
        ball = vpt.ball_search(point, eps)
        near = [y for y in data_refs if dist_refs(point, y) < eps]
        self.assertEqual(set(near), set(ball))
    
    def testBallSearchData(self):
        eps = 0.25
        data = dataset()
        vpt = VPTree(dist, data, min_radius=eps)
        for x in data:
            neigh = vpt.ball_search(x, eps)
            self.assertTrue(any((x == y).all() for y in neigh))
        point = random.choice(data)
        ball = vpt.ball_search(point, eps)
        near = [y for y in data if dist(point, y) < eps]
        for x in ball:
            self.assertTrue(any((x == y).all() for y in near))
        for y in near:
            self.assertTrue(any((x == y).all() for x in ball))

    def testKNNSearchDataRefs(self):
        data = dataset()
        data_refs = [i for i in range(len(data))]
        dist_refs = lambda i, j: dist(data[i], data[j])
        vpt = VPTree(dist_refs, data_refs)
        for x in data_refs:
            neigh = vpt.knn_search(x, 1)
            self.assertEqual(x, neigh)
    
    def testKNNSearchData(self):
        data = dataset()
        vpt = VPTree(dist, data)
        for x in data:
            neigh = vpt.knn_search(x, 1)
            self.assertEqual(0.0, dist(x, neigh))

    def testKNNSearchDataRefs(self):
        k = 5
        data = dataset()
        data_refs = [i for i in range(len(data))]
        dist_refs = lambda i, j: dist(data[i], data[j])
        vpt = VPTree(dist_refs, data_refs, max_count=k)
        for x in data_refs:
            neigh = vpt.knn_search(x, k)
            self.assertLessEqual(len(neigh), k)
        point = random.choice(data_refs)
        neigh = vpt.knn_search(point, k)
        data_refs.sort(key=lambda y: dist_refs(point, y))
        self.assertEqual(set(data_refs[:k]), set(neigh))
            
    def testKNNSearchData(self):
        k = 5
        data = dataset()
        vpt = VPTree(dist, data, max_count=k)
        for x in data:
            neigh = vpt.knn_search(x, k)
            self.assertLessEqual(len(neigh), k)
            self.assertTrue(any((x == y).all() for y in neigh))
        point = random.choice(data)
        neigh = vpt.knn_search(point, k)
        data.sort(key=lambda y: dist(point, y))
        for x in neigh:
            self.assertTrue(any((x == y).all() for y in data[:k]))
        for y in data[:k]:
            self.assertTrue(any((x == y).all() for x in neigh))

if __name__=='__main__':
    unittest.main()