import unittest
import numpy as np
from sklearn.cluster import DBSCAN
from tdamapper.core import MapperAlgorithm, build_connected_components
from tdamapper.cover import TrivialCover, BallCover
from tdamapper.clustering import TrivialClustering


def dist(x, y):
    return np.linalg.norm(x - y)


def dataset(dim=10, num=1000):
    return [np.random.rand(dim) for _ in range(num)]


class TestMapper(unittest.TestCase):

    def testTrivial(self):
        data = dataset()
        mp = MapperAlgorithm(TrivialCover(), TrivialClustering())
        g = mp.fit_transform(data, data)
        self.assertEqual(1, len(g))
        self.assertEqual([], list(g.neighbors(0)))
        ccs = build_connected_components(g)
        self.assertEqual(1, len(set(ccs.values())))

    def testBallSmallRadius(self):
        data = np.array([[float(i)] for i in range(1000)])
        mp = MapperAlgorithm(BallCover(0.5, metric=dist), TrivialClustering())
        g = mp.fit_transform(data, data)
        self.assertEqual(1000, len(g))
        for node in g.nodes():
            self.assertEqual([], list(g.neighbors(node)))
        ccs = build_connected_components(g)
        self.assertEqual(1000, len(set(ccs.values())))

    def testBallSmallRadiusList(self):
        data = [np.array([float(i)]) for i in range(1000)]
        mp = MapperAlgorithm(cover=BallCover(0.5, metric=dist),
            clustering=DBSCAN(eps=1.0, min_samples=1))
        g = mp.fit_transform(data, data)
        self.assertEqual(1000, len(g))
        for node in g.nodes():
            self.assertEqual([], list(g.neighbors(node)))
        ccs = build_connected_components(g)
        self.assertEqual(1000, len(set(ccs.values())))

    def testBallLargeRadius(self):
        data = np.array([[float(i)] for i in range(1000)])
        mp = MapperAlgorithm(cover=BallCover(1000.0, metric=dist),
            clustering=TrivialClustering())
        g = mp.fit_transform(data, data)
        self.assertEqual(1, len(g))
        for node in g.nodes():
            self.assertEqual([], list(g.neighbors(node)))
        ccs = build_connected_components(g)
        self.assertEqual(1, len(set(ccs.values())))

    def testTwoDisconnectedClusters(self):
        data = [np.array([float(i), 0.0]) for i in range(100)]
        data.extend([np.array([float(i), 500.0]) for i in range(100)])
        data = np.array(data)
        mp = MapperAlgorithm(cover=BallCover(150.0, metric=dist),
            clustering=TrivialClustering())
        g = mp.fit_transform(data, data)
        self.assertEqual(2, len(g))
        for node in g.nodes():
            self.assertEqual([], list(g.neighbors(node)))
        ccs = build_connected_components(g)
        self.assertEqual(2, len(set(ccs.values())))

    def testTwoConnectedClusters(self):
        data = [
            np.array([0.0, 1.0]), np.array([1.0, 0.0]),
            np.array([0.0, 0.0]), np.array([1.0, 1.0])]
        mp = MapperAlgorithm(cover=BallCover(1.1, metric=dist),
            clustering=TrivialClustering())
        g = mp.fit_transform(data, data)
        self.assertEqual(2, len(g))
        for node in g.nodes():
            self.assertEqual(1, len(list(g.neighbors(node))))
        ccs = build_connected_components(g)
        self.assertEqual(1, len(set(ccs.values())))
