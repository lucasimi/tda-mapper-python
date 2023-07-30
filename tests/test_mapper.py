import unittest
import numpy as np

from sklearn.cluster import DBSCAN

from mapper.search import BallSearch, KnnSearch
from mapper.pipeline import MapperAlgorithm, compute_connected_components
from mapper.cover import TrivialCover, BallCover, KnnCover, TrivialClustering


def dist(x, y):
    return np.linalg.norm(x - y)


def dataset(dim=10, num=1000):
    return [np.random.rand(dim) for _ in range(num)]


class TestMapper(unittest.TestCase):

    def testTrivial(self):
        data = dataset()
        mp = MapperAlgorithm(TrivialCover(), TrivialClustering())
        labels = mp.build_labels(data)
        self.assertEqual(len(data), len(labels))
        adj = mp.build_adjaciency(mp.build_labels(data))
        self.assertEqual(1, len(adj))
        self.assertEqual((list(range(len(data))), []), adj[0])
        g = mp.build_graph(data)
        self.assertEqual(1, len(g))
        self.assertEqual([], list(g.neighbors(0)))

    def testBallSmallRadius(self):
        data = np.array([[float(i)] for i in range(1000)])
        mp = MapperAlgorithm(BallCover(0.5, metric=dist), TrivialClustering())
        g = mp.build_graph(data)
        self.assertEqual(1000, len(g))
        for node in g.nodes():
            self.assertEqual([], list(g.neighbors(node)))

    def testBallSmallRadiusList(self):
        data = [np.array([float(i)]) for i in range(1000)]
        mp = MapperAlgorithm(cover=BallCover(0.5, metric=dist), clustering=DBSCAN(eps=1.0, min_samples=1))
        g = mp.build_graph(data)
        self.assertEqual(1000, len(g))
        for node in g.nodes():
            self.assertEqual([], list(g.neighbors(node)))

    def testBallLargeRadius(self):
        data = np.array([[float(i)] for i in range(1000)])
        mp = MapperAlgorithm(cover=BallCover(1000.0, metric=dist), clustering=TrivialClustering())
        g = mp.build_graph(data)
        self.assertEqual(1, len(g))
        for node in g.nodes():
            self.assertEqual([], list(g.neighbors(node)))

    def testTwoDisconnectedClusters(self):
        data = [np.array([float(i), 0.0]) for i in range(100)]
        data.extend([np.array([float(i), 500.0]) for i in range(100)])
        data = np.array(data)
        mp = MapperAlgorithm(cover=BallCover(150.0, metric=dist), clustering=TrivialClustering())
        g = mp.build_graph(data)
        self.assertEqual(2, len(g))
        for node in g.nodes():
            self.assertEqual([], list(g.neighbors(node)))
