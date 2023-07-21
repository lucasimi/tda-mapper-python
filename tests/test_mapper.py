import unittest
import numpy as np

from sklearn.cluster import DBSCAN

from mapper.search import BallSearch, KnnSearch
from mapper.pipeline import MapperPipeline


def dist(x, y):
    return np.linalg.norm(x - y)


def dataset(dim=10, num=1000):
    return [np.random.rand(dim) for _ in range(num)]


class TestMapper(unittest.TestCase):

    def testTrivial(self):
        data = dataset()
        mp = MapperPipeline()
        g = mp.fit(data).get_graph()
        self.assertEqual(1, len(g))
        for node in g.nodes():
            self.assertEqual([], list(g.neighbors(node)))
        cc = MapperPipeline.compute_connected_components(g)
        self.assertEqual(1, len(cc))

    def testBallSmallRadius(self):
        data = np.array([[float(i)] for i in range(1000)])
        mp = MapperPipeline(search=BallSearch(0.5, metric=dist))
        g = mp.fit(data).get_graph()
        self.assertEqual(1000, len(g))
        for node in g.nodes():
            self.assertEqual([], list(g.neighbors(node)))

    def testBallSmallRadiusList(self):
        data = [[float(i)] for i in range(1000)]
        mp = MapperPipeline(search=BallSearch(0.5, metric=dist), clustering=DBSCAN(eps=1.0, min_samples=1))
        g = mp.fit(data).get_graph()
        self.assertEqual(1000, len(g))
        for node in g.nodes():
            self.assertEqual([], list(g.neighbors(node)))

    def testBallLargeRadius(self):
        data = np.array([[float(i)] for i in range(1000)])
        mp = MapperPipeline(search=BallSearch(1000.0, metric=dist))
        g = mp.fit(data).get_graph()
        self.assertEqual(1, len(g))
        for node in g.nodes():
            self.assertEqual([], list(g.neighbors(node)))

    def testTwoDisconnectedClusters(self):
        data = [np.array([float(i), 0.0]) for i in range(100)]
        data.extend([np.array([float(i), 500.0]) for i in range(100)])
        data = np.array(data)
        mp = MapperPipeline(search=BallSearch(150.0, metric=dist))
        g = mp.fit(data).get_graph()
        self.assertEqual(2, len(g))
        for node in g.nodes():
            self.assertEqual([], list(g.neighbors(node)))
