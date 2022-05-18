import unittest
import numpy as np

import mapper.graph
from mapper.cover import SearchCover, TrivialCover
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
        g = mp.fit(data)
        self.assertEqual(1, len(g))
        for node in g.nodes():
            self.assertEqual([], list(g.neighbors(node)))

    def testBallSmallRadius(self):
        lens = lambda x: x
        data = [float(i) for i in range(1000)]
        mp = MapperPipeline(cover_algo=SearchCover(search_algo=BallSearch(0.5), lens=lens, metric=dist))
        g = mp.fit(data)
        self.assertEqual(1000, len(g))
        for node in g.nodes():
            self.assertEqual([], list(g.neighbors(node)))

    def testBallLargeRadius(self):
        lens = lambda x: x
        data = [float(i) for i in range(1000)]
        mp = MapperPipeline(cover_algo=SearchCover(search_algo=BallSearch(1000.0), lens=lens, metric=dist))
        g = mp.fit(data)
        self.assertEqual(1, len(g))
        for node in g.nodes():
            self.assertEqual([], list(g.neighbors(node)))

    def testTwoDisconnectedClusters(self):
        lens = lambda x: x
        data = [np.array([float(i), 0.0]) for i in range(100)]
        data.extend([np.array([float(i), 500.0]) for i in range(100)])
        mp = MapperPipeline(cover_algo=SearchCover(search_algo=BallSearch(150.0), lens=lens, metric=dist))
        g = mp.fit(data)
        self.assertEqual(2, len(g))
        for node in g.nodes():
            self.assertEqual([], list(g.neighbors(node)))


if __name__=='__main__':
    unittest.main()

