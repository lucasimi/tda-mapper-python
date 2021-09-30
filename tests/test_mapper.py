import unittest
import numpy as np

import mapper.graph
from mapper.cover import SearchCover, TrivialCover
from mapper.clustering import ClusteringAlgorithm, TrivialClustering
from mapper.search import BallSearch, KnnSearch
from mapper.exact import Mapper


def dist(x, y):
    return np.linalg.norm(x - y)


def dataset(dim=10, num=1000):
    return [np.random.rand(dim) for _ in range(num)]


class TestMapper(unittest.TestCase):

    def testTrivial(self):
        lens = lambda x: x
        data = dataset()
        mp = Mapper(lens, dist, cover_algo=TrivialCover(), clustering_algo=TrivialClustering())
        g = mp.fit(data)
        self.assertEqual(1, len(g.get_vertices()))
        for vert_id in g.get_vertices():
            self.assertEqual([], g.get_adjaciency(vert_id))

    def testBallSmallRadius(self):
        lens = lambda x: x
        data = [float(i) for i in range(1000)]
        mp = Mapper(lens, dist, cover_algo=SearchCover(BallSearch(0.5)), clustering_algo=TrivialClustering())
        g = mp.fit(data)
        self.assertEqual(1000, len(g.get_vertices()))
        for vert_id in g.get_vertices():
            self.assertEqual([], g.get_adjaciency(vert_id))

    def testBallLargeRadius(self):
        lens = lambda x: x
        data = [float(i) for i in range(1000)]
        mp = Mapper(lens, dist, cover_algo=SearchCover(BallSearch(1000.0)), clustering_algo=TrivialClustering())
        g = mp.fit(data)
        self.assertEqual(1, len(g.get_vertices()))
        for vert_id in g.get_vertices():
            self.assertEqual([], g.get_adjaciency(vert_id))

    def testTwoDisconnectedClusters(self):
        lens = lambda x: x
        data = [np.array([float(i), 0.0]) for i in range(100)]
        data.extend([np.array([float(i), 500.0]) for i in range(100)])
        mp = Mapper(lens, dist, cover_algo=SearchCover(BallSearch(150.0)), clustering_algo=TrivialClustering())
        g = mp.fit(data)
        self.assertEqual(2, len(g.get_vertices()))
        for vert_id in g.get_vertices():
            self.assertEqual([], g.get_adjaciency(vert_id))


if __name__=='__main__':
    unittest.main()

