import unittest
import numpy as np

import mapper.graph
import mapper.cover
import mapper.clustering
import mapper.exact

def dist(x, y):
    return np.linalg.norm(x - y)

def dataset(dim=10, num=1000):
    return [np.random.rand(dim) for _ in range(num)]

class TestMapper(unittest.TestCase):

    def testTrivial(self):
        lens = lambda x: x
        data = dataset()
        balls = mapper.cover.trivial_cover(data, None, lens)
        labels = mapper.clustering.trivial(data, balls)
        g = mapper.exact.compute_mapper(data, labels, lens, colormap=np.nanmean)
        self.assertEqual(1, len(g.get_vertices()))
        for vert_id in g.get_vertices():
            self.assertEqual([], g.get_adjaciency(vert_id))

    def testBallSmallRadius(self):
        lens = lambda x: x
        data = [float(i) for i in range(1000)]
        balls = mapper.cover.ball_cover(data, dist, lens, 0.5)
        labels = mapper.clustering.trivial(data, balls)
        g = mapper.exact.compute_mapper(data, labels, lens, colormap=np.nanmean)
        self.assertEqual(1000, len(g.get_vertices()))
        for vert_id in g.get_vertices():
            self.assertEqual([], g.get_adjaciency(vert_id))

    def testBallLargeRadius(self):
        lens = lambda x: x
        data = [float(i) for i in range(1000)]
        balls = mapper.cover.ball_cover(data, dist, lens, 1000.0)
        labels = mapper.clustering.trivial(data, balls)
        g = mapper.exact.compute_mapper(data, labels, lens, colormap=np.nanmean)
        self.assertEqual(1, len(g.get_vertices()))
        for vert_id in g.get_vertices():
            self.assertEqual([], g.get_adjaciency(vert_id))

    def testTwoDisconnectedClusters(self):
        lens = lambda x: x
        data = [np.array([float(i), 0.0]) for i in range(100)]
        data.extend([np.array([float(i), 500.0]) for i in range(100)])
        balls = mapper.cover.ball_cover(data, dist, lens, 150.0)
        self.assertEqual(2, len(balls))
        labels = mapper.clustering.trivial(data, balls)
        g = mapper.exact.compute_mapper(data, labels, lens, colormap=np.nanmean)
        self.assertEqual(2, len(g.get_vertices()))
        for vert_id in g.get_vertices():
            self.assertEqual([], g.get_adjaciency(vert_id))

if __name__=='__main__':
    unittest.main()