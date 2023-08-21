import unittest

import numpy as np
from mapper.cover import TrivialCover, BallCover, KnnCover, CubicCover


def dist(x, y):
    return np.linalg.norm(x - y)


def dataset(dim=10, num=1000):
    return [np.random.rand(dim) for _ in range(num)]


class TestCover(unittest.TestCase):

    def testTrivialCover(self):
        data = dataset()
        cover = TrivialCover()
        charts = list(cover.charts(data))
        self.assertEqual(1, len(charts))

    def testBallCover(self):
        data = [
            np.array([0.0, 1.0]), np.array([1.0, 0.0]),
            np.array([0.0, 0.0]), np.array([1.0, 1.0])]
        cover = BallCover(1.1, metric=dist)
        charts = list(cover.charts(data))
        self.assertEqual(2, len(charts))

    def testKnnCover(self):
        data = [
            np.array([0.0, 1.0]), np.array([1.1, 0.0]),
            np.array([0.0, 0.0]), np.array([1.1, 1.0])]
        cover = KnnCover(2, metric=dist)
        charts = list(cover.charts(data))
        self.assertEqual(2, len(charts))

    def testCubicCover(self):
        data = [
            np.array([0.0, 1.0]), np.array([1.1, 0.0]),
            np.array([0.0, 0.0]), np.array([1.1, 1.0])]
        cover = CubicCover(2, 0.5)
        charts = list(cover.charts(data))
        self.assertEqual(4, len(charts))