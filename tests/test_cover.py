import math
import unittest

import numpy as np

from tdamapper.cover import (
    TrivialCover,
    BallCover,
    KNNCover,
    CubicalCover
)


def dist(x, y):
    return np.linalg.norm(x - y)


def dataset(dim=1, num=10000):
    return [np.random.rand(dim) for _ in range(num)]


class TestCover(unittest.TestCase):

    def testTrivialCover(self):
        data = dataset()
        cover = TrivialCover()
        charts = list(cover.apply(data))
        self.assertEqual(1, len(charts))

    def testBallCoverSimple(self):
        data = [
            np.array([0.0, 1.0]), np.array([1.0, 0.0]),
            np.array([0.0, 0.0]), np.array([1.0, 1.0])]
        cover = BallCover(radius=1.1, metric=dist)
        charts = list(cover.apply(data))
        self.assertEqual(2, len(charts))

    def testBallCover(self):
        data = list(range(100))
        cover = BallCover(radius=10, metric=lambda x,y: abs(x - y))
        cover.fit(data)
        for x in data:
            result = cover.search(x)
            expected = [y for y in data if abs(x - y) <= 10]
            self.assertEqual(len(expected), len(result))

    def testKNNCoverSimple(self):
        data = [
            np.array([0.0, 1.0]), np.array([1.1, 0.0]),
            np.array([0.0, 0.0]), np.array([1.1, 1.0])]
        cover = KNNCover(neighbors=2, metric=dist)
        charts = list(cover.apply(data))
        self.assertEqual(2, len(charts))

    def testKNNCover(self):
        data = list(range(100))
        cover = KNNCover(neighbors=11, metric=lambda x,y: abs(x - y))
        cover.fit(data)
        for x in range(5, 94):
            result = cover.search(x)
            expected = [x + i for i in range(-5, 6)]
            self.assertEqual(set(expected), set(result))

    def testCubicalCoverSimple(self):
        data = [
            np.array([0.0, 1.0]), np.array([1.1, 0.0]),
            np.array([0.0, 0.0]), np.array([1.1, 1.0])]
        cover = CubicalCover(n_intervals=2, overlap_frac=0.5)
        charts = list(cover.apply(data))
        self.assertEqual(4, len(charts))

    def testCubicalCover(self):
        m, M = 0, 99
        n = 10
        p = 0.1
        w = (M - m) / (n * (1.0 - p))
        delta = p * w
        data = list(range(m, M + 1))
        cover = CubicalCover(n_intervals=n, overlap_frac=p)
        cover.fit(data)
        for x in data:
            result = cover.search(x)
            i = math.floor((x - m) / (w - delta))
            a_i = m + i * (w - delta) - delta / 2.0
            b_i = m + (i + 1) * (w - delta) + delta / 2.0
            expected = [y for y in data if y > a_i and y < b_i]
            self.assertEqual(set(expected), set(result))
