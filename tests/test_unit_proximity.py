import math
import unittest

import numpy as np

from tdamapper.cover import (
    BallCover,
    KNNCover,
    CubicalCover
)


def dataset(dim=1, num=10000):
    return [np.random.rand(dim) for _ in range(num)]


class TestProximity(unittest.TestCase):

    def test_ball_proximity(self):
        data = list(range(100))
        cover = BallCover(radius=10, metric=lambda x,y: abs(x - y))
        cover.fit(data)
        for x in data:
            result = cover.search(x)
            expected = [y for y in data if abs(x - y) < 10]
            self.assertEqual(len(expected), len(result))

    def test_knn_proximity(self):
        data = list(range(100))
        cover = KNNCover(neighbors=11, metric=lambda x,y: abs(x - y))
        cover.fit(data)
        for x in range(5, 94):
            result = cover.search(x)
            expected = [x + i for i in range(-5, 6)]
            self.assertEqual(set(expected), set(result))

    def test_cubical_proximity(self):
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
            for c in result:
                self.assertTrue(c in expected)
            for c in expected:
                self.assertTrue(c in result)
            #self.assertEqual(set(expected), set(result))

    def test_cubical_params(self):
        cover = CubicalCover(n_intervals=10, overlap_frac=0.5)
        params = cover.get_params()
        self.assertEqual(10, params['n_intervals'])
        self.assertEqual(0.5, params['overlap_frac'])
        cover.set_params(n_intervals=5, overlap_frac=0.25)
        params = cover.get_params()
        self.assertEqual(5, params['n_intervals'])
        self.assertEqual(0.25, params['overlap_frac'])

    def test_knn_params(self):
        cover = KNNCover(neighbors=10, metric='chebyshev')
        params = cover.get_params()
        self.assertEqual(10, params['neighbors'])
        self.assertEqual('chebyshev', params['metric'])
        cover.set_params(neighbors=5, metric='euclidean')
        params = cover.get_params()
        self.assertEqual(5, params['neighbors'])
        self.assertEqual('euclidean', params['metric'])

    def test_ball_params(self):
        cover = BallCover(radius=10.0, metric='chebyshev')
        params = cover.get_params()
        self.assertEqual(10.0, params['radius'])
        self.assertEqual('chebyshev', params['metric'])
        cover.set_params(radius=5.0, metric='euclidean')
        params = cover.get_params()
        self.assertEqual(5.0, params['radius'])
        self.assertEqual('euclidean', params['metric'])
