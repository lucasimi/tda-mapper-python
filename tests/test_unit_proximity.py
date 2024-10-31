import math
import unittest

import numpy as np

from tdamapper.proximity import (
    BallProximity,
    KNNProximity,
    CubicalProximity
)


def dataset(dim=1, num=10000):
    return [np.random.rand(dim) for _ in range(num)]


class TestProximity(unittest.TestCase):

    def test_ball_proximity(self):
        data = list(range(100))
        prox = BallProximity(radius=10, metric=lambda x,y: abs(x - y))
        prox.fit(data)
        for x in data:
            result = prox.search(x)
            expected = [y for y in data if abs(x - y) < 10]
            self.assertEqual(len(expected), len(result))

    def test_knn_proximity(self):
        data = list(range(100))
        prox = KNNProximity(neighbors=11, metric=lambda x,y: abs(x - y))
        prox.fit(data)
        for x in range(5, 94):
            result = prox.search(x)
            expected = [x + i for i in range(-5, 6)]
            self.assertEqual(set(expected), set(result))

    def test_cubical_proximity(self):
        m, M = 0, 99
        n = 10
        p = 0.1
        w = (M - m) / (n * (1.0 - p))
        delta = p * w
        data = list(range(m, M + 1))
        prox = CubicalProximity(n_intervals=n, overlap_frac=p)
        prox.fit(data)
        for x in data:
            result = prox.search(x)
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
        prox = CubicalProximity(n_intervals=10, overlap_frac=0.5)
        params = prox.get_params()
        self.assertEqual(10, params['n_intervals'])
        self.assertEqual(0.5, params['overlap_frac'])
        prox.set_params(n_intervals=5, overlap_frac=0.25)
        params = prox.get_params()
        self.assertEqual(5, params['n_intervals'])
        self.assertEqual(0.25, params['overlap_frac'])

    def test_knn_params(self):
        prox = KNNProximity(neighbors=10, metric='chebyshev')
        params = prox.get_params()
        self.assertEqual(10, params['neighbors'])
        self.assertEqual('chebyshev', params['metric'])
        prox.set_params(neighbors=5, metric='euclidean')
        params = prox.get_params()
        self.assertEqual(5, params['neighbors'])
        self.assertEqual('euclidean', params['metric'])

    def test_ball_params(self):
        prox = BallProximity(radius=10.0, metric='chebyshev')
        params = prox.get_params()
        self.assertEqual(10.0, params['radius'])
        self.assertEqual('chebyshev', params['metric'])
        prox.set_params(radius=5.0, metric='euclidean')
        params = prox.get_params()
        self.assertEqual(5.0, params['radius'])
        self.assertEqual('euclidean', params['metric'])
