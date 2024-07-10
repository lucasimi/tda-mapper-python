import unittest

import numpy as np

import tdamapper.utils.metrics as metrics


class TestMetrics(unittest.TestCase):

    def test_euclidean(self):
        d = metrics.euclidean()
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        ab = d(a, b)
        self.assertGreaterEqual(ab, 1.414)
        self.assertLessEqual(ab, 1.415)

    def test_manhattan(self):
        d = metrics.manhattan()
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        ab = d(a, b)
        self.assertEqual(ab, 2.0)

    def test_chebyshev(self):
        d = metrics.chebyshev()
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        ab = d(a, b)
        self.assertEqual(ab, 1.0)

    def test_cosine(self):
        d = metrics.cosine()
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        c = np.array([0.0, 2.0])
        ab = d(a, b)
        self.assertGreaterEqual(ab, 1.414)
        self.assertLessEqual(ab, 1.415)
        bc = d(b, c)
        self.assertEqual(bc, 0.0)

    def test_get_metric(self):
        self.assertEqual(metrics.euclidean(), metrics.get_metric('euclidean'))
        self.assertEqual(metrics.euclidean(), metrics.get_metric('minkowski'))
        self.assertEqual(metrics.chebyshev(), metrics.get_metric('chebyshev'))
        self.assertEqual(metrics.chebyshev(), metrics.get_metric('minkowski', p=np.inf))
        self.assertEqual(metrics.chebyshev(), metrics.get_metric('minkowski', p=float('inf')))
        self.assertEqual(metrics.manhattan(), metrics.get_metric('manhattan'))
        self.assertEqual(metrics.manhattan(), metrics.get_metric('minkowski', p=1))
        self.assertEqual(metrics.cosine(), metrics.get_metric('cosine'))
