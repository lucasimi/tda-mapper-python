import unittest
import numpy as np

from tdamapper.utils.metrics import euclidean
from tdamapper.utils.cover_tree import CoverTree


def dataset(dim=1, num=10000):
    return [np.random.rand(dim) for _ in range(num)]


class TestCoverTree(unittest.TestCase):

    def _check_separation(self, dist, t, i):
        i_level_points = t.get_level_points(i)
        for p in i_level_points:
            for q in i_level_points:
                if not p == q:
                    d = dist(p, q)
                    self.assertGreaterEqual(d, 2**i)

    def _check_covering(self, dist, t, i):
        i_level_nodes = t.get_level_nodes(i)
        i1_level_nodes = t.get_level_nodes(i - 1)
        for p in i1_level_nodes:
            found = False
            for q in i_level_nodes:
                if p in q.get_children():
                    if found:
                        self.fail()
                    found = True
                    self.assertLessEqual(dist(p.get_point(), q.get_point()), 2**i)
            if not found:
                self.fail()

    def _check_cover_tree(self, dist, t):
        l_max = t.get_max_level()
        l_min = t.get_min_level()
        for i in range(l_min, l_max + 1):
            self._check_separation(dist, t, i)
        for i in range(l_max, l_min, -1):
            self._check_covering(dist, t, i)

    def test_cover_tree(self):
        X = list(np.array([
            [0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0]
        ]))
        dist = euclidean()
        ct = CoverTree(X, dist)
        l_max = ct.get_max_level()
        l_min = ct.get_min_level()
        l_max_points = ct.get_level_points(l_max)
        l_min_points = ct.get_level_points(l_min)
        self.assertEqual(1, len(l_max_points))
        self.assertEqual(len(X), len(l_min_points))
        self._check_cover_tree(dist, ct)
