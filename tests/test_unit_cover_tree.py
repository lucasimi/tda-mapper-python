import unittest
import numpy as np

from tdamapper.utils.metrics import euclidean
from tdamapper.utils.cover_tree import CoverTree


def dataset(dim=1, num=10000):
    return [np.random.rand(dim) for _ in range(num)]


class TestCoverTree(unittest.TestCase):

    def test_cover_tree(self):
        X = list(np.array([
            [0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0]
        ]))
        ct = CoverTree(X, euclidean())
        l_max = ct.get_max_level()
        l_min = ct.get_min_level()
        l_max_points = ct.get_level_points(l_max)
        l_min_points = ct.get_level_points(l_min)
        self.assertEqual(1, len(l_max_points))
        self.assertEqual(len(X), len(l_min_points))
