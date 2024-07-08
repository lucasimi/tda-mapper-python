import unittest

import numpy as np

from tdamapper.cover import (
    TrivialCover,
    BallCover,
    KNNCover,
    CubicalCover
)


def dataset(dim=1, num=10000):
    return [np.random.rand(dim) for _ in range(num)]


class TestCover(unittest.TestCase):

    def test_trivial_cover(self):
        data = dataset()
        cover = TrivialCover()
        charts = list(cover.apply(data))
        self.assertEqual(1, len(charts))

    def test_ball_cover_simple(self):
        data = [
            np.array([0.0, 1.0]), np.array([1.0, 0.0]),
            np.array([0.0, 0.0]), np.array([1.0, 1.0])]
        cover = BallCover(radius=1.1, metric='euclidean')
        charts = list(cover.apply(data))
        self.assertEqual(2, len(charts))

    def test_knn_cover_simple(self):
        data = [
            np.array([0.0, 1.0]), np.array([1.1, 0.0]),
            np.array([0.0, 0.0]), np.array([1.1, 1.0])]
        cover = KNNCover(neighbors=2, metric='euclidean')
        charts = list(cover.apply(data))
        self.assertEqual(2, len(charts))

    def test_cubical_cover_simple(self):
        data = [
            np.array([0.0, 1.0]), np.array([1.1, 0.0]),
            np.array([0.0, 0.0]), np.array([1.1, 1.0])]
        cover = CubicalCover(n_intervals=2, overlap_frac=0.5)
        charts = list(cover.apply(data))
        self.assertEqual(4, len(charts))
