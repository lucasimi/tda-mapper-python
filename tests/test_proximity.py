import time
import logging
import random
import unittest
import numpy as np
from tdamapper.proximity import TrivialProximity, BallProximity, KNNProximity, CubicalProximity
from tdamapper.core import proximity_net


def dist(x, y):
    return np.linalg.norm(x - y)


def dataset(dim=1, num=10000):
    return [np.random.rand(dim) for _ in range(num)]


class TestProximity(unittest.TestCase):

    def testTrivialProximity(self):
        data = dataset()
        proximity = TrivialProximity()
        charts = list(proximity_net(data, proximity))
        self.assertEqual(1, len(charts))

    def testBallProximity(self):
        data = [
            np.array([0.0, 1.0]), np.array([1.0, 0.0]),
            np.array([0.0, 0.0]), np.array([1.0, 1.0])]
        proximity = BallProximity(radius=1.1, metric=dist)
        charts = list(proximity_net(data, proximity))
        self.assertEqual(2, len(charts))

    def testKnnProximity(self):
        data = [
            np.array([0.0, 1.0]), np.array([1.1, 0.0]),
            np.array([0.0, 0.0]), np.array([1.1, 1.0])]
        proximity = KNNProximity(neighbors=2, metric=dist)
        charts = list(proximity_net(data, proximity))
        self.assertEqual(2, len(charts))

    def testCubicalProximity(self):
        data = [
            np.array([0.0, 1.0]), np.array([1.1, 0.0]),
            np.array([0.0, 0.0]), np.array([1.1, 1.0])]
        proximity = CubicalProximity(n_intervals=2, overlap_frac=0.5)
        charts = list(proximity_net(data, proximity))
        self.assertEqual(4, len(charts))
