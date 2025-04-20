import unittest

import networkx as nx
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.utils.estimator_checks import check_estimator

from tdamapper.core import TrivialClustering, TrivialCover
from tdamapper.cover import BallCover
from tdamapper.learn import (
    MapperAlgorithm,
    MapperClustering,
)


def euclidean(x, y):
    return np.linalg.norm(x - y)


def dataset(dim=10, num=1000):
    return [np.random.rand(dim) for _ in range(num)]


class TestMapper(unittest.TestCase):

    def run_tests(self, estimator):
        for est, check in check_estimator(estimator, generate_only=True):
            check(est)

    def test_mapper_learn(self):
        data = dataset()
        mp = MapperAlgorithm(TrivialCover(), TrivialClustering())
        g = mp.fit_transform(data, data)
        self.assertEqual(1, len(g))
        self.assertEqual([], list(g.neighbors(0)))
        ccs = list(nx.connected_components(g))
        self.assertEqual(1, len(ccs))

    def test_mapper_learn_est(self):
        est = MapperAlgorithm()
        self.run_tests(est)

    def test_mapper_clustering_trivial(self):
        est = MapperClustering()
        self.run_tests(est)

    def test_mapper_clustering_ball(self):
        est = MapperClustering(cover=BallCover(metric=euclidean))
        self.run_tests(est)
