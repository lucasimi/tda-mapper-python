import unittest
import logging

import numpy as np

from sklearn.utils.estimator_checks import check_estimator

from tdamapper.core import MapperAlgorithm
from tdamapper.cover import (
    BallCover,
    KNNCover,
    CubicalCover
)
from tdamapper.clustering import MapperClustering

from tests.setup_logging import setup_logging


def euclidean(x, y):
    return np.linalg.norm(x - y)


class TestSklearn(unittest.TestCase):

    setup_logging()
    logger = logging.getLogger(__name__)

    def run_tests(self, estimator):
        for est, check in check_estimator(estimator, generate_only=True):
            # self.logger.info(f'{check}')
            check(est)

    def test_trivial(self):
        est = MapperAlgorithm()
        self.run_tests(est)

    def test_ball(self):
        est = MapperAlgorithm(cover=BallCover(metric=euclidean))
        self.run_tests(est)

    def test_knn(self):
        est = MapperAlgorithm(cover=KNNCover(metric=euclidean))
        self.run_tests(est)

    def test_cubical(self):
        est = MapperAlgorithm(cover=CubicalCover())
        self.run_tests(est)

    def test_clustering_trivial(self):
        est = MapperClustering()
        self.run_tests(est)

    def test_clustering_ball(self):
        est = MapperClustering(cover=BallCover(metric=euclidean))
        self.run_tests(est)
