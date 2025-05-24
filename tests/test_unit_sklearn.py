import logging

import numpy as np
from sklearn.utils.estimator_checks import check_estimator

from tdamapper.cover import BallCover, CubicalCover, KNNCover
from tdamapper.learn import MapperAlgorithm, MapperClustering
from tests.setup_logging import setup_logging


def euclidean(x, y):
    return np.linalg.norm(x - y)


setup_logging()
logger = logging.getLogger(__name__)


def run_tests(estimator):
    for est, check in check_estimator(estimator, generate_only=True):
        # logger.info(f'{check}')
        check(est)


def test_trivial():
    est = MapperAlgorithm()
    run_tests(est)


def test_ball():
    est = MapperAlgorithm(cover=BallCover(metric=euclidean))
    run_tests(est)


def test_knn():
    est = MapperAlgorithm(cover=KNNCover(metric=euclidean))
    run_tests(est)


def test_cubical():
    est = MapperAlgorithm(cover=CubicalCover())
    run_tests(est)


def test_clustering_trivial():
    est = MapperClustering()
    run_tests(est)


def test_clustering_ball():
    est = MapperClustering(cover=BallCover(metric=euclidean))
    run_tests(est)
