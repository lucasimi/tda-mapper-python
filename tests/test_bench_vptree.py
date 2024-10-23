import unittest
import logging
from time import time

import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_digits

from tdamapper.utils.metrics import get_metric, euclidean
from tdamapper.utils.vptree_hier import VPTree as HVPT
from tdamapper.utils.vptree_flat import VPTree as FVPT

from tests.ball_tree import SkBallTree
from tests.setup_logging import setup_logging


dist = euclidean()


dist(np.array([0.0]), np.array([0.0]))  # jit-compile numba


def dataset(dim=10, num=1000):
    return [np.random.rand(dim) for _ in range(num)]


class TestBenchmark(unittest.TestCase):

    setup_logging()
    logger = logging.getLogger(__name__)

    eps = 0.25

    k = 5

    def test_bench(self):
        self.logger.info('==== Dataset random =============')
        self._test_compare(dataset())
        self.logger.info('==== Dataset iris ===============')
        iris, _ = load_iris(as_frame=True, return_X_y=True)
        self._test_compare(list(iris.to_numpy()))
        self.logger.info('==== Dataset breast_cancer ======')
        breast_cancer, _ = load_breast_cancer(as_frame=True, return_X_y=True)
        self._test_compare(list(breast_cancer.to_numpy()))
        self.logger.info('==== Dataset digits =============')
        digits, _ = load_digits(as_frame=True, return_X_y=True)
        self._test_compare(list(digits.to_numpy()))

    def _test_compare(self, data):
        self.logger.info('[build]')
        hvpt = self._test_build(data, ' * HVPT ', HVPT)
        fvpt = self._test_build(data, ' * FVPT ', FVPT)
        skbt = self._test_build(data, ' * SKBT', SkBallTree)
        self.logger.info('[ball search]')
        self._test_ball_search_naive(data, ' * Naive ')
        self._test_ball_search(data, ' * HVPT ', hvpt)
        self._test_ball_search(data, ' * FVPT ', fvpt)
        self._test_ball_search(data, ' * SKBT', skbt)
        self.logger.info('[knn search]')
        self._test_knn_search_naive(data, ' * Naive ')
        self._test_knn_search(data, ' * HVPT ', hvpt)
        self._test_knn_search(data, ' * FVPT ', fvpt)
        self._test_knn_search(data, ' * SKBT ', skbt)

    def _test_build(self, data, name, builder):
        t0 = time()
        vpt = builder(data, metric=dist, leaf_radius=self.eps, leaf_capacity=self.k, pivoting='furthest')
        t1 = time()
        self.logger.info(f'{name}: {t1 - t0}')
        return vpt

    def _test_ball_search_naive(self, data, name):
        d = get_metric(dist)
        d(np.array([0.0]), np.array([0.0]))  # jit-compile numba
        t0 = time()
        for val in data:
            neigh = [x for x in data if d(val, x) <= self.eps]
        t1 = time()
        self.logger.info(f'{name}: {t1 - t0}')

    def _test_ball_search(self, data, name, vpt):
        t0 = time()
        for val in data:
            neigh = vpt.ball_search(val, self.eps)
        t1 = time()
        self.logger.info(f'{name}: {t1 - t0}')

    def _test_knn_search_naive(self, data, name):
        d = get_metric(dist)
        d(np.array([0.0]), np.array([0.0]))  # jit-compile numba
        t0 = time()
        for val in data:
            data.sort(key=lambda x: d(x, val))
            neigh = [x for x in data[:self.k]]
        t1 = time()
        self.logger.info(f'{name}: {t1 - t0}')

    def _test_knn_search(self, data, name, vpt):
        t0 = time()
        for val in data:
            neigh = vpt.knn_search(val, self.k)
        t1 = time()
        self.logger.info(f'{name}: {t1 - t0}')
