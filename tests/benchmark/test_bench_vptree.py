import unittest
import logging
from time import time

import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_digits

from tdamapper.utils.vptree_hier import VPTree as HVPT
from tdamapper.utils.vptree_flat import VPTree as FVPT
from tdamapper.utils.vptree_sklearn import VPTree as SKVPT
from tdamapper.utils.metrics import euclidean


def _dataset(dim=10, num=1000):
    return [np.random.rand(dim) for _ in range(num)]


class TestBenchmark(unittest.TestCase):

    eps = 0.25

    k = 5

    logger = logging.getLogger(__name__)

    logging.basicConfig(
        format = '%(asctime)s %(module)s %(levelname)s: %(message)s',
        datefmt = '%m/%d/%Y %I:%M:%S %p', 
        level = logging.INFO)

    def a_test_bench(self):
        self.logger.info('==== Dataset random =============')
        self._test_compare(_dataset())
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
        hvpt = self._test_build(data, ' *  HVPT  ', HVPT)
        fvpt = self._test_build(data, ' *  FVPT  ', FVPT)
        skvpt = self._test_build(data, ' * SKVPT  ', SKVPT)
        self.logger.info('[ball search]')
        self._test_ball_search_naive(data, ' * Naive ')
        self._test_ball_search(data, ' *  HVPT  ', hvpt)
        self._test_ball_search(data, ' *  FVPT  ', fvpt)
        self._test_ball_search(data, ' * SKVPT  ', skvpt)
        self.logger.info('[knn search]')
        self._test_knn_search_naive(data, ' * Naive ')
        self._test_knn_search(data, ' *  HVPT  ', hvpt)
        self._test_knn_search(data, ' *  FVPT  ', fvpt)
        self._test_knn_search(data, ' * SKVPT  ', skvpt)

    def _test_build(self, data, name, builder):
        t0 = time()
        vpt = builder(
            metric='euclidean',
            leaf_radius=self.eps,
            leaf_capacity=self.k,
            strategy='furthest')
        vpt.fit(data)
        t1 = time()
        self.logger.info(f'{name}: {t1 - t0}')
        return vpt

    def _test_ball_search_naive(self, data, name):
        t0 = time()
        dist = euclidean()
        for val in data:
            neigh = [x for x in data if dist(val, x) <= self.eps]
        t1 = time()
        self.logger.info(f'{name}: {t1 - t0}')

    def _test_ball_search(self, data, name, vpt):
        t0 = time()
        for val in data:
            neigh = vpt.ball_search(val, self.eps)
        t1 = time()
        self.logger.info(f'{name}: {t1 - t0}')

    def _test_knn_search_naive(self, data, name):
        t0 = time()
        dist = euclidean()
        for val in data:
            data.sort(key=lambda x: dist(x, val))
            neigh = [x for x in data[:self.k]]
        t1 = time()
        self.logger.info(f'{name}: {t1 - t0}')

    def _test_knn_search(self, data, name, vpt):
        t0 = time()
        for val in data:
            neigh = vpt.knn_search(val, self.k)
        t1 = time()
        self.logger.info(f'{name}: {t1 - t0}')
