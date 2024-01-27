import unittest
import logging
from time import time
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_digits
from tdamapper.utils.vptree import VPTree as VPT
from tdamapper.utils.vptree_flat import VPTree as FVPT


def dist(x, y):
    return np.linalg.norm(x - y)


def dataset(dim=10, num=1000):
    return [np.random.rand(dim) for _ in range(num)]


class TestBenchmark(unittest.TestCase):

    eps = 0.25

    k = 5

    logger = logging.getLogger(__name__)

    logging.basicConfig(
        format = '%(asctime)s %(module)s %(levelname)s: %(message)s',
        datefmt = '%m/%d/%Y %I:%M:%S %p', 
        level = logging.INFO)

    def testBench(self):
        self.logger.info('==== Dataset random =============')
        self._testCompare(dataset())
        self.logger.info('==== Dataset iris ===============')
        iris, _ = load_iris(as_frame=True, return_X_y=True)
        self._testCompare(list(iris.to_numpy()))
        self.logger.info('==== Dataset breast_cancer ======')
        breast_cancer, _ = load_breast_cancer(as_frame=True, return_X_y=True)
        self._testCompare(list(breast_cancer.to_numpy()))
        self.logger.info('==== Dataset digits =============')
        digits, _ = load_digits(as_frame=True, return_X_y=True)
        self._testCompare(list(digits.to_numpy()))
    
    def _testCompare(self, data):
        self.logger.info('[build]')
        vpt = self._testBuild(data, ' * VPT  ', VPT)
        fvpt = self._testBuild(data, ' * FVPT ', FVPT)
        self.logger.info('[ball search]')
        self._testBallSearchNaive(data, ' * Naive ')
        self._testBallSearch(data, ' * VPT  ', vpt)
        self._testBallSearch(data, ' * FVPT ', fvpt)
        self.logger.info('[knn search]')
        self._testKNNSearchNaive(data, ' * Naive ')
        self._testKNNSearch(data, ' * VPT  ', vpt)
        self._testKNNSearch(data, ' * FVPT ', fvpt)

    def _testBuild(self, data, name, builder):
        t0 = time()
        vpt = builder(dist, data, leaf_radius=self.eps, leaf_capacity=self.k, pivoting='furthest')
        t1 = time()
        self.logger.info(f'{name}: {t1 - t0}')
        return vpt

    def _testBallSearchNaive(self, data, name):
        t0 = time()
        for val in data:
            neigh = [x for x in data if dist(val, x) <= self.eps]
        t1 = time()
        self.logger.info(f'{name}: {t1 - t0}')

    def _testBallSearch(self, data, name, vpt):
        t0 = time()
        for val in data:
            neigh = vpt.ball_search(val, self.eps)
        t1 = time()
        self.logger.info(f'{name}: {t1 - t0}')

    def _testKNNSearchNaive(self, data, name):
        t0 = time()
        for val in data:
            data.sort(key=lambda x: dist(x, val))
            neigh = [x for x in data[:self.k]]
        t1 = time()
        self.logger.info(f'{name}: {t1 - t0}')

    def _testKNNSearch(self, data, name, vpt):
        t0 = time()
        for val in data:
            neigh = vpt.knn_search(val, self.k)
        t1 = time()
        self.logger.info(f'{name}: {t1 - t0}')
