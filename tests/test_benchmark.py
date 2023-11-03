import unittest
import random
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
        self._testCompare(dataset(), 'random')
        iris, _ = load_iris(as_frame=True, return_X_y=True)
        self._testCompare(list(iris.to_numpy()), 'iris')
        breast_cancer, _ = load_breast_cancer(as_frame=True, return_X_y=True)
        self._testCompare(list(breast_cancer.to_numpy()), 'breast_cancer')
    
    def _testCompare(self, data, name):
        vpt = self._testBuild(data, f' VPT {name}', VPT)
        fvpt = self._testBuild(data, f'FVPT {name}', FVPT)
        self._testBallSearch(data, f' VPT {name}', vpt)
        self._testBallSearch(data, f'FVPT {name}', fvpt)
        self._testKNNSearch(data, f' VPT {name}', vpt)
        self._testKNNSearch(data, f'FVPT {name}', fvpt)

    def _testBuild(self, data, name, builder):
        t0 = time()
        vpt = builder(dist, data, leaf_radius=self.eps, leaf_size=self.k)
        t1 = time()
        self.logger.info(f'{name} build      : {t1 - t0}')
        return vpt

    def _testBallSearch(self, data, name, vpt):
        t0 = time()
        for val in data:
            neigh = vpt.ball_search(val, self.eps)
        t1 = time()
        self.logger.info(f'{name} ball search: {t1 - t0}')

    def _testKNNSearch(self, data, name, vpt):
        t0 = time()
        for val in data:
            neigh = vpt.knn_search(val, self.k)
        t1 = time()
        self.logger.info(f'{name} knn search : {t1 - t0}')
