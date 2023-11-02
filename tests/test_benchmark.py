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
        self._testBench(data, f' VPT {name}', VPT)
        self._testBench(data, f'FVPT {name}', FVPT)

    def _testBench(self, data, name, builder):
        eps = 0.25
        k = 5
        t0 = time()
        vpt = builder(dist, data, leaf_radius=eps, leaf_size=k)
        t1 = time()
        for val in data:
            neigh = vpt.ball_search(val, eps)
        t2 = time()
        #for val in data:
        #    neigh = vpt.knn_search(val, k)
        t3 = time()
        self.logger.info(f'{name} build      : {t1 - t0}')
        self.logger.info(f'{name} ball search: {t2 - t1}')
        self.logger.info(f'{name} knn search : {t3 - t2}')