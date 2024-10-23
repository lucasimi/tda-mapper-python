import unittest
import logging
import time

import numpy as np
from sklearn.datasets import load_digits

from tdamapper.utils.metrics import euclidean
from tdamapper.utils.vptree_hier import VPTree as HVPT
from tdamapper.utils.vptree_flat import VPTree as FVPT

from tests.ball_tree import SkBallTree
from tests.setup_logging import setup_logging


dist = euclidean()


dist(np.array([0.0]), np.array([0.0]))  # jit-compile numba


def dataset(dim=10, num=1000):
    return [np.random.rand(dim) for _ in range(num)]


class TestVpSettings(unittest.TestCase):

    setup_logging()
    logger = logging.getLogger(__name__)

    def cover(self, vpt, X, r):
        covered_ids = set()
        for i, xi in enumerate(X):
            if i not in covered_ids:
                neigh = vpt.ball_search(xi, r)
                neigh_ids = [int(x[0]) for x in neigh]
                covered_ids.update(neigh_ids)
                if neigh_ids:
                    yield neigh_ids

    def run_bench(self, X, r, dist, vp, **kwargs):
        XX = np.array([[i] + [xi for xi in x] for i, x in enumerate(X)])
        d = lambda x, y: dist(x[1:], y[1:])
        t0 = time.time()
        vpt = vp(XX, metric=d, **kwargs)
        list(self.cover(vpt, XX, r))
        t1 = time.time()
        self.logger.info(f'time: {t1 - t0}')

    def test_cover_random(self):
        for r in [1.0, 10.0, 100.0]:
            for n in [100, 1000, 10000]:
                self.logger.info(f'============ Cover Bench Random ==========')
                self.logger.info(f'[n: {n}, r: {r}]')
                X = dataset(num=n)
                self.logger.info('>>>>>>> HVPT >>>>>>')
                self.run_bench(X, r, dist, HVPT, leaf_radius=r, pivoting='random')
                self.run_bench(X, r, dist, HVPT, leaf_radius=r, pivoting='furthest')
                self.logger.info('>>>>>>> FVPT >>>>>>')
                self.run_bench(X, r, dist, FVPT, leaf_radius=r, pivoting='random')
                self.run_bench(X, r, dist, FVPT, leaf_radius=r, pivoting='furthest')
                self.logger.info('>>>>>> SKBT >>>>>>')
                self.run_bench(X, r, dist, SkBallTree)
                self.run_bench(X, r, dist, SkBallTree, leaf_radius=r)
                self.logger.info('')

    def test_cover_digits(self):
        X, _ = load_digits(return_X_y=True)
        #X = PCA(n_components=3).fit_transform(X)
        for r in [1.0, 10.0, 100.0]:
            self.logger.info(f'======= Cover Bench Digits =======')
            self.logger.info(f'[r: {r}]')
            self.logger.info('>>>>>>> HVPT >>>>>>')
            self.run_bench(X, r, dist, HVPT, leaf_radius=r, pivoting='random')
            self.run_bench(X, r, dist, HVPT, leaf_radius=r, pivoting='furthest')
            self.logger.info('>>>>>>> FVPT >>>>>>')
            self.run_bench(X, r, dist, FVPT, leaf_radius=r, pivoting='random')
            self.run_bench(X, r, dist, FVPT, leaf_radius=r, pivoting='furthest')
            self.logger.info('>>>>>> SKBT >>>>>>')
            self.run_bench(X, r, dist, SkBallTree)
            self.run_bench(X, r, dist, SkBallTree, leaf_radius=r)
            self.logger.info('')
