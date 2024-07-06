import unittest
import time

import numpy as np
from sklearn.datasets import load_digits

from tdamapper.utils.metrics import euclidean
from tdamapper.utils.vptree import VPTree as HVPT
from tdamapper.utils.vptree_flat import VPTree as FVPT

from tests.ball_tree import SkBallTree


dist = euclidean()


def dataset(dim=10, num=1000):
    return [np.random.rand(dim) for _ in range(num)]


def cover(vpt, X, r):
    covered_ids = set()
    for i, xi in enumerate(X):
        if i not in covered_ids:
            neigh = vpt.ball_search(xi, r)
            neigh_ids = [int(x[0]) for x in neigh]
            covered_ids.update(neigh_ids)
            if neigh_ids:
                yield neigh_ids


def run(X, r, dist, vp, **kwargs):
    XX = np.array([[1] + [xi for xi in x] for x in X])
    d = lambda x, y: dist(x[1:], y[1:])
    t0 = time.time()
    vpt = vp(dataset=XX, distance=d, **kwargs)
    list(cover(vpt, XX, r))
    t1 = time.time()
    print(f'time: {t1 - t0}')


class TestVpSettings(unittest.TestCase):

    def testCoverRandom(self):
        for r in [10.0, 100.0]:
            for n in [100, 1000]:
                print(f'============= n: {n}, r: {r} =============')
                X = dataset(num=n)
                print('>>>>>>> HVPT >>>>>>')
                run(X, r, dist, HVPT)
                run(X, r, dist, HVPT, leaf_radius=r)
                print('>>>>>>> FVPT >>>>>>')
                run(X, r, dist, FVPT)
                run(X, r, dist, FVPT, leaf_radius=r)
                print('>>>>>> SKBT >>>>>>')
                run(X, r, dist, SkBallTree)
                run(X, r, dist, SkBallTree, leaf_radius=r)
                print('')

    def testCoverDigits(self):
        X, _ = load_digits(return_X_y=True)
        #X = PCA(n_components=3).fit_transform(X)
        for r in [1.0, 10.0, 100.0]:
            print(f'============= r: {r} =============')
            print('>>>>>>> HVPT >>>>>>')
            run(X, r, dist, HVPT)
            run(X, r, dist, HVPT, leaf_radius=r)
            print('>>>>>>> FVPT >>>>>>')
            run(X, r, dist, FVPT)
            run(X, r, dist, FVPT, leaf_radius=r)
            print('>>>>>> SKBT >>>>>>')
            run(X, r, dist, SkBallTree)
            run(X, r, dist, SkBallTree, leaf_radius=r)
            print('')