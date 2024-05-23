import unittest
import time

import numpy as np
from tdamapper.utils.vptree import VPTree as VPT
from tdamapper.utils.vptree_alt import VPTree as VPTA


def dist(x, y):
    return np.linalg.norm(x - y)


def dataset(dim=10, num=1000):
    return [np.random.rand(dim) for _ in range(num)]


def cover(vpt, X, r):
    covered_ids = set()
    for i, xi in enumerate(X):
        if i not in covered_ids:
            neigh = vpt.ball_search(xi, r)
            neigh_ids = [x for x, _ in neigh]
            covered_ids.update(neigh_ids)
            if neigh_ids:
                yield neigh_ids


def run(X, r, dist, vp, **kwargs):
    t0 = time.time()
    XX = list(enumerate(X))
    d = lambda x, y: dist(x[1], y[1])
    vpt = vp(dataset=XX, distance=d, **kwargs)
    list(cover(vpt, XX, r))
    t1 = time.time()
    print(f'time: {t1 - t0}')


class TestVpSettings(unittest.TestCase):

    def testCover(self):
        for r in [1.0, 10.0, 100.0]:
            for n in [100, 1000, 10000]:
                print(f'============= n: {n}, r: {r} =============')
                X = dataset(num=n)
                print('>>>>>> VPT >>>>>>')
                run(X, r, dist, VPT)
                run(X, r, dist, VPT, leaf_radius=r)
                run(X, r, dist, VPT, leaf_radius=r, pivoting='random')
                run(X, r, dist, VPT, leaf_radius=r, pivoting='furthest')
                print('>>>>>> VPTA >>>>>')
                run(X, r, dist, VPTA)
                run(X, r, dist, VPTA, leaf_radius=r)
                run(X, r, dist, VPTA, leaf_radius=r, pivoting='random')
                run(X, r, dist, VPTA, leaf_radius=r, pivoting='furthest')
                print('')

