import unittest
import time

import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

import tdamapper.utils.cython.metrics as metrics

from tdamapper.utils.vptree import VPTree as HVPT
from tdamapper.utils.vptree_flat import VPTree as FVPT
from tdamapper.utils.cython.vptree_flat import VPTree as CVPT


def dist_(metric):
    m = dist(metric)
    return lambda x, y: m(x[1:], y[1:])


def dist(metric):
    return metrics.get_metric(metric)


def dataset(dim=10, num=1000):
    return [np.random.rand(dim) for _ in range(num)]


def dataset_(dim=10, num=1000):
    X = dataset(dim, num)
    X_aug = []
    for i, x in enumerate(X):
        X_aug.append(np.concatenate((i, x)))
    return np.array(X_aug)


def cover(vpt, X, r):
    covered_ids = set()
    for i, xi in enumerate(X):
        if i not in covered_ids:
            neigh = vpt.ball_search(xi, r)
            neigh_ids = [int(x[0]) for x in neigh]
            covered_ids.update(neigh_ids)
            if neigh_ids:
                yield neigh_ids


def run_(X, r, d, vp, **kwargs):
    X = dataset_(X)
    d = dist_(d)
    run(X, r, d, vp, **kwargs)


def run(X, r, d, vp, **kwargs):
    t0 = time.time()
    vpt = vp(distance=d, dataset=X, **kwargs)
    list(cover(vpt, X, r))
    t1 = time.time()
    print(f'time: {t1 - t0}')


class TestVpSettings(unittest.TestCase):

    def test_cover_random(self):
        for r in [1.0, 10.0, 100.0]:
            for n in [100, 1000]:
                print(f'============= n: {n}, r: {r} =============')
                X = dataset(num=n)
                print('>>>>>>> CVPT >>>>>>')
                run(X, r, 'euclidean', CVPT, leaf_radius=r)
                print('>>>>>>> HVPT >>>>>>')
                #run(X, r, dist, HVPT)
                run(X, r, 'euclidean', HVPT, leaf_radius=r)
                print('>>>>>>> FVPT >>>>>>')
                #run(X, r, dist, FVPT)
                run(X, r, 'euclidean', FVPT, leaf_radius=r)
                print('')

    def test_cover_digits(self):
        X, _ = load_digits(return_X_y=True)
        X = PCA(n_components=2).fit_transform(X)
        X = list(X)
        for r in [1.0, 10.0, 100.0]:
            print(f'============= r: {r} =============')
            print('>>>>>>> CVPT >>>>>>')
            run(X, r, 'euclidean', CVPT, leaf_radius=r)
            print('>>>>>>> HVPT >>>>>>')
            #run(X, r, dist, HVPT)
            run(X, r, 'euclidean', HVPT, leaf_radius=r)
            print('>>>>>>> FVPT >>>>>>')
            #run(X, r, dist, FVPT)
            run(X, r, 'euclidean', FVPT, leaf_radius=r)
            print('')