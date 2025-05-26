import logging
import time

import numpy as np
from sklearn.datasets import load_digits

from tdamapper._common import profile
from tdamapper.utils.metrics import euclidean
from tdamapper.utils.vptree_flat.vptree import VPTree as FVPT
from tdamapper.utils.vptree_hier.vptree import VPTree as HVPT
from tests.ball_tree import SkBallTree
from tests.setup_logging import setup_logging

dist = euclidean()


dist(np.array([0.0]), np.array([0.0]))  # jit-compile numba


def dataset(dim=10, num=1000):
    return [np.random.rand(dim) for _ in range(num)]


def dist_proj(x, y):
    return dist(x[1:], x[1:])


setup_logging()
logger = logging.getLogger(__name__)


def cover(vpt, X, r):
    covered_ids = set()
    for i, xi in enumerate(X):
        if i not in covered_ids:
            neigh = vpt.ball_search(xi, r)
            neigh_ids = [int(x[0]) for x in neigh]
            covered_ids.update(neigh_ids)
            if neigh_ids:
                yield neigh_ids


def run_bench(X, r, dist, vp, **kwargs):
    XX = np.array([[i] + [xi for xi in x] for i, x in enumerate(X)])
    vpt = vp(XX, metric=dist_proj, **kwargs)  # first run of jit-compiled functions
    t0 = time.time()
    vpt = vp(XX, metric=dist_proj, **kwargs)
    list(cover(vpt, XX, r))
    t1 = time.time()
    logger.info(f"time: {t1 - t0}")


@profile(n_lines=20)
def test_cover_random():
    for r in [1.0, 10.0, 100.0]:
        for n in [100, 1000, 10000]:
            logger.info("============ Cover Bench Random ==========")
            logger.info(f"[n: {n}, r: {r}]")
            X = dataset(num=n)
            logger.info(">>>>>>> HVPT >>>>>>")
            run_bench(X, r, dist, HVPT, leaf_radius=r, pivoting=None)
            run_bench(X, r, dist, HVPT, leaf_radius=r, pivoting="random")
            run_bench(X, r, dist, HVPT, leaf_radius=r, pivoting="furthest")
            logger.info(">>>>>>> FVPT >>>>>>")
            run_bench(X, r, dist, FVPT, leaf_radius=r, pivoting=None)
            run_bench(X, r, dist, FVPT, leaf_radius=r, pivoting="random")
            run_bench(X, r, dist, FVPT, leaf_radius=r, pivoting="furthest")
            logger.info(">>>>>> SKBT >>>>>>")
            run_bench(X, r, dist, SkBallTree)
            run_bench(X, r, dist, SkBallTree, leaf_radius=r)
            logger.info("")


@profile(n_lines=20)
def test_cover_digits():
    X, _ = load_digits(return_X_y=True)
    # X = PCA(n_components=3).fit_transform(X)
    for r in [1.0, 10.0, 100.0]:
        logger.info("======= Cover Bench Digits =======")
        logger.info(f"[r: {r}]")
        logger.info(">>>>>>> HVPT >>>>>>")
        run_bench(X, r, dist, HVPT, leaf_radius=r, pivoting="random")
        run_bench(X, r, dist, HVPT, leaf_radius=r, pivoting="furthest")
        logger.info(">>>>>>> FVPT >>>>>>")
        run_bench(X, r, dist, FVPT, leaf_radius=r, pivoting="random")
        run_bench(X, r, dist, FVPT, leaf_radius=r, pivoting="furthest")
        logger.info(">>>>>> SKBT >>>>>>")
        run_bench(X, r, dist, SkBallTree)
        run_bench(X, r, dist, SkBallTree, leaf_radius=r)
        logger.info("")
