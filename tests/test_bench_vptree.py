import logging
from time import time

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, load_iris

from tdamapper._common import profile
from tdamapper.utils.metrics import euclidean, get_metric
from tdamapper.utils.vptree_flat.vptree import VPTree as FVPT
from tdamapper.utils.vptree_hier.vptree import VPTree as HVPT
from tests.ball_tree import SkBallTree
from tests.setup_logging import setup_logging

dist = euclidean()


dist(np.array([0.0]), np.array([0.0]))  # jit-compile numba


def dataset(dim=10, num=1000):
    return [np.random.rand(dim) for _ in range(num)]


setup_logging()
logger = logging.getLogger(__name__)

eps = 0.25

k = 5


@profile(n_lines=20)
def test_bench():
    logger.info("==== Dataset random =============")
    _test_compare(dataset())
    logger.info("==== Dataset iris ===============")
    iris, _ = load_iris(as_frame=True, return_X_y=True)
    _test_compare(list(iris.to_numpy()))
    logger.info("==== Dataset breast_cancer ======")
    breast_cancer, _ = load_breast_cancer(as_frame=True, return_X_y=True)
    _test_compare(list(breast_cancer.to_numpy()))
    logger.info("==== Dataset digits =============")
    digits, _ = load_digits(as_frame=True, return_X_y=True)
    _test_compare(list(digits.to_numpy()))


def _test_compare(data):
    logger.info("[build]")
    hvpt = _test_build(data, " * HVPT ", HVPT)
    fvpt = _test_build(data, " * FVPT ", FVPT)
    skbt = _test_build(data, " * SKBT", SkBallTree)
    logger.info("[ball search]")
    _test_ball_search_naive(data, " * Naive ")
    _test_ball_search(data, " * HVPT ", hvpt)
    _test_ball_search(data, " * FVPT ", fvpt)
    _test_ball_search(data, " * SKBT", skbt)
    logger.info("[knn search]")
    _test_knn_search_naive(data, " * Naive ")
    _test_knn_search(data, " * HVPT ", hvpt)
    _test_knn_search(data, " * FVPT ", fvpt)
    _test_knn_search(data, " * SKBT ", skbt)


def _test_build(data, name, builder):
    t0 = time()
    vpt = builder(
        data,
        metric=dist,
        leaf_radius=eps,
        leaf_capacity=k,
        pivoting="furthest",
    )
    t1 = time()
    logger.info(f"{name}: {t1 - t0}")
    return vpt


def _test_ball_search_naive(data, name):
    d = get_metric(dist)
    d(np.array([0.0]), np.array([0.0]))  # jit-compile numba
    t0 = time()
    for val in data:
        [x for x in data if d(val, x) <= eps]
    t1 = time()
    logger.info(f"{name}: {t1 - t0}")


def _test_ball_search(data, name, vpt):
    t0 = time()
    for val in data:
        vpt.ball_search(val, eps)
    t1 = time()
    logger.info(f"{name}: {t1 - t0}")


def _test_knn_search_naive(data, name):
    d = get_metric(dist)
    d(np.array([0.0]), np.array([0.0]))  # jit-compile numba
    t0 = time()
    for val in data:

        def _dist_key(x):
            return d(x, val)

        data.sort(key=_dist_key)
        [x for x in data[:k]]
    t1 = time()
    logger.info(f"{name}: {t1 - t0}")


def _test_knn_search(data, name, vpt):
    t0 = time()
    for val in data:
        vpt.knn_search(val, k)
    t1 = time()
    logger.info(f"{name}: {t1 - t0}")
