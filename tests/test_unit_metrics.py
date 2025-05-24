import numpy as np

import tdamapper.utils.metrics as metrics


def test_euclidean():
    d = metrics.euclidean()
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    ab = d(a, b)
    assert ab >= 1.414
    assert ab <= 1.415


def test_manhattan():
    d = metrics.manhattan()
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    ab = d(a, b)
    assert ab == 2.0


def test_chebyshev():
    d = metrics.chebyshev()
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    ab = d(a, b)
    assert ab == 1.0


def test_cosine():
    d = metrics.cosine()
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    c = np.array([0.0, 2.0])
    ab = d(a, b)
    assert ab >= 1.414
    assert ab <= 1.415
    bc = d(b, c)
    assert bc == 0.0


def test_get_metric():
    assert metrics.euclidean() == metrics.get_metric("euclidean")
    assert metrics.euclidean() == metrics.get_metric("minkowski")
    assert metrics.chebyshev() == metrics.get_metric("chebyshev")
    assert metrics.chebyshev() == metrics.get_metric("minkowski", p=np.inf)
    assert metrics.chebyshev() == metrics.get_metric("minkowski", p=float("inf"))
    assert metrics.manhattan() == metrics.get_metric("manhattan")
    assert metrics.manhattan() == metrics.get_metric("minkowski", p=1)
    assert metrics.cosine() == metrics.get_metric("cosine")
