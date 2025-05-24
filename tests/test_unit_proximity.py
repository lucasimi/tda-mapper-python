import math

import numpy as np

from tdamapper.cover import BallCover, CubicalCover, KNNCover


def dataset(dim=1, num=10000):
    return [np.random.rand(dim) for _ in range(num)]


def absdist(x, y):
    return abs(x - y)


def test_ball_proximity():
    data = list(range(100))
    cover = BallCover(radius=10, metric=absdist)
    cover.fit(data)
    for x in data:
        result = cover.search(x)
        expected = [y for y in data if abs(x - y) < 10]
        assert len(expected) == len(result)


def test_knn_proximity():
    data = list(range(100))
    cover = KNNCover(neighbors=11, metric=absdist)
    cover.fit(data)
    for x in range(5, 94):
        result = cover.search(x)
        expected = [x + i for i in range(-5, 6)]
        assert set(expected) == set(result)


def test_cubical_proximity():
    m, M = 0, 99
    n = 10
    p = 0.1
    w = (M - m) / (n * (1.0 - p))
    delta = p * w
    data = list(range(m, M + 1))
    cover = CubicalCover(n_intervals=n, overlap_frac=p)
    cover.fit(data)
    for x in data[:-1]:
        result = cover.search(x)
        i = math.floor((x - m) / (w - delta))
        a_i = m + i * (w - delta) - delta / 2.0
        b_i = m + (i + 1) * (w - delta) + delta / 2.0
        expected = [y for y in data if y > a_i and y < b_i]
        for c in result:
            assert c in expected
        for c in expected:
            assert c in result
    x = data[-1]
    last_result = cover.search(x)
    assert result == last_result


def test_cubical_params():
    cover = CubicalCover(n_intervals=10, overlap_frac=0.5)
    params = cover.get_params()
    assert 10 == params["n_intervals"]
    assert 0.5 == params["overlap_frac"]
    cover.set_params(n_intervals=5, overlap_frac=0.25)
    params = cover.get_params()
    assert 5 == params["n_intervals"]
    assert 0.25 == params["overlap_frac"]


def test_knn_params():
    cover = KNNCover(neighbors=10, metric="chebyshev")
    params = cover.get_params()
    assert 10 == params["neighbors"]
    assert "chebyshev" == params["metric"]
    cover.set_params(neighbors=5, metric="euclidean")
    params = cover.get_params()
    assert 5 == params["neighbors"]
    assert "euclidean" == params["metric"]


def test_ball_params():
    cover = BallCover(radius=10.0, metric="chebyshev")
    params = cover.get_params()
    assert 10.0 == params["radius"]
    assert "chebyshev" == params["metric"]
    cover.set_params(radius=5.0, metric="euclidean")
    params = cover.get_params()
    assert 5.0 == params["radius"]
    assert "euclidean" == params["metric"]
