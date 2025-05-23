import numpy as np

from tdamapper.core import TrivialCover
from tdamapper.cover import BallCover, CubicalCover, KNNCover


def dataset(dim=1, num=10000):
    return [np.random.rand(dim) for _ in range(num)]


def test_trivial_cover():
    data = dataset()
    cover = TrivialCover()
    charts = list(cover.apply(data))
    assert 1 == len(charts)


def test_ball_cover_simple():
    data = [
        np.array([0.0, 1.0]),
        np.array([1.0, 0.0]),
        np.array([0.0, 0.0]),
        np.array([1.0, 1.0]),
    ]
    cover = BallCover(radius=1.1, metric="euclidean")
    charts = list(cover.apply(data))
    assert 2 == len(charts)


def test_knn_cover_simple():
    data = [
        np.array([0.0, 1.0]),
        np.array([1.1, 0.0]),
        np.array([0.0, 0.0]),
        np.array([1.1, 1.0]),
    ]
    cover = KNNCover(neighbors=2, metric="euclidean")
    charts = list(cover.apply(data))
    assert 2 == len(charts)


def test_cubical_cover_simple():
    data = [
        np.array([0.0, 1.0]),
        np.array([1.1, 0.0]),
        np.array([0.0, 0.0]),
        np.array([1.1, 1.0]),
    ]
    cover = CubicalCover(n_intervals=2, overlap_frac=0.5)
    charts = list(cover.apply(data))
    assert 4 == len(charts)


def test_params():
    cover = CubicalCover(n_intervals=2, overlap_frac=0.5)
    params = cover.get_params(deep=True)
    assert 2 == params["n_intervals"]
    assert 0.5 == params["overlap_frac"]


def test_standard_cover_simple():
    data = [
        np.array([0.0, 1.0]),
        np.array([1.1, 0.0]),
        np.array([0.0, 0.0]),
        np.array([1.1, 1.0]),
    ]
    cover = CubicalCover(
        n_intervals=2,
        overlap_frac=0.5,
        algorithm="standard",
    )
    charts = list(cover.apply(data))
    assert 4 == len(charts)
