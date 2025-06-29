import numpy as np

from tdamapper.core import TrivialCover
from tdamapper.cover import (
    BallCover,
    KNNCover,
    ProximityCubicalCover,
    StandardCubicalCover,
)


def test_trivial_cover_empty():
    data = []
    cover = TrivialCover()
    cover.fit(data)
    charts = list(cover.transform(data))
    assert 0 == len(charts)


def test_trivial_cover_ok():
    data = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [1.0, 1.0],
        ]
    )
    cover = TrivialCover()
    cover.fit(data)
    charts = list(cover.transform(data))
    assert 1 == len(charts)


def test_ball_cover_empty():
    data = []
    cover = BallCover(radius=1.0, metric="euclidean")
    cover.fit(data)
    charts = list(cover.transform(data))
    assert 0 == len(charts)


def test_ball_cover_ok():
    data = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [1.0, 1.0],
        ]
    )
    cover = BallCover(radius=1.1, metric="euclidean")
    cover.fit(data)
    charts = list(cover.transform(data))
    assert 2 == len(charts)

    data_extra = np.array([[1.25, 1.25], [1.7, 1.7]])
    charts = list(cover.transform(data_extra))
    assert 1 == len(charts)
    assert 2 == len(charts[0])


def test_ball_cover_params():
    cover = BallCover(radius=1.0, metric="euclidean")
    params = cover.get_params(deep=True)
    assert 1.0 == params["radius"]
    assert "euclidean" == params["metric"]


def test_knn_cover_empty():
    data = []
    cover = KNNCover(neighbors=2, metric="euclidean")
    cover.fit(data)
    charts = list(cover.transform(data))
    assert 0 == len(charts)


def test_knn_cover_ok():
    data = np.array(
        [
            [0.0, 1.0],
            [1.1, 0.0],
            [0.0, 0.0],
            [1.1, 1.0],
        ]
    )
    cover = KNNCover(neighbors=2, metric="euclidean")
    cover.fit(data)
    charts = list(cover.transform(data))
    assert 2 == len(charts)

    data_extra = np.array([[1.25, 1.25], [1.7, 1.7]])
    charts = list(cover.transform(data_extra))
    assert 1 == len(charts)
    assert 2 == len(charts[0])


def test_knn_cover_params():
    cover = KNNCover(neighbors=2, metric="euclidean")
    params = cover.get_params(deep=True)
    assert 2 == params["neighbors"]
    assert "euclidean" == params["metric"]


def test_proximity_cubical_cover_empty():
    data = []
    cover = ProximityCubicalCover(n_intervals=2, overlap_frac=0.5)
    cover.fit(data)
    charts = list(cover.transform(data))
    assert 0 == len(charts)


def test_proximity_cubical_cover_ok():
    data = np.array(
        [
            [0.0, 1.0],
            [1.1, 0.0],
            [0.0, 0.0],
            [1.1, 1.0],
        ]
    )
    cover = ProximityCubicalCover(n_intervals=2, overlap_frac=0.5)
    cover.fit(data)
    charts = list(cover.transform(data))
    assert 4 == len(charts)

    data_extra = np.array([[1.25, 1.25], [1.26, 1.26]])
    charts = list(cover.transform(data_extra))
    assert 1 == len(charts)
    assert 2 == len(charts[0])


def test_proximity_cubical_cover_params():
    cover = ProximityCubicalCover(n_intervals=2, overlap_frac=0.5)
    params = cover.get_params(deep=True)
    assert 2 == params["n_intervals"]
    assert 0.5 == params["overlap_frac"]


def test_standard_cover_empty():
    data = []
    cover = StandardCubicalCover(
        n_intervals=2,
        overlap_frac=0.5,
    )
    cover.fit(data)
    charts = list(cover.transform(data))
    assert 0 == len(charts)


def test_standard_cover_ok():
    data = np.array(
        [
            [0.0, 1.0],
            [1.1, 0.0],
            [0.0, 0.0],
            [1.1, 1.0],
        ]
    )
    cover = StandardCubicalCover(
        n_intervals=2,
        overlap_frac=0.5,
    )
    cover.fit(data)
    charts = list(cover.transform(data))
    assert 4 == len(charts)


def test_standard_cover_params():
    cover = StandardCubicalCover(
        n_intervals=2,
        overlap_frac=0.5,
    )
    params = cover.get_params(deep=True)
    assert 2 == params["n_intervals"]
    assert 0.5 == params["overlap_frac"]
