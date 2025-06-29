import numpy as np

from tdamapper.search import BallSearch, CubicalSearch, KNNSearch


def test_ball_search_empty():
    search = BallSearch(radius=1.0)
    data = np.array([])
    search.fit(data)

    assert 0 == len(search.search(np.array([0.0])))


def test_ball_search_ok():
    search = BallSearch(radius=1.0)
    data = np.array([[0.0], [1.0], [2.0]])
    search.fit(data)

    assert 1 == len(search.search(np.array([0.0])))

    assert 0 in search.search(np.array([0.0]))
    assert 1 not in search.search(np.array([0.0]))

    assert 0 in search.search(np.array([0.5]))
    assert 1 in search.search(np.array([0.5]))


def test_knn_search_empty():
    search = KNNSearch(neighbors=2)
    data = np.array([])
    search.fit(data)

    assert 0 == len(search.search(np.array([0.0])))


def test_knn_search_ok():
    search = KNNSearch(neighbors=2)
    data = np.array([[0.0], [1.0], [2.0], [3.0]])
    search.fit(data)

    assert 2 == len(search.search(np.array([0.0])))

    assert 0 in search.search(np.array([0.0]))
    assert 1 in search.search(np.array([0.0]))

    assert 0 in search.search(np.array([0.5]))
    assert 1 in search.search(np.array([0.5]))


def test_cubical_search_empty():
    search = CubicalSearch(n_intervals=4, overlap_frac=0.25)
    data = np.array([])
    search.fit(data)

    assert 0 == len(search.search(np.array([1.0])))


def test_cubical_search_ok():
    search = CubicalSearch(n_intervals=4, overlap_frac=0.25)
    data = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    search.fit(data)

    assert 2 == len(search.search(np.array([1.0])))

    assert 1 in search.search(np.array([1.0]))
    assert 2 in search.search(np.array([1.0]))
