import networkx as nx
import numpy as np
import pytest
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

from tdamapper.core import (
    TrivialClustering,
    TrivialCover,
    mapper_connected_components,
    mapper_labels,
)
from tdamapper.cover import (
    BallCover,
    CubicalCover,
    ProximityCubicalCover,
    StandardCubicalCover,
)
from tdamapper.learn import MapperAlgorithm

dist = "euclidean"


def dataset(dim=10, num=1000):
    return [np.random.rand(dim) for _ in range(num)]


def test_trivial():
    data = dataset()
    mp = MapperAlgorithm(TrivialCover(), TrivialClustering())
    g = mp.fit_transform(data, data)
    assert 1 == len(g)
    assert [] == list(g.neighbors(0))
    ccs = list(nx.connected_components(g))
    assert 1 == len(ccs)
    ccs2 = mapper_connected_components(
        data,
        data,
        TrivialCover(),
        TrivialClustering(),
    )
    assert len(data) == len(ccs2)


def test_ball_small_radius():
    data = np.array([[float(i)] for i in range(1000)])
    cover = BallCover(0.5, metric=dist)
    clustering = TrivialClustering()
    mp = MapperAlgorithm(cover, clustering)
    g = mp.fit_transform(data, data)
    assert 1000 == len(g)
    for node in g.nodes():
        assert [] == list(g.neighbors(node))
    ccs = list(nx.connected_components(g))
    assert 1000 == len(ccs)
    ccs2 = mapper_connected_components(data, data, cover, clustering)
    assert len(data) == len(ccs2)


def test_ball_small_radius_list():
    data = [np.array([float(i)]) for i in range(1000)]
    cover = BallCover(0.5, metric=dist)
    clustering = DBSCAN(eps=1.0, min_samples=1)
    mp = MapperAlgorithm(cover=cover, clustering=clustering)
    g = mp.fit_transform(data, data)
    assert 1000 == len(g)
    for node in g.nodes():
        assert [] == list(g.neighbors(node))
    ccs = list(nx.connected_components(g))
    assert 1000 == len(ccs)
    ccs2 = mapper_connected_components(data, data, cover, clustering)
    assert len(data) == len(ccs2)


def test_ball_large_radius():
    data = np.array([[float(i)] for i in range(1000)])
    cover = BallCover(1000.0, metric=dist)
    clustering = TrivialClustering()
    mp = MapperAlgorithm(cover=cover, clustering=clustering)
    g = mp.fit_transform(data, data)
    assert 1 == len(g)
    for node in g.nodes():
        assert [] == list(g.neighbors(node))
    ccs = list(nx.connected_components(g))
    assert 1 == len(ccs)
    ccs2 = mapper_connected_components(data, data, cover, clustering)
    assert len(data) == len(ccs2)


def test_ball_two_disconnected_clusters():
    data = [np.array([float(i), 0.0]) for i in range(100)]
    data.extend([np.array([float(i), 500.0]) for i in range(100)])
    data = np.array(data)
    cover = BallCover(150.0, metric=dist)
    clustering = TrivialClustering()
    mp = MapperAlgorithm(cover=cover, clustering=clustering)
    g = mp.fit_transform(data, data)
    assert 2 == len(g)
    for node in g.nodes():
        assert [] == list(g.neighbors(node))
    ccs = list(nx.connected_components(g))
    assert 2 == len(ccs)
    ccs2 = mapper_connected_components(data, data, cover, clustering)
    assert len(data) == len(ccs2)


def test_ball_two_connected_clusters():
    data = [
        np.array([0.0, 1.0]),
        np.array([1.0, 0.0]),
        np.array([0.0, 0.0]),
        np.array([1.0, 1.0]),
    ]
    cover = BallCover(1.1, metric=dist)
    clustering = TrivialClustering()
    mp = MapperAlgorithm(cover=cover, clustering=clustering)
    g = mp.fit_transform(data, data)
    assert 2 == len(g)
    for node in g.nodes():
        assert 1 == len(list(g.neighbors(node)))
    ccs = list(nx.connected_components(g))
    assert 1 == len(ccs)
    ccs2 = mapper_connected_components(data, data, cover, clustering)
    assert len(data) == len(ccs2)


def test_ball_two_connected_clusters_parallel():
    data = [
        np.array([0.0, 1.0]),
        np.array([1.0, 0.0]),
        np.array([0.0, 0.0]),
        np.array([1.0, 1.0]),
    ]
    cover = BallCover(1.1, metric=dist)
    clustering = TrivialClustering()
    mp = MapperAlgorithm(
        cover=cover,
        clustering=clustering,
        n_jobs=4,
    )
    g = mp.fit_transform(data, data)
    assert 2 == len(g)
    for node in g.nodes():
        assert 1 == len(list(g.neighbors(node)))
    ccs = list(nx.connected_components(g))
    assert 1 == len(ccs)
    ccs2 = mapper_connected_components(data, data, cover, clustering)
    assert len(data) == len(ccs2)


def test_proximity_cubical_line():
    data = np.array([[float(i)] for i in range(1000)])
    cover = ProximityCubicalCover(n_intervals=4, overlap_frac=0.5)
    clustering = TrivialClustering()
    mp = MapperAlgorithm(cover, clustering)
    g = mp.fit_transform(data, data)
    assert 4 == len(g.nodes)


def test_standard_cubical_line():
    data = np.array([[float(i)] for i in range(1000)])
    cover = StandardCubicalCover(n_intervals=4, overlap_frac=0.5)
    clustering = TrivialClustering()
    mp = MapperAlgorithm(cover, clustering)
    g = mp.fit_transform(data, data)
    assert 4 == len(g.nodes)


def test_cubical_line():
    data = np.array([[float(i)] for i in range(1000)])
    cover = CubicalCover(n_intervals=4, overlap_frac=0.5)
    clustering = TrivialClustering()
    mp = MapperAlgorithm(cover, clustering)
    g = mp.fit_transform(data, data)
    assert 4 == len(g.nodes)


def test_cubical_no_overlap():
    data = np.array([[0.0], [1.0], [2.0]])
    cover = StandardCubicalCover(n_intervals=2, overlap_frac=0)
    clustering = TrivialClustering()
    mp = MapperAlgorithm(cover, clustering)
    with pytest.raises(ValueError):
        mp.fit_transform(data, data)


def test_mock_connected_components():
    data = [0, 1, 2, 3]

    class MockCover:

        def apply(self, X):
            yield [0, 3]
            yield [1, 3]
            yield [1, 2]
            yield [0, 1, 3]

    cover = MockCover()
    clustering = TrivialClustering()
    ccs = mapper_connected_components(data, data, cover, clustering)
    assert len(data) == len(ccs)
    cc0 = ccs[0]
    assert cc0 == ccs[1]
    assert cc0 == ccs[2]
    assert cc0 == ccs[3]


def test_mock_labels():
    data = [0, 1, 2, 3]

    class MockCover:

        def apply(self, X):
            yield [0, 3]
            yield [1, 3]
            yield [1, 2]
            yield [0, 1, 3]

    cover = MockCover()
    clustering = TrivialClustering()
    labels = mapper_labels(data, data, cover, clustering)
    assert len(data) == len(labels)
    assert [0, 3] == labels[0]
    assert [1, 2, 3] == labels[1]
    assert [2] == labels[2]
    assert [0, 1, 3] == labels[3]


def test_full():
    X, _ = load_digits(return_X_y=True)
    y = PCA(2, random_state=42).fit_transform(X)
    mapper = MapperAlgorithm(
        cover=CubicalCover(n_intervals=10, overlap_frac=0.5),
        clustering=AgglomerativeClustering(10),
        verbose=False,
    )
    graph = mapper.fit_transform(X, y)
    assert 381 == len(graph.nodes())
    assert 736 == len(graph.edges())
