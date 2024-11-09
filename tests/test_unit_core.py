import unittest

import numpy as np
import networkx as nx
from sklearn.cluster import DBSCAN

from tdamapper.core import (
    MapperAlgorithm,
    mapper_connected_components,
    mapper_labels,
    TrivialCover,
)
from tdamapper.cover import BallCover
from tdamapper.clustering import TrivialClustering


dist = 'euclidean'


def dataset(dim=10, num=1000):
    return [np.random.rand(dim) for _ in range(num)]


class TestMapper(unittest.TestCase):

    def test_trivial(self):
        data = dataset()
        mp = MapperAlgorithm(TrivialCover(), TrivialClustering())
        g = mp.fit_transform(data, data)
        self.assertEqual(1, len(g))
        self.assertEqual([], list(g.neighbors(0)))
        ccs = list(nx.connected_components(g))
        self.assertEqual(1, len(ccs))
        ccs2 = mapper_connected_components(data, data, TrivialCover(), TrivialClustering())
        self.assertEqual(len(data), len(ccs2))

    def test_ball_small_radius(self):
        data = np.array([[float(i)] for i in range(1000)])
        cover = BallCover(0.5, metric=dist)
        clustering = TrivialClustering()
        mp = MapperAlgorithm(cover, clustering)
        g = mp.fit_transform(data, data)
        self.assertEqual(1000, len(g))
        for node in g.nodes():
            self.assertEqual([], list(g.neighbors(node)))
        ccs = list(nx.connected_components(g))
        self.assertEqual(1000, len(ccs))
        ccs2 = mapper_connected_components(data, data, cover, clustering)
        self.assertEqual(len(data), len(ccs2))

    def test_ball_small_radius_list(self):
        data = [np.array([float(i)]) for i in range(1000)]
        cover = BallCover(0.5, metric=dist)
        clustering = DBSCAN(eps=1.0, min_samples=1)
        mp = MapperAlgorithm(
            cover=cover,
            clustering=clustering)
        g = mp.fit_transform(data, data)
        self.assertEqual(1000, len(g))
        for node in g.nodes():
            self.assertEqual([], list(g.neighbors(node)))
        ccs = list(nx.connected_components(g))
        self.assertEqual(1000, len(ccs))
        ccs2 = mapper_connected_components(data, data, cover, clustering)
        self.assertEqual(len(data), len(ccs2))

    def test_ball_large_radius(self):
        data = np.array([[float(i)] for i in range(1000)])
        cover = BallCover(1000.0, metric=dist)
        clustering = TrivialClustering()
        mp = MapperAlgorithm(
            cover=cover,
            clustering=clustering)
        g = mp.fit_transform(data, data)
        self.assertEqual(1, len(g))
        for node in g.nodes():
            self.assertEqual([], list(g.neighbors(node)))
        ccs = list(nx.connected_components(g))
        self.assertEqual(1, len(ccs))
        ccs2 = mapper_connected_components(data, data, cover, clustering)
        self.assertEqual(len(data), len(ccs2))

    def test_two_disconnected_clusters(self):
        data = [np.array([float(i), 0.0]) for i in range(100)]
        data.extend([np.array([float(i), 500.0]) for i in range(100)])
        data = np.array(data)
        cover = BallCover(150.0, metric=dist)
        clustering = TrivialClustering()
        mp = MapperAlgorithm(
            cover=cover,
            clustering=clustering)
        g = mp.fit_transform(data, data)
        self.assertEqual(2, len(g))
        for node in g.nodes():
            self.assertEqual([], list(g.neighbors(node)))
        ccs = list(nx.connected_components(g))
        self.assertEqual(2, len(ccs))
        ccs2 = mapper_connected_components(data, data, cover, clustering)
        self.assertEqual(len(data), len(ccs2))

    def test_two_connected_clusters(self):
        data = [
            np.array([0.0, 1.0]), np.array([1.0, 0.0]),
            np.array([0.0, 0.0]), np.array([1.0, 1.0])]
        cover = BallCover(1.1, metric=dist)
        clustering = TrivialClustering()
        mp = MapperAlgorithm(
            cover=cover,
            clustering=clustering)
        g = mp.fit_transform(data, data)
        self.assertEqual(2, len(g))
        for node in g.nodes():
            self.assertEqual(1, len(list(g.neighbors(node))))
        ccs = list(nx.connected_components(g))
        self.assertEqual(1, len(ccs))
        ccs2 = mapper_connected_components(data, data, cover, clustering)
        self.assertEqual(len(data), len(ccs2))

    def test_two_connected_clusters_parallel(self):
        data = [
            np.array([0.0, 1.0]), np.array([1.0, 0.0]),
            np.array([0.0, 0.0]), np.array([1.0, 1.0])]
        cover = BallCover(1.1, metric=dist)
        clustering = TrivialClustering()
        mp = MapperAlgorithm(
            cover=cover,
            clustering=clustering,
            n_jobs=4,
        )
        g = mp.fit_transform(data, data)
        self.assertEqual(2, len(g))
        for node in g.nodes():
            self.assertEqual(1, len(list(g.neighbors(node))))
        ccs = list(nx.connected_components(g))
        self.assertEqual(1, len(ccs))
        ccs2 = mapper_connected_components(data, data, cover, clustering)
        self.assertEqual(len(data), len(ccs2))

    def test_connected_components(self):
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
        self.assertEqual(len(data), len(ccs))
        cc0 = ccs[0]
        self.assertEqual(cc0, ccs[1])
        self.assertEqual(cc0, ccs[2])
        self.assertEqual(cc0, ccs[3])

    def test_labels(self):
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
        self.assertEqual(len(data), len(labels))
        self.assertEqual([0, 3], labels[0])
        self.assertEqual([1, 2, 3], labels[1])
        self.assertEqual([2], labels[2])
        self.assertEqual([0, 1, 3], labels[3])
