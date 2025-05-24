import networkx as nx
import numpy as np
from sklearn.utils.estimator_checks import check_estimator

from tdamapper.core import TrivialClustering, TrivialCover
from tdamapper.cover import BallCover
from tdamapper.learn import MapperAlgorithm, MapperClustering


def euclidean(x, y):
    return np.linalg.norm(x - y)


def dataset(dim=10, num=1000):
    return [np.random.rand(dim) for _ in range(num)]


def run_tests(estimator):
    for est, check in check_estimator(estimator, generate_only=True):
        check(est)


def test_mapper_learn():
    data = dataset()
    mp = MapperAlgorithm(TrivialCover(), TrivialClustering())
    g = mp.fit_transform(data, data)
    assert 1 == len(g)
    assert [] == list(g.neighbors(0))
    ccs = list(nx.connected_components(g))
    assert 1 == len(ccs)


def test_mapper_learn_est():
    est = MapperAlgorithm()
    run_tests(est)


def test_mapper_clustering_trivial():
    est = MapperClustering()
    run_tests(est)


def test_mapper_clustering_ball():
    est = MapperClustering(cover=BallCover(metric=euclidean))
    run_tests(est)
