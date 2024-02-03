import unittest

import numpy as np
from sklearn.utils import check_X_y
from sklearn.utils.estimator_checks import check_estimator
from sklearn.cluster import KMeans

from tdamapper.clustering import (
    TrivialClustering,
    MapperClustering,
    FailSafeClustering
)
from tdamapper.cover import (
    TrivialCover,
    BallCover,
    KNNCover,
    CubicalCover
)


def euclidean(x, y):
    return np.linalg.norm(x - y)


class Estimator:

    def get_params(self, deep=True):
        params = {}
        for k, v in vars(self).items():
            if not k.startswith('_'):
                params[k] = v
        return params

    def set_params(self, **parmeters):
        for k, v in parmeters.items():
            if not k.startswith('_'):
                setattr(self, k, v)
        return self

    def get_clustering(self):
        return TrivialClustering()

    def fit(self, X, y=None):
        X, y = check_X_y(X, y)
        clustering = self.get_clustering()
        self.labels_ = clustering.fit(X, y).labels_
        self.n_features_in_ = X.shape[1]
        return self


class MapperClusteringEstimator(Estimator):

    def __init__(self,
            cover='cubical',
            n_intervals=10,
            overlap_frac=0.25,
            radius=0.5,
            neighbors=5,
            metric=euclidean):
        self.cover = cover
        self.n_intervals = n_intervals
        self.overlap_frac = overlap_frac
        self.radius = radius
        self.neighbors = neighbors
        self.metric = metric

    def __get_cover(self):
        if self.cover == 'trivial':
            return TrivialCover()
        elif self.cover == 'cubical':
            return CubicalCover(n_intervals=self.n_intervals, overlap_frac=self.overlap_frac)
        elif self.cover == 'ball':
            return BallCover(radius=self.radius, metric=self.metric)
        elif self.cover == 'knn':
            return KNNCover(neighbors=self.neighbors, metric=self.metric)
        else:
            raise ValueError(f'Unknown cover algorithm {self.cover}')

    def get_clustering(self):
        return MapperClustering(self.__get_cover())


class PermissiveKMeans(Estimator):

    def __init__(self, n=8):
        self.n = n

    def get_clustering(self):
        return FailSafeClustering(KMeans(n_clusters=self.n, n_init='auto'), verbose=False)


class TrivialClusteringEstimator(Estimator):

    def get_clustering(self):
        return TrivialClustering()


class TestSklearn(unittest.TestCase):

    def testClustering(self):
        check_estimator(TrivialClusteringEstimator())

    def testBall(self):
        check_estimator(MapperClusteringEstimator(cover='ball'))

    def testKNN(self):
        check_estimator(MapperClusteringEstimator(cover='knn'))

    def testCubical(self):
        check_estimator(MapperClusteringEstimator())

    def testPermissive(self):
        check_estimator(PermissiveKMeans())
