import unittest
import numpy as np
from sklearn.utils import check_X_y
from sklearn.utils.estimator_checks import check_estimator
from sklearn.cluster import KMeans
from tdamapper.clustering import TrivialClustering, CoverClustering, PermissiveClustering, MapperGraphClustering
from tdamapper.cover import TrivialCover, BallCover, KNNCover, CubicalCover


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


class ClusteringEstimator(Estimator):

    def get_clustering(self):
        return TrivialClustering()

    def fit(self, X, y=None):
        X, y = check_X_y(X, y)
        clustering = self.get_clustering()
        self.labels_ = clustering.fit(X, y).labels_
        self.n_features_in_ = X.shape[1]
        return self


class CoverClusteringEstimator(ClusteringEstimator):

    def get_cover(self):
        return TrivialCover()

    def get_clustering(self):
        return CoverClustering(cover=self.get_cover())


class PermissiveKMeans(ClusteringEstimator):

    def __init__(self, n=8):
        self.n = n

    def get_clustering(self):
        return PermissiveClustering(KMeans(n_clusters=self.n, n_init='auto'), verbose=False)


class TrivialClusteringEstimator(ClusteringEstimator):

    def get_clustering(self):
        return TrivialClustering()


class BallCoverEstimator(CoverClusteringEstimator):

    def __init__(self, radius=1.0, metric=euclidean):
        self.radius = radius
        self.metric = metric

    def get_cover(self):
        return BallCover(radius=self.radius, metric=self.metric)


class CubicalCoverEstimator(CoverClusteringEstimator):

    def __init__(self, n_intervals=10, overlap_frac=0.25):
        self.n_intervals = n_intervals
        self.overlap_frac = overlap_frac

    def get_cover(self):
        return CubicalCover(n_intervals=self.n_intervals, overlap_frac=self.overlap_frac)


class KNNCoverEstimator(CoverClusteringEstimator):

    def __init__(self, neighbors=5, metric=euclidean):
        self.neighbors = neighbors
        self.metric = metric

    def get_cover(self):
        return KNNCover(neighbors=self.neighbors, metric=self.metric)


class MapperSklearnEstimator(MapperGraphClustering, Estimator):

    def fit(self, X, y=None):
        X, y = check_X_y(X, y)
        self.labels_ = super().fit(X, y).labels_
        self.n_features_in_ = X.shape[1]
        return self


class TestSklearn(unittest.TestCase):

    def testClustering(self):
        check_estimator(TrivialClusteringEstimator())

    def testBall(self):
        check_estimator(BallCoverEstimator())

    def testCubical(self):
        check_estimator(CubicalCoverEstimator())

    def testKNN(self):
        check_estimator(KNNCoverEstimator())

    def testPermissive(self):
        check_estimator(PermissiveKMeans())

    def testMapper(self):
        mapper_est = MapperSklearnEstimator()
        check_estimator(mapper_est)
