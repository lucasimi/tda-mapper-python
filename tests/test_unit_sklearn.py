import unittest

import numpy as np
from sklearn.utils import check_X_y
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_array
from sklearn.cluster import KMeans

from tdamapper.clustering import (
    TrivialClustering,
    MapperClustering,
    FailSafeClustering
)
from tdamapper.core import TrivialCover, MapperAlgorithm
from tdamapper.cover import (
    BallCover,
    KNNCover,
    CubicalCover
)
from tdamapper._common import ParamsMixin


def euclidean(x, y):
    return np.linalg.norm(x - y)


class ValidationMixin:

    def fit(self, X, y=None):
        X, y = check_X_y(X, y)
        res = super().fit(X, y)
        self.n_features_in_ = X.shape[1]
        return res


class MapperEstimator(ValidationMixin, MapperAlgorithm):
    pass


class TestSklearn(unittest.TestCase):

    def test_trivial(self):
        est = MapperEstimator()
        check_estimator(est)

    def test_ball(self):
        est = MapperEstimator(cover=BallCover(metric=euclidean))
        check_estimator(est)

    def test_knn(self):
        est = MapperEstimator(cover=KNNCover(metric=euclidean))
        check_estimator(est)

    def test_cubical(self):
        est = MapperEstimator(cover=CubicalCover())
        check_estimator(est)
