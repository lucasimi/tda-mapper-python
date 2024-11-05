import unittest
import logging

import numpy as np


from sklearn.utils.estimator_checks import check_estimator

from tdamapper.core import MapperAlgorithm
from tdamapper.cover import (
    BallCover,
    KNNCover,
    CubicalCover
)

from .setup_logging import setup_logging


def euclidean(x, y):
    return np.linalg.norm(x - y)


class ValidationMixin:

    def _is_sparse(self, X):
        # in alternative use scipy.sparse.issparse
        return hasattr(X, 'toarray')

    def _validate_X_y(self, X, y):
        if self._is_sparse(X):
            raise ValueError('Sparse data not supported.')

        if X.size == 0:
            msg = f'0 feature(s) (shape={X.shape}) while a minimum of 1 is required.'
            raise ValueError(msg)

        if y.size == 0:
            msg = f'0 feature(s) (shape={y.shape}) while a minimum of 1 is required.'
            raise ValueError(msg)

        if X.ndim == 1:
            raise ValueError('1d-arrays not supported.')

        if np.iscomplexobj(X) or np.iscomplexobj(y):
            raise ValueError('Complex data not supported.')

        if X.dtype == np.object_:
            X = np.array(X, dtype=float)

        if y.dtype == np.object_:
            y = np.array(y, dtype=float)

        if np.isnan(X).any() or np.isinf(X).any() or \
           np.isnan(y).any() or np.isinf(y).any():
            raise ValueError('NaNs or infinite values not supported.')

        return X, y

    def fit(self, X, y=None):
        X, y = self._validate_X_y(X, y)
        res = super().fit(X, y)
        self.n_features_in_ = X.shape[1]
        return res


class MapperEstimator(ValidationMixin, MapperAlgorithm):
    pass


class TestSklearn(unittest.TestCase):

    setup_logging()
    logger = logging.getLogger(__name__)

    def run_tests(self, estimator):
        for est, check in check_estimator(estimator, generate_only=True):
            # self.logger.info(f'{check}')
            check(est)

    def test_trivial(self):
        est = MapperEstimator()
        self.run_tests(est)

    def test_ball(self):
        est = MapperEstimator(cover=BallCover(metric=euclidean))
        self.run_tests(est)

    def test_knn(self):
        est = MapperEstimator(cover=KNNCover(metric=euclidean))
        self.run_tests(est)

    def test_cubical(self):
        est = MapperEstimator(cover=CubicalCover())
        self.run_tests(est)
