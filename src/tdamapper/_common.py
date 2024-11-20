"""
This module provides common functionalities for internal use.
"""
import warnings

import numpy as np


warnings.filterwarnings(
    'default',
    category=DeprecationWarning,
    module=r'^tdamapper\.'
)


def deprecated(msg):
    def deprecated_func(func):
        def wrapper(*args, **kwargs):
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)
        return wrapper
    return deprecated_func


def warn_user(msg):
    warnings.warn(msg, UserWarning, stacklevel=2)


class EstimatorMixin:

    def __is_sparse(self, X):
        # simple alternative use scipy.sparse.issparse
        return hasattr(X, 'toarray')

    def _validate_X_y(self, X, y):
        if self.__is_sparse(X):
            raise ValueError('Sparse data not supported.')

        X = np.asarray(X)
        y = np.asarray(y)

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

    def _set_n_features_in(self, X):
        self.n_features_in_ = X.shape[1]


class ParamsMixin:
    """
    Mixin to add setters and getters for public parameters, compatible with
    scikit-learn `get_params` and `set_params`.
    """

    def __is_param_internal(self, k):
        return k.startswith('_') or k.endswith('_')

    def get_params(self, deep=True):
        """
        Get all public parameters of the object as a dictionary.

        :param deep: A flag for returning also nested parameters.
        :type deep: bool, optional.
        """
        params = self.__dict__.items()
        return {k: v for k, v in params if not self.__is_param_internal(k)}

    def set_params(self, **params):
        """
        Set public parameters. Only updates attributes that already exist.
        """
        for k, v in params.items():
            if hasattr(self, k) and not self.__is_param_internal(k):
                setattr(self, k, v)
        return self


def clone(estimator):
    """
    Clone an estimator, returning a new one, unfitted, having the same public
    parameters.

    :param estimator: An estimator to be cloned.
    :type estimator: A scikit-learn compatible estimator
    :return: A new estimator with the same parameters.
    :rtype: A scikit-learn compatible estimator
    """
    params = estimator.get_params(deep=True)
    return type(estimator)(**params)
