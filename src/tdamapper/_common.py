"""
This module provides common functionalities for internal use.
"""

import cProfile
import io
import pstats
import warnings

import numpy as np

warnings.filterwarnings("default", category=DeprecationWarning, module=r"^tdamapper\.")


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
        return hasattr(X, "toarray")

    def _validate_X_y(self, X, y):
        if self.__is_sparse(X):
            raise ValueError("Sparse data not supported.")

        X = np.asarray(X)
        y = np.asarray(y)

        if X.size == 0:
            msg = f"0 feature(s) (shape={X.shape}) while a minimum of 1 is " "required."
            raise ValueError(msg)

        if y.size == 0:
            msg = f"0 feature(s) (shape={y.shape}) while a minimum of 1 is " "required."
            raise ValueError(msg)

        if X.ndim == 1:
            raise ValueError("1d-arrays not supported.")

        if np.iscomplexobj(X) or np.iscomplexobj(y):
            raise ValueError("Complex data not supported.")

        if X.dtype == np.object_:
            X = np.array(X, dtype=float)

        if y.dtype == np.object_:
            y = np.array(y, dtype=float)

        if (
            np.isnan(X).any()
            or np.isinf(X).any()
            or np.isnan(y).any()
            or np.isinf(y).any()
        ):
            raise ValueError("NaNs or infinite values not supported.")

        return X, y

    def _set_n_features_in(self, X):
        self.n_features_in_ = X.shape[1]


class ParamsMixin:
    """
    Mixin to add setters and getters for public parameters, compatible with
    scikit-learn `get_params` and `set_params`.
    """

    def __is_param_public(self, k):
        return (not k.startswith("_")) and (not k.endswith("_"))

    def __split_param(self, k):
        k_split = k.split("__")
        outer = k_split[0]
        inner = "__".join(k_split[1:])
        return outer, inner

    def get_params(self, deep=True):
        """
        Get all public parameters of the object as a dictionary.

        :param deep: A flag for returning also nested parameters.
        :type deep: bool, optional.
        """
        params = {}
        for k, v in self.__dict__.items():
            if self.__is_param_public(k):
                params[k] = v
                if hasattr(v, "get_params") and deep:
                    for _k, _v in v.get_params().items():
                        params[f"{k}__{_k}"] = _v
        return params

    def set_params(self, **params):
        """
        Set public parameters. Only updates attributes that already exist.
        """
        nested_params = []
        for k, v in params.items():
            if self.__is_param_public(k):
                k_outer, k_inner = self.__split_param(k)
                if not k_inner:
                    if hasattr(self, k_outer):
                        setattr(self, k_outer, v)
                else:
                    nested_params.append((k_outer, k_inner, v))
        for k_outer, k_inner, v in nested_params:
            if hasattr(self, k_outer):
                k_attr = getattr(self, k_outer)
                k_attr.set_params(**{k_inner: v})
        return self

    def __repr__(self):
        obj_noargs = type(self)()
        args_repr = []
        for k, v in self.__dict__.items():
            v_default = getattr(obj_noargs, k)
            v_default_repr = repr(v_default)
            v_repr = repr(v)
            if self.__is_param_public(k) and not v_repr == v_default_repr:
                args_repr.append(f"{k}={v_repr}")
        return f"{self.__class__.__name__}({', '.join(args_repr)})"


def clone(obj):
    """
    Clone an estimator, returning a new one, unfitted, having the same public
    parameters.

    :param estimator: An estimator to be cloned.
    :type estimator: A scikit-learn compatible estimator
    :return: A new estimator with the same parameters.
    :rtype: A scikit-learn compatible estimator
    """
    params = obj.get_params(deep=True)
    obj_noargs = type(obj)()
    obj_noargs.set_params(**params)
    return obj_noargs


def profile(n_lines=10):
    def decorator(func):
        def wrapper(*args, **kwargs):
            profiler = cProfile.Profile()
            profiler.enable()
            result = func(*args, **kwargs)
            profiler.disable()

            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
            ps.print_stats(n_lines)
            print(s.getvalue())
            return result

        return wrapper

    return decorator
