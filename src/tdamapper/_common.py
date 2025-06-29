"""
This module provides common functionalities for internal use.
"""

from __future__ import annotations

import cProfile
import io
import pstats
import warnings
from typing import Any, Callable, Dict, List, Union

import numpy as np
from numpy.typing import NDArray

warnings.filterwarnings("default", category=DeprecationWarning, module=r"^tdamapper\.")


PointLike = Union[Any, NDArray[np.float64]]

ArrayLike = Union[List[Any], NDArray[np.float64]]


def deprecated(msg: str) -> Callable:
    """
    Decorator to mark a function as deprecated.

    :param msg: A message to be shown when the function is called.
    :return: A decorator that wraps the function and issues a warning when called.
    """

    def deprecated_func(func):
        def wrapper(*args, **kwargs):
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper

    return deprecated_func


def warn_user(msg: str) -> None:
    """
    Issues a warning to the user.

    :param msg: A message to be shown to the user.
    """
    warnings.warn(msg, UserWarning, stacklevel=2)


class EstimatorMixin:
    """
    Mixin to add common functionalities to estimators, such as validation of
    input data, setting the number of features, and checking for sparse data.
    This mixin is intended to be used with scikit-learn compatible estimators.
    """

    def _is_sparse(self, x_arr: ArrayLike) -> bool:
        """
        Checks if the input array `x_arr` is sparse.

        :param x_arr: The input array to check.
        :return: True if `x_arr` is sparse, False otherwise.
        """
        # simple alternative use scipy.sparse.issparse
        return hasattr(x_arr, "toarray")

    def _validate_x_y(
        self,
        x_arr: ArrayLike,
        y_arr: ArrayLike,
    ) -> tuple[NDArray, NDArray]:
        """
        Validates the input arrays `x_arr` and `y_arr`.

        :param x_arr: The input features array.
        :param y_arr: The target values array.
        :return: A tuple of validated numpy arrays (x_arr, y_arr).
        :raises ValueError: If the input arrays are not valid, e.g., if they
            are sparse, empty, 1-dimensional, contain complex numbers, or have
            NaNs or infinite values.
        """
        if self._is_sparse(x_arr):
            raise ValueError("Sparse data not supported.")

        x_arr_ = np.asarray(x_arr)
        y_arr_ = np.asarray(y_arr)

        if x_arr_.size == 0:
            msg = (
                f"0 feature(s) (shape={x_arr_.shape}) while a minimum of 1 is "
                "required."
            )
            raise ValueError(msg)

        if y_arr_.size == 0:
            msg = (
                f"0 feature(s) (shape={y_arr_.shape}) while a minimum of 1 is "
                "required."
            )
            raise ValueError(msg)

        if x_arr_.ndim == 1:
            raise ValueError("1d-arrays not supported.")

        if np.iscomplexobj(x_arr_) or np.iscomplexobj(y_arr_):
            raise ValueError("Complex data not supported.")

        if x_arr_.dtype == np.object_:
            x_arr_ = np.array(x_arr_, dtype=float)

        if y_arr_.dtype == np.object_:
            y_arr_ = np.array(y_arr_, dtype=float)

        if (
            np.isnan(x_arr_).any()
            or np.isinf(x_arr_).any()
            or np.isnan(y_arr_).any()
            or np.isinf(y_arr_).any()
        ):
            raise ValueError("NaNs or infinite values not supported.")

        return x_arr_, y_arr_

    def _set_n_features_in(self, arr: ArrayLike) -> None:
        if hasattr(arr, "shape"):
            self.n_features_in_ = arr.shape[1]


class ParamsMixin:
    """
    Mixin to add setters and getters for public parameters, compatible with
    scikit-learn `get_params` and `set_params`.
    """

    def _is_param_public(self, k: str) -> bool:
        return (not k.startswith("_")) and (not k.endswith("_"))

    def _split_param(self, k: str) -> tuple[str, str]:
        k_split = k.split("__")
        outer = k_split[0]
        inner = "__".join(k_split[1:])
        return outer, inner

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Get all public parameters of the object as a dictionary.

        :param deep: A flag for returning also nested parameters.
        """
        params = {}
        for k, v in self.__dict__.items():
            if self._is_param_public(k):
                params[k] = v
                if hasattr(v, "get_params") and deep:
                    for _k, _v in v.get_params().items():
                        params[f"{k}__{_k}"] = _v
        return params

    def set_params(self, **params: Dict[str, Any]) -> ParamsMixin:
        """
        Set public parameters. Only updates attributes that already exist.
        """
        nested_params = []
        for k, v in params.items():
            if self._is_param_public(k):
                k_outer, k_inner = self._split_param(k)
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

    def __repr__(self) -> str:
        obj_noargs = type(self)()
        args_repr = []
        for k, v in self.__dict__.items():
            v_default = getattr(obj_noargs, k)
            v_default_repr = repr(v_default)
            v_repr = repr(v)
            if self._is_param_public(k) and not v_repr == v_default_repr:
                args_repr.append(f"{k}={v_repr}")
        return f"{self.__class__.__name__}({', '.join(args_repr)})"


def clone(obj: Any) -> Any:
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


def profile(n_lines: int = 10) -> Callable:
    """
    Decorator to profile a function using cProfile and print the top `n_lines`
    lines of the profiling report.

    :param n_lines: The number of lines to print from the profiling report.
    :return: A decorator that wraps the function and profiles its execution.
    """

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
