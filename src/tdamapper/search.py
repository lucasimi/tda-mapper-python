"""
This module provides search algorithms for the tdamapper package.

It includes classes for ball search, KNN search, and cubical search.
These classes are used to efficiently find neighbors in a dataset based on
various distance metrics and search strategies.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from tdamapper._common import ParamsMixin, warn_user
from tdamapper.core import ArrayLike, PointLike
from tdamapper.metrics import chebyshev, get_metric
from tdamapper.vptree import VPTree


class _Pullback:

    def __init__(self, fun, dist):
        self.fun = fun
        self.dist = dist

    def __call__(self, x, y):
        return self.dist(self.fun(x), self.fun(y))


def _snd(x):
    return x[1]


class BallSearch(ParamsMixin):
    """
    Search points within a given radius from a query point.

    This class uses a vantage point tree (VPTree) for efficient searching of
    neighbors in a dataset. It implements a search algorithm that returns all
    points within a specified radius from a query point. The search can be
    customized with various parameters such as the distance metric, kind of
    vantage point tree, leaf capacity, and pivoting method. This class implements
    the :class:`tdamapper.core.SpatialSearch` protocol.

    :param radius: The radius within which to search for neighbors.
        Must be a positive value. Defaults to 1.0.
    :param metric: The distance metric to use for searching. Can be a string
        (e.g., 'euclidean') or a callable function. Defaults to 'euclidean'.
    :param metric_params: Additional parameters for the distance metric.
        This should be a dictionary containing parameters specific to the
        chosen metric. Defaults to None.
    :param kind: Specifies whether to use a flat or a hierarchical vantage
        point tree. Acceptable values are 'flat' or 'hierarchical'. Defaults to
        'flat'.
    :param leaf_capacity: The maximum number of points in a leaf node of the
        vantage point tree. Must be a positive value. Defaults to 1.
    :param leaf_radius: The radius of the leaf nodes. If not specified, it
        defaults to the value of `radius`. Must be a positive value. Defaults to None.
    :param pivoting: The method used for pivoting in the vantage point tree.
        Acceptable values are None, 'random', or 'furthest'. Defaults to None.
    """

    _radius: float
    _data: List[tuple[int, Any]]
    _vptree: VPTree

    def __init__(
        self,
        radius: float = 1.0,
        metric: Union[str, Callable] = "euclidean",
        metric_params: Optional[Dict[str, Any]] = None,
        kind: str = "flat",
        leaf_capacity: int = 1,
        leaf_radius: Optional[float] = None,
        pivoting: Optional[str] = None,
    ):
        self.radius = radius
        self.metric = metric
        self.metric_params = metric_params
        self.kind = kind
        self.leaf_capacity = leaf_capacity
        self.leaf_radius = leaf_radius
        self.pivoting = pivoting

    def fit(self, X: ArrayLike) -> BallSearch:
        """
        Train internal parameters.

        This method creates a vptree on the dataset in order to perform fast
        range queries in the func:`tdamapper.cover.BallSearch.search`
        method.

        :param X: A dataset of n points.
        :return: The object itself.
        """
        metric = get_metric(self.metric, **(self.metric_params or {}))
        self._radius = self.radius
        self._data = list(enumerate(X))
        self._vptree = VPTree(
            self._data,
            metric=_Pullback(_snd, metric),
            metric_params=None,
            kind=self.kind,
            leaf_capacity=self.leaf_capacity,
            leaf_radius=self.leaf_radius or self.radius,
            pivoting=self.pivoting,
        )
        return self

    def search(self, x: PointLike) -> List[int]:
        """
        Return a list of neighbors for the query point.

        This method uses the internal vptree to perform fast range queries.

        :param x: A query point for which we want to find neighbors.
        :return: The indices of the neighbors contained in the dataset.
        """
        if self._vptree is None:
            return []
        neighs = self._vptree.ball_search(
            (-1, x),
            self._radius,
            inclusive=False,
        )
        return [x for (x, _) in neighs]


class KNNSearch(ParamsMixin):
    """
    Search for k-nearest neighbors in a dataset.

    This class uses a vantage point tree (VPTree) to efficiently find the
    k-nearest neighbors of a query point in a dataset. It implements a search
    algorithm that returns the indices of the k-nearest neighbors based on a
    specified distance metric. The search can be customized with various
    parameters such as the number of neighbors, distance metric, and kind of
    vantage point tree. This class implements the
    :class:`tdamapper.core.SpatialSearch` protocol.

    :param neighbors: The number of nearest neighbors to search for.
        Must be a positive integer. Defaults to 1.
    :param metric: The distance metric to use for searching. Can be a string
        (e.g., 'euclidean') or a callable function. Defaults to 'euclidean'.
    :param metric_params: Additional parameters for the distance metric.
        This should be a dictionary containing parameters specific to the
        chosen metric. Defaults to None.
    :param kind: Specifies whether to use a flat or a hierarchical vantage
        point tree. Acceptable values are 'flat' or 'hierarchical'. Defaults to
        'flat'.
    :param leaf_capacity: The maximum number of points in a leaf node of the
        vantage point tree. Must be a positive value. Defaults to None, which
        means it will be set to the value of `neighbors`.
    :param leaf_radius: The radius of the leaf nodes. If not specified, it
        defaults to 0.0. Must be a non-negative value. Defaults to 0.0.
    :param pivoting: The method used for pivoting in the vantage point tree.
        Acceptable values are None, 'random', or 'furthest'. Defaults to None.
    """

    _neighbors: int
    _data: List[tuple[int, Any]]
    _vptree: VPTree

    def __init__(
        self,
        neighbors: int = 1,
        metric: Union[str, Callable] = "euclidean",
        metric_params: Optional[Dict[str, Any]] = None,
        kind: str = "flat",
        leaf_capacity: Optional[int] = None,
        leaf_radius: float = 0.0,
        pivoting: Optional[str] = None,
    ):
        self.neighbors = neighbors
        self.metric = metric
        self.metric_params = metric_params
        self.kind = kind
        self.leaf_capacity = leaf_capacity
        self.leaf_radius = leaf_radius
        self.pivoting = pivoting

    def fit(self, X: ArrayLike) -> KNNSearch:
        """
        Train internal parameters.

        This method creates a vptree on the dataset in order to perform fast
        KNN queries in the func:`tdamapper.cover.KNNSearch.search`
        method.

        :param X: A dataset of n points.
        :return: The object itself.
        """
        metric = get_metric(self.metric, **(self.metric_params or {}))
        self._neighbors = self.neighbors
        self._data = list(enumerate(X))
        self._vptree = VPTree(
            self._data,
            metric=_Pullback(_snd, metric),
            metric_params=None,
            kind=self.kind,
            leaf_capacity=self.leaf_capacity or self.neighbors,
            leaf_radius=self.leaf_radius,
            pivoting=self.pivoting,
        )
        return self

    def search(self, x: PointLike) -> List[int]:
        """
        Return a list of neighbors for the query point.

        This method queries the internal vptree in order to perform fast KNN
        queries.

        :param x: A query point for which we want to find neighbors.
        :return: The indices of the neighbors contained in the dataset.
        """
        if self._vptree is None:
            return []
        neighs = self._vptree.knn_search((-1, x), self._neighbors)
        return [x for (x, _) in neighs]


class CubicalSearch(ParamsMixin):
    """
    Search points within a cubical grid.

    This class implements a search algorithm that returns the indices of the
    hypercube whose center is closest to the target point. The hypercubes are
    defined by a uniform grid of intervals in each dimension, with a specified
    overlap fraction. The search can be customized with various parameters such as
    the number of intervals, overlap fraction, kind of vantage point tree,
    leaf capacity, leaf radius, and pivoting method. This class implements the
    :class:`tdamapper.core.SpatialSearch` protocol.

    :param n_intervals: The number of intervals to use for each dimension.
        Must be positive and less than or equal to the length of the dataset.
        Defaults to 1.
    :param overlap_frac: The fraction of overlap between adjacent intervals on
        each dimension, must be in the range (0.0, 0.5]. If not specified, the
        overlap_frac is computed such that the volume of the overlap within
        each hypercube is half the total volume. Defaults to None.
    :param kind: Specifies whether to use a flat or a hierarchical vantage
        point tree. Acceptable values are 'flat' or 'hierarchical'. Defaults to
        'flat'.
    :param leaf_capacity: The maximum number of points in a leaf node of the
        vantage point tree. Must be a positive value. Defaults to 1.
    :param leaf_radius: The radius of the leaf nodes. If not specified, it
        defaults to the value of `radius`. Must be a positive value. Defaults to None.
    :param pivoting: The method used for pivoting in the vantage point tree.
        Acceptable values are None, 'random', or 'furthest'. Defaults to None.
    """

    _n_intervals: int
    _overlap_frac: float
    _min: np.ndarray
    _max: np.ndarray
    _delta: np.ndarray
    _ball_search: BallSearch

    def __init__(
        self,
        n_intervals: int = 1,
        overlap_frac: Optional[float] = None,
        kind: str = "flat",
        leaf_capacity: int = 1,
        leaf_radius: Optional[float] = None,
        pivoting: Optional[str] = None,
    ):
        self.n_intervals = n_intervals
        self.overlap_frac = overlap_frac
        self.kind = kind
        self.leaf_capacity = leaf_capacity
        self.leaf_radius = leaf_radius
        self.pivoting = pivoting

    def _get_center(self, x):
        offset = self._offset(x)
        center = self._phi(x)
        return tuple(offset), center

    def _get_overlap_frac(self, dim, overlap_vol_frac):
        beta = math.pow(1.0 - overlap_vol_frac, 1.0 / dim)
        return 1.0 - 1.0 / (2.0 - beta)

    def _offset(self, x):
        return np.minimum(self._n_intervals - 1, np.floor(self._gamma_n(x)))

    def _phi(self, x):
        offset = self._offset(x)
        return self._gamma_n_inv(0.5 + offset)

    def _gamma_n(self, x):
        return self._n_intervals * (x - self._min) / self._delta

    def _gamma_n_inv(self, x):
        return self._min + self._delta * x / self._n_intervals

    def _get_bounds(self, X):
        if (X is None) or len(X) == 0:
            return
        _min, _max = X[0], X[0]
        eps = np.finfo(np.float64).eps
        _min = np.min(X, axis=0)
        _max = np.max(X, axis=0)
        _delta = _max - _min
        _delta[(_delta >= -eps) & (_delta <= eps)] = self._n_intervals
        return _min, _max, _delta

    def fit(self, X: ArrayLike) -> CubicalSearch:
        """
        Train internal parameters.

        This method builds an internal :class:`tdamapper.search.BallSearch`
        instance that allows efficient queries of the dataset.

        :param X: A dataset of n points.
        :return: The object itself.
        """
        X = np.asarray(X).reshape(len(X), -1).astype(float)
        if self.overlap_frac is None:
            dim = 1 if X.ndim == 1 else X.shape[1]
            self._overlap_frac = self._get_overlap_frac(dim, 0.5)
        else:
            self._overlap_frac = self.overlap_frac
        self._n_intervals = self.n_intervals
        if self._overlap_frac <= 0.0:
            raise ValueError("The parameter overlap_frac is expected to be " "> 0.0")
        if self._overlap_frac > 0.5:
            warn_user("The parameter overlap_frac is expected to be <= 0.5")
        self._min, self._max, self._delta = self._get_bounds(X)
        radius = 1.0 / (2.0 - 2.0 * self._overlap_frac)
        self._ball_search = BallSearch(
            radius,
            metric=_Pullback(self._gamma_n, chebyshev()),
            kind=self.kind,
            leaf_capacity=self.leaf_capacity,
            leaf_radius=self.leaf_radius,
            pivoting=self.pivoting,
        )
        self._ball_search.fit(X)
        return self

    def search(self, x: PointLike) -> List[int]:
        """
        Return a list of neighbors for the query point.

        This method takes a target point as input and returns the hypercube
        whose center is closest to the target point.

        :param x: A query point for which we want to find neighbors.
        :return: The indices of the neighbors contained in the dataset.
        """
        center = self._phi(x)
        return self._ball_search.search(center)


class CubicalLandmarks(CubicalSearch):
    """
    Search points within a cubical grid and identify landmarks.

    This class extends the :class:`tdamapper.search.CubicalSearch` adding a
    `landmarks` method that identifies unique hypercubes based on the centers of
    the hypercubes that intersect the dataset. This class implements the
    :class:`tdamapper.core.SpatialSearch` protocol.

    :param n_intervals: The number of intervals to use for each dimension.
        Must be positive and less than or equal to the length of the dataset.
        Defaults to 1.
    :param overlap_frac: The fraction of overlap between adjacent intervals on
        each dimension, must be in the range (0.0, 0.5]. If not specified, the
        overlap_frac is computed such that the volume of the overlap within
        each hypercube is half the total volume. Defaults to None.
    :param kind: Specifies whether to use a flat or a hierarchical vantage
        point tree. Acceptable values are 'flat' or 'hierarchical'. Defaults to
        'flat'.
    :param leaf_capacity: The maximum number of points in a leaf node of the
        vantage point tree. Must be a positive value. Defaults to 1.
    :param leaf_radius: The radius of the leaf nodes. If not specified, it
        defaults to the value of `radius`. Must be a positive value. Defaults
        to None.
    :param pivoting: The method used for pivoting in the vantage point tree.
        Acceptable values are None, 'random', or 'furthest'. Defaults to None.
    """

    def __init__(
        self,
        n_intervals: int = 1,
        overlap_frac: Optional[float] = None,
        kind: str = "flat",
        leaf_capacity: int = 1,
        leaf_radius: Optional[float] = None,
        pivoting: Optional[str] = None,
    ):
        super().__init__(
            n_intervals=n_intervals,
            overlap_frac=overlap_frac,
            kind=kind,
            leaf_capacity=leaf_capacity,
            leaf_radius=leaf_radius,
            pivoting=pivoting,
        )

    def landmarks(self, X: ArrayLike) -> Dict:
        """
        Identify unique hypercubes based on the centers of the hypercubes that
        intersect the dataset.
        This method returns a dictionary where the keys are the centers of the
        hypercubes and the values are the first point found in that hypercube.

        :param X: A dataset of n points.
        :return: A dictionary with hypercube centers as keys and the first point
            found in that hypercube as values.
        """
        lmrks = {}
        for x in X:
            lmrk, _ = self._get_center(x)
            if lmrk not in lmrks:
                lmrks[lmrk] = x
        return lmrks
