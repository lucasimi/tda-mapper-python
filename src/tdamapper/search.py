"""
Open cover construction for the Mapper algorithm.

An open cover is a collection of open subsets of a dataset whose union spans
the whole dataset. Unlike clustering, open subsets do not need to be disjoint.
Indeed, the overlaps of the open subsets define the edges of the Mapper graph.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Generator, List, Optional, Union

import numpy as np

from tdamapper._common import warn_user
from tdamapper.core import ArrayLike, PointLike
from tdamapper.utils.metrics import chebyshev, get_metric
from tdamapper.utils.vptree import VPTree


class _Pullback:

    def __init__(self, fun, dist):
        self.fun = fun
        self.dist = dist

    def __call__(self, x, y):
        return self.dist(self.fun(x), self.fun(y))


def _snd(x):
    return x[1]


class BallSearch:

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
        range queries in the func:`tdamapper.cover.BallCover.search`
        method.

        :param X: A dataset of n points.
        :type X: array-like of shape (n, m) or list-like of length n
        :return: The object itself.
        :rtype: self
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
        :type x: Any
        :return: The indices of the neighbors contained in the dataset.
        :rtype: list[int]
        """
        if self._vptree is None:
            return []
        neighs = self._vptree.ball_search(
            (-1, x),
            self._radius,
            inclusive=False,
        )
        return [x for (x, _) in neighs]


class KNNSearch:

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
        KNN queries in the func:`tdamapper.cover.BallCover.search`
        method.

        :param X: A dataset of n points.
        :type X: array-like of shape (n, m) or list-like of length n
        :return: The object itself.
        :rtype: self
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
        :type x: Any
        :return: The indices of the neighbors contained in the dataset.
        :rtype: list[int]
        """
        if self._vptree is None:
            return []
        neighs = self._vptree.knn_search((-1, x), self._neighbors)
        return [x for (x, _) in neighs]


class CubicalSearch:

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

        This method builds an internal :class:`tdamapper.cover.BallCover`
        attribute that allows efficient queries of the dataset.

        :param X: A dataset of n points.
        :type X: array-like of shape (n, m) or list-like of length n
        :return: The object itself.
        :rtype: self
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
        self._cover = BallSearch(
            radius,
            metric=_Pullback(self._gamma_n, chebyshev()),
            kind=self.kind,
            leaf_capacity=self.leaf_capacity,
            leaf_radius=self.leaf_radius,
            pivoting=self.pivoting,
        )
        self._cover.fit(X)
        return self

    def search(self, x: PointLike) -> List[int]:
        """
        Return a list of neighbors for the query point.

        This method takes a target point as input and returns the hypercube
        whose center is closest to the target point.

        :param x: A query point for which we want to find neighbors.
        :type x: Any
        :return: The indices of the neighbors contained in the dataset.
        :rtype: list[int]
        """
        center = self._phi(x)
        return self._cover.search(center)


class CubicalLandmarks(CubicalSearch):
    """
    Cover algorithm based on the standard open cover, which covers data with
    open hypercubes of uniform size and overlap. The standard cover is
    obtained by selecting all the hypercubes that intersect the dataset.

    A hypercube is a multidimensional generalization of a square or a cube.
    The size and overlap of the hypercubes are determined by the number of
    intervals and the overlap fraction parameters. This class maps each point
    to the hypercube with the nearest center.

    :param n_intervals: The number of intervals to use for each dimension.
        Must be positive and less than or equal to the length of the dataset.
        Defaults to 1.
    :type n_intervals: int
    :param overlap_frac: The fraction of overlap between adjacent intervals on
        each dimension, must be in the range (0.0, 0.5]. If not specified, the
        overlap_frac is computed such that the volume of the overlap within
        each hypercube is half the total volume. Defaults to None.
    :type overlap_frac: float
    :param kind: Specifies whether to use a flat or a hierarchical vantage
        point tree. Acceptable values are 'flat' or 'hierarchical'. Defaults to
        'flat'.
    :type kind: str
    :param leaf_capacity: The maximum number of points in a leaf node of the
        vantage point tree. Must be a positive value. Defaults to 1.
    :type leaf_capacity: int
    :param leaf_radius: The radius of the leaf nodes. If not specified, it
        defaults to the value of `radius`. Must be a positive value. Defaults
        to None.
    :type leaf_radius: float, optional
    :param pivoting: The method used for pivoting in the vantage point tree.
        Acceptable values are None, 'random', or 'furthest'. Defaults to None.
    :type pivoting: str or callable, optional
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
        lmrks = {}
        for x in X:
            lmrk, center = self._get_center(x)
            if lmrk not in lmrks:
                lmrks[lmrk] = x
        return lmrks

    def transform(self, X: ArrayLike) -> Generator[List[int], None, None]:
        """
        Covers the dataset using landmarks.

        This function yields all the hypercubes intersecting the dataset.

        This function returns a generator that yields each element of the
        open cover as a list of ids. The ids are the indices of the points
        in the original dataset.

        :param X: A dataset of n points.
        :type X: array-like of shape (n, m) or list-like of length n
        :return: A generator of lists of ids.
        :rtype: generator of lists of ints
        """
        self.fit(X)
        lmrks_to_cover = self.landmarks(X)
        while lmrks_to_cover:
            _, x = lmrks_to_cover.popitem()
            neigh_ids = self.search(x)
            if neigh_ids:
                yield neigh_ids
