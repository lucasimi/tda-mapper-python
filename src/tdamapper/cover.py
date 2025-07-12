"""
Open cover construction for the Mapper algorithm.

An open cover is a collection of open subsets of a dataset whose union spans
the whole dataset. Unlike clustering, open subsets do not need to be disjoint.
Indeed, the overlaps of the open subsets define the edges of the Mapper graph.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Iterator, Literal, Optional, Union

import numpy as np
from numpy.typing import NDArray

from tdamapper._common import Array, ParamsMixin, warn_user
from tdamapper.core import Cover, proximity_net
from tdamapper.utils.metrics import Metric, MetricLiteral, chebyshev, get_metric
from tdamapper.utils.vptree import PivotingStrategy, VPTree, VPTreeKind


class _Pullback:

    def __init__(self, fun: Callable[..., Any], dist: Metric):
        self.fun = fun
        self.dist = dist

    def __call__(self, x: Any, y: Any) -> float:
        return self.dist(self.fun(x), self.fun(y))


def _snd(x: tuple[Any, ...]) -> Any:
    return x[1]


class BallCover(ParamsMixin):
    """
    Cover algorithm based on `ball proximity function`, which covers data with
    open balls of fixed radius.

    An open ball is a set of points within a specified distance from a center
    point. This class maps each point to its corresponding open ball with a
    fixed radius centered on the point itself.

    :param radius: The radius of the open balls. Must be a positive value.
        Defaults to 1.0.
    :param metric: The metric used to define the distance between points.
        Accepts any value compatible with `tdamapper.utils.metrics.get_metric`.
        Defaults to 'euclidean'.
    :param metric_params: Additional parameters for the metric function, to be
        passed to `tdamapper.utils.metrics.get_metric`. Defaults to None.
    :param kind: Specifies whether to use a flat or a hierarchical vantage
        point tree. Acceptable values are 'flat' or 'hierarchical'. Defaults
        to 'flat'.
    :param leaf_capacity: The maximum number of points in a leaf node of the
        vantage point tree. Must be a positive value. Defaults to 1.
    :param leaf_radius: The radius of the leaf nodes. If not specified, it
        defaults to the value of `radius`. Must be a positive value. Defaults
        to None.
    :param pivoting: The method used for pivoting in the vantage point tree.
        Acceptable values are None, 'random', or 'furthest'. Defaults to None.
    """

    _radius: float
    _data: list[tuple[int, Any]]
    _vptree: VPTree

    def __init__(
        self,
        radius: float = 1.0,
        metric: Union[MetricLiteral, Metric] = "euclidean",
        metric_params: Optional[dict[str, Any]] = None,
        kind: VPTreeKind = "flat",
        leaf_capacity: int = 1,
        leaf_radius: Optional[float] = None,
        pivoting: PivotingStrategy = "disabled",
    ) -> None:
        self.radius = radius
        self.metric = metric
        self.metric_params = metric_params
        self.kind = kind
        self.leaf_capacity = leaf_capacity
        self.leaf_radius = leaf_radius
        self.pivoting = pivoting

    def fit(self, X: Array[Any]) -> BallCover:
        """
        Train internal parameters.

        This method creates a vptree on the dataset in order to perform fast
        range queries in the func:`tdamapper.cover.BallCover.search`
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

    def search(self, x: Any) -> list[int]:
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

    def apply(self, X: Array[Any]) -> Iterator[list[int]]:
        """
        Covers the dataset using proximity-net.

        This function returns a generator that yields each element of the
        open cover as a list of ids. The ids are the indices of the points
        in the original dataset.

        :param X: A dataset of n points.
        :return: A generator of lists of ids.
        """
        return proximity_net(self, X)


class KNNCover(ParamsMixin):
    """
    Cover algorithm based on `KNN proximity function`, which covers data using
    k-nearest neighbors (KNN).

    This class maps each point to the set of the k nearest neighbors to the
    point itself.

    :param neighbors: The number of neighbors to use for the KNN Proximity
        function, must be positive and less than the length of the dataset.
        Defaults to 1.
    :param metric: The metric used to define the distance between points.
        Accepts any value compatible with `tdamapper.utils.metrics.get_metric`.
        Defaults to 'euclidean'.
    :param metric_params: Additional parameters for the metric function, to be
        passed to `tdamapper.utils.metrics.get_metric`. Defaults to None.
    :param kind: Specifies whether to use a flat or a hierarchical vantage
        point tree. Acceptable values are 'flat' or 'hierarchical'. Defaults
        to 'flat'.
    :param leaf_capacity: The maximum number of points in a leaf node of the
        vantage point tree. If not specified, it defaults to the value of
        `neighbors`. Must be a positive value. Defaults to None.
    :param leaf_radius: The radius of the leaf nodes. Must be a positive value.
        Defaults to 0.0.
    :param pivoting: The method used for pivoting in the vantage point tree.
        Acceptable values are None, 'random', or 'furthest'. Defaults to None.
    """

    _neighbors: int
    _data: list[tuple[int, Any]]
    _vptree: VPTree

    def __init__(
        self,
        neighbors: int = 1,
        metric: Union[MetricLiteral, Metric] = "euclidean",
        metric_params: Optional[dict[str, Any]] = None,
        kind: VPTreeKind = "flat",
        leaf_capacity: Optional[int] = None,
        leaf_radius: float = 0.0,
        pivoting: PivotingStrategy = "disabled",
    ) -> None:
        self.neighbors = neighbors
        self.metric = metric
        self.metric_params = metric_params
        self.kind = kind
        self.leaf_capacity = leaf_capacity
        self.leaf_radius = leaf_radius
        self.pivoting = pivoting

    def fit(self, X: Array[Any]) -> KNNCover:
        """
        Train internal parameters.

        This method creates a vptree on the dataset in order to perform fast
        KNN queries in the func:`tdamapper.cover.BallCover.search`
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

    def search(self, x: Any) -> list[int]:
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

    def apply(self, X: Array[Any]) -> Iterator[list[int]]:
        """
        Covers the dataset using proximity-net.

        This function returns a generator that yields each element of the
        open cover as a list of ids. The ids are the indices of the points
        in the original dataset.

        :param X: A dataset of n points.
        :return: A generator of lists of ids.
        """
        return proximity_net(self, X)


class BaseCubicalCover:

    _overlap_frac: float
    _n_intervals: int
    _min: NDArray[np.float_]
    _max: NDArray[np.float_]
    _delta: NDArray[np.float_]
    _cover: BallCover

    def __init__(
        self,
        n_intervals: int = 1,
        overlap_frac: Optional[float] = None,
        kind: VPTreeKind = "flat",
        leaf_capacity: int = 1,
        leaf_radius: Optional[float] = None,
        pivoting: PivotingStrategy = "disabled",
    ) -> None:
        self.n_intervals = n_intervals
        self.overlap_frac = overlap_frac
        self.kind = kind
        self.leaf_capacity = leaf_capacity
        self.leaf_radius = leaf_radius
        self.pivoting = pivoting

    def _get_center(
        self, x: NDArray[np.float_]
    ) -> tuple[tuple[float], NDArray[np.float_]]:
        offset = self._offset(x)
        center = self._phi(x)
        return tuple(offset), center

    def _get_overlap_frac(self, dim: int, overlap_vol_frac: float) -> float:
        beta = math.pow(1.0 - overlap_vol_frac, 1.0 / dim)
        return 1.0 - 1.0 / (2.0 - beta)

    def _offset(self, x: NDArray[np.float_]) -> NDArray[np.float_]:
        return np.minimum(self._n_intervals - 1, np.floor(self._gamma_n(x)))

    def _phi(self, x: NDArray[np.float_]) -> NDArray[np.float_]:
        offset = self._offset(x)
        return self._gamma_n_inv(0.5 + offset)

    def _gamma_n(self, x: NDArray[np.float_]) -> NDArray[np.float_]:
        return self._n_intervals * (x - self._min) / self._delta

    def _gamma_n_inv(self, x: NDArray[np.float_]) -> NDArray[np.float_]:
        return self._min + self._delta * x / self._n_intervals

    def _get_bounds(
        self, X: NDArray[np.float_]
    ) -> tuple[NDArray[np.float_], NDArray[np.float_], NDArray[np.float_]]:
        if (X is None) or len(X) == 0:
            raise ValueError("The dataset is empty.")
        _min, _max = X[0], X[0]
        eps = np.finfo(np.float64).eps
        _min = np.min(X, axis=0)
        _max = np.max(X, axis=0)
        _delta = _max - _min
        _delta[(_delta >= -eps) & (_delta <= eps)] = self._n_intervals
        return _min, _max, _delta

    def fit(self, X: Array[Any]) -> BaseCubicalCover:
        """
        Train internal parameters.

        This method builds an internal :class:`tdamapper.cover.BallCover`
        attribute that allows efficient queries of the dataset.

        :param X: A dataset of n points.
        :return: The object itself.
        """
        X_ = np.asarray(X).reshape(len(X), -1).astype(float)
        if self.overlap_frac is None:
            dim = 1 if X_.ndim == 1 else X_.shape[1]
            self._overlap_frac = self._get_overlap_frac(dim, 0.5)
        else:
            self._overlap_frac = self.overlap_frac
        self._n_intervals = self.n_intervals
        if self._overlap_frac <= 0.0:
            raise ValueError("The parameter overlap_frac is expected to be > 0.0")
        if self._overlap_frac > 0.5:
            warn_user("The parameter overlap_frac is expected to be <= 0.5")
        self._min, self._max, self._delta = self._get_bounds(X_)
        radius = 1.0 / (2.0 - 2.0 * self._overlap_frac)
        self._cover = BallCover(
            radius,
            metric=_Pullback(self._gamma_n, chebyshev()),
            kind=self.kind,
            leaf_capacity=self.leaf_capacity,
            leaf_radius=self.leaf_radius,
            pivoting=self.pivoting,
        )
        self._cover.fit(X_)
        return self

    def search(self, x: Any) -> list[int]:
        """
        Return a list of neighbors for the query point.

        This method takes a target point as input and returns the hypercube
        whose center is closest to the target point.

        :param x: A query point for which we want to find neighbors.
        :return: The indices of the neighbors contained in the dataset.
        """
        center = self._phi(x)
        return self._cover.search(center)


class ProximityCubicalCover(BaseCubicalCover, ParamsMixin):
    """
    Cover algorithm based on the `cubical proximity function`, which covers
    data with open hypercubes of uniform size and overlap. The cubical cover is
    obtained by selecting a subsect of all the hypercubes that intersect the
    dataset using proximity net (see :class:`tdamapper.core.Proximity`).
    For an open cover containing all the hypercubes interecting the dataset
    use :class:`tdamapper.core.StandardCubicalCover`.

    A hypercube is a multidimensional generalization of a square or a cube.
    The size and overlap of the hypercubes are determined by the number of
    intervals and the overlap fraction parameters. This class maps each point
    to the hypercube with the nearest center.

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
        kind: VPTreeKind = "flat",
        leaf_capacity: int = 1,
        leaf_radius: Optional[float] = None,
        pivoting: PivotingStrategy = "disabled",
    ) -> None:
        super().__init__(
            n_intervals=n_intervals,
            overlap_frac=overlap_frac,
            kind=kind,
            leaf_capacity=leaf_capacity,
            leaf_radius=leaf_radius,
            pivoting=pivoting,
        )

    def apply(self, X: Array[Any]) -> Iterator[list[int]]:
        """
        Covers the dataset using proximity-net.

        This function returns a generator that yields each element of the
        open cover as a list of ids. The ids are the indices of the points
        in the original dataset.

        :param X: A dataset of n points.
        :return: A generator of lists of ids.
        """
        return proximity_net(self, X)


class StandardCubicalCover(BaseCubicalCover, ParamsMixin):
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
        kind: VPTreeKind = "flat",
        leaf_capacity: int = 1,
        leaf_radius: Optional[float] = None,
        pivoting: PivotingStrategy = "disabled",
    ) -> None:
        super().__init__(
            n_intervals=n_intervals,
            overlap_frac=overlap_frac,
            kind=kind,
            leaf_capacity=leaf_capacity,
            leaf_radius=leaf_radius,
            pivoting=pivoting,
        )

    def _landmarks(self, X: Array[Any]) -> dict[tuple[float], Array[Any]]:
        lmrks = {}
        for x in X:
            lmrk, _ = self._get_center(x)
            if lmrk not in lmrks:
                lmrks[lmrk] = x
        return lmrks

    def apply(self, X: Array[Any]) -> Iterator[list[int]]:
        """
        Covers the dataset using landmarks.

        This function yields all the hypercubes intersecting the dataset.

        This function returns a generator that yields each element of the
        open cover as a list of ids. The ids are the indices of the points
        in the original dataset.

        :param X: A dataset of n points.
        :return: A generator of lists of ids.
        """
        self.fit(X)
        lmrks_to_cover = self._landmarks(X)
        while lmrks_to_cover:
            _, x = lmrks_to_cover.popitem()
            neigh_ids = self.search(x)
            if neigh_ids:
                yield neigh_ids


CubicalAlgorithm = Literal["standard", "proximity"]


class CubicalCover(ParamsMixin):
    """
    Wrapper class for cubical cover algorithms, which cover data with open
    hypercubes of uniform size and overlap. This class delegates its methods to
    either :class:`tdamapper.cover.StandardCubicalCover` or
    :class:`tdamapper.cover.ProximityCubicalCover`, based on the `algorithm`
    parameter.

    A hypercube is a multidimensional generalization of a square or a cube.
    The size and overlap of the hypercubes are determined by the number of
    intervals and the overlap fraction parameters.

    :param n_intervals: The number of intervals to use for each dimension.
        Must be positive and less than or equal to the length of the dataset.
        Defaults to 1.
    :param overlap_frac: The fraction of overlap between adjacent intervals on
        each dimension, must be in the range (0.0, 0.5]. If not specified, the
        overlap_frac is computed such that the volume of the overlap within
        each hypercube is half the total volume. Defaults to None.
    :param algorithm: Specifies whether to use standard cubical cover, as in
        :class:`tdamapper.cover.StandardCubicalCover` or proximity cubical
        cover, as in :class:`tdamapper.cover.ProximityCubicalCover`.
        Acceptable values are 'standard' or 'proximity'. Defaults to
        'proximity'.
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

    _cubical_cover: Union[ProximityCubicalCover, StandardCubicalCover]

    def __init__(
        self,
        n_intervals: int = 1,
        overlap_frac: Optional[float] = None,
        algorithm: CubicalAlgorithm = "proximity",
        kind: VPTreeKind = "flat",
        leaf_capacity: int = 1,
        leaf_radius: Optional[float] = None,
        pivoting: PivotingStrategy = "disabled",
    ):
        self.n_intervals = n_intervals
        self.overlap_frac = overlap_frac
        self.algorithm = algorithm
        self.kind = kind
        self.leaf_capacity = leaf_capacity
        self.leaf_radius = leaf_radius
        self.pivoting = pivoting

    def _get_cubical_cover(self) -> Union[ProximityCubicalCover, StandardCubicalCover]:
        params: dict[str, Any] = dict(
            n_intervals=self.n_intervals,
            overlap_frac=self.overlap_frac,
            kind=self.kind,
            leaf_capacity=self.leaf_capacity,
            leaf_radius=self.leaf_radius,
            pivoting=self.pivoting,
        )
        if self.algorithm == "proximity":
            return ProximityCubicalCover(**params)
        if self.algorithm == "standard":
            return StandardCubicalCover(**params)
        raise ValueError(
            "The only possible values for algorithm are 'standard' and 'proximity'."
        )

    def fit(self, X: Array[Any]) -> CubicalCover:
        """
        Train internal parameters.

        This method delegates to the :func:`fit` method of the internal cubical
        cover used.

        :param X: A dataset of n points.
        :return: The object itself.
        """
        self._cubical_cover = self._get_cubical_cover()
        self._cubical_cover.fit(X)
        return self

    def search(self, x: Any) -> list[int]:
        """
        Return a list of neighbors for the query point.

        This method delegates to the `search` method of the internal cubical
        cover used.

        :param x: A query point for which we want to find neighbors.
        :return: The indices of the neighbors contained in the dataset.
        """
        return self._cubical_cover.search(x)

    def apply(self, X: Array[Any]) -> Iterator[list[int]]:
        """
        Covers the dataset using hypercubes.

        This method delegates to the `apply` method of the internal cubical
        cover used.

        :param X: A dataset of n points.
        :return: A generator of lists of ids.
        """
        self._cubical_cover = self._get_cubical_cover()
        return self._cubical_cover.apply(X)
