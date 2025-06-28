"""
This module provides various cover algorithms for the Mapper algorithm,

including ball cover, KNN cover, proximity cubical cover, standard cubical
cover, and cubical cover. These algorithms are used to create open covers of
datasets, which are essential for constructing the Mapper graph. An open cover
is a collection of open subsets of a dataset whose union spans the whole dataset.
Unlike clustering, open subsets do not need to be disjoint. The overlaps of the
open subsets define the edges of the Mapper graph.
The module also includes a ProximityNet class that provides a common interface
for applying spatial search algorithms to datasets, yielding the indices of
covered points based on the search results.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Generator, List, Optional, Union

from tdamapper._common import ParamsMixin
from tdamapper.core import ArrayLike, SpatialSearch
from tdamapper.search import BallSearch, CubicalLandmarks, CubicalSearch, KNNSearch


class ProximityNet:
    """
    A class that provides a common interface for applying spatial search
    algorithms to datasets, yielding the indices of covered points based on
    the search results.

    :param search: An instance of a spatial search algorithm that implements
        the `fit` and `search` methods. This can be any class that adheres
        to the :class:`tdamapper.core.SpatialSearch` interface, such as
        `BallSearch` or `KNNSearch`.
    :type search: SpatialSearch
    """

    def __init__(self, search: SpatialSearch):
        self._search = search

    def fit(self, X: ArrayLike) -> ProximityNet:
        """
        Train internal parameters of the search algorithm.

        This method fits the search algorithm to the dataset `X`, allowing it
        to prepare for efficient searching.

        :param X: A dataset of n points.
        :type X: array-like of shape (n, m) or list-like of length n
        :return: The object itself.
        """
        self._search.fit(X)
        return self

    def apply(self, X: ArrayLike) -> Generator[List[int], None, None]:
        """
        Applies the search algorithm to the dataset `X` and yields the indices
        of covered points.

        This method iterates over each point in `X`, performs a search using
        the search algorithm, and yields the indices of the points that are
        covered by the search results. It ensures that each point is only
        covered once, even if multiple points are found in the same search.

        :param X: A dataset of n points.
        :type X: array-like of shape (n, m) or list-like of length n
        :return: A generator that yields lists of indices of covered points.
        :rtype: generator of lists of ints
        :yield: Lists of indices of points covered by the search results.
        :rtype: Lists of ints
        """
        covered_ids = set()
        for i, xi in enumerate(X):
            if i not in covered_ids:
                neigh_ids = self._search.search(xi)
                covered_ids.update(neigh_ids)
                if neigh_ids:
                    yield neigh_ids


class BallCover(ParamsMixin):
    """
    Cover algorithm based on `ball proximity function`, which covers data with
    open balls of fixed radius.

    An open ball is a set of points within a specified distance from a center
    point. This class maps each point to its corresponding open ball with a
    fixed radius centered on the point itself.

    :param radius: The radius of the open balls. Must be a positive value.
        Defaults to 1.0.
    :type radius: float
    :param metric: The metric used to define the distance between points.
        Accepts any value compatible with `tdamapper.metrics.get_metric`.
        Defaults to 'euclidean'.
    :type metric: str or callable
    :param metric_params: Additional parameters for the metric function, to be
        passed to `tdamapper.metrics.get_metric`. Defaults to None.
    :type metric_params: dict, optional
    :param kind: Specifies whether to use a flat or a hierarchical vantage
        point tree. Acceptable values are 'flat' or 'hierarchical'. Defaults
        to 'flat'.
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

    def fit(self, X: ArrayLike) -> BallCover:
        """
        Train internal parameters of the ball search algorithm.
        This method fits the ball search algorithm to the dataset `X`, allowing
        it to prepare for efficient searching.

        :param X: A dataset of n points.
        :type X: array-like of shape (n, m) or list-like of length n
        :return: The object itself.
        :rtype: BallCover
        """
        search = BallSearch(
            radius=self.radius,
            metric=self.metric,
            metric_params=self.metric_params,
            kind=self.kind,
            leaf_capacity=self.leaf_capacity,
            leaf_radius=self.leaf_radius,
            pivoting=self.pivoting,
        )
        self._proximity_net = ProximityNet(search)
        self._proximity_net.fit(X)
        return self

    def transform(self, X: ArrayLike) -> Generator[List[int], None, None]:
        """
        Applies the ball search algorithm to the dataset `X` and yields the
        indices of covered points.

        This method iterates over each point in `X`, performs a search using
        the ball search algorithm, and yields the indices of the points that are
        covered by the search results. It ensures that each point is only
        covered once, even if multiple points are found in the same search.

        :param X: A dataset of n points.
        :type X: array-like of shape (n, m) or list-like of length n
        :return: A generator that yields lists of indices of covered points.
        :rtype: generator of lists of ints
        :yield: Lists of indices of points covered by the search results.
        :rtype: Lists of ints
        """
        return self._proximity_net.apply(X)


class KNNCover(ParamsMixin):
    """
    Cover algorithm based on `KNN proximity function`, which covers data using
    k-nearest neighbors (KNN).

    This class maps each point to the set of the k nearest neighbors to the
    point itself.

    :param neighbors: The number of neighbors to use for the KNN Proximity
        function, must be positive and less than the length of the dataset.
        Defaults to 1.
    :type neighbors: int
    :param metric: The metric used to define the distance between points.
        Accepts any value compatible with `tdamapper.metrics.get_metric`.
        Defaults to 'euclidean'.
    :type metric: str or callable
    :param metric_params: Additional parameters for the metric function, to be
        passed to `tdamapper.metrics.get_metric`. Defaults to None.
    :type metric_params: dict, optional
    :param kind: Specifies whether to use a flat or a hierarchical vantage
        point tree. Acceptable values are 'flat' or 'hierarchical'. Defaults
        to 'flat'.
    :type kind: str
    :param leaf_capacity: The maximum number of points in a leaf node of the
        vantage point tree. If not specified, it defaults to the value of
        `neighbors`. Must be a positive value. Defaults to None.
    :type leaf_capacity: int, optional
    :param leaf_radius: The radius of the leaf nodes. Must be a positive value.
        Defaults to 0.0.
    :type leaf_radius: float
    :param pivoting: The method used for pivoting in the vantage point tree.
        Acceptable values are None, 'random', or 'furthest'. Defaults to None.
    :type pivoting: str or callable, optional
    """

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

    def fit(self, X: ArrayLike) -> KNNCover:
        search = KNNSearch(
            neighbors=self.neighbors,
            metric=self.metric,
            metric_params=self.metric_params,
            kind=self.kind,
            leaf_capacity=self.leaf_capacity,
            leaf_radius=self.leaf_radius,
            pivoting=self.pivoting,
        )
        self._proximity_net = ProximityNet(search)
        self._proximity_net.fit(X)
        return self

    def transform(self, X: ArrayLike) -> Generator[List[int], None, None]:
        return self._proximity_net.apply(X)


class ProximityCubicalCover(ParamsMixin):
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
        self.n_intervals = n_intervals
        self.overlap_frac = overlap_frac
        self.kind = kind
        self.leaf_capacity = leaf_capacity
        self.leaf_radius = leaf_radius
        self.pivoting = pivoting

    def fit(self, X: ArrayLike) -> ProximityCubicalCover:
        search = CubicalSearch(
            n_intervals=self.n_intervals,
            overlap_frac=self.overlap_frac,
            kind=self.kind,
            leaf_capacity=self.leaf_capacity,
            leaf_radius=self.leaf_radius,
            pivoting=self.pivoting,
        )
        self._proximity_net = ProximityNet(search)
        self._proximity_net.fit(X)
        return self

    def transform(self, X: ArrayLike) -> Generator[List[int], None, None]:
        return self._proximity_net.apply(X)


class StandardCubicalCover(ParamsMixin):
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
        self.n_intervals = n_intervals
        self.overlap_frac = overlap_frac
        self.kind = kind
        self.leaf_capacity = leaf_capacity
        self.leaf_radius = leaf_radius
        self.pivoting = pivoting

    def fit(self, X: ArrayLike) -> StandardCubicalCover:
        self._landmarks = CubicalLandmarks(
            n_intervals=self.n_intervals,
            overlap_frac=self.overlap_frac,
            kind=self.kind,
            leaf_capacity=self.leaf_capacity,
            leaf_radius=self.leaf_radius,
            pivoting=self.pivoting,
        )
        self._landmarks.fit(X)
        return self

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
        lmrks_to_cover = self._landmarks.landmarks(X)
        while lmrks_to_cover:
            _, x = lmrks_to_cover.popitem()
            neigh_ids = self._landmarks.search(x)
            if neigh_ids:
                yield neigh_ids


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
    :type n_intervals: int
    :param overlap_frac: The fraction of overlap between adjacent intervals on
        each dimension, must be in the range (0.0, 0.5]. If not specified, the
        overlap_frac is computed such that the volume of the overlap within
        each hypercube is half the total volume. Defaults to None.
    :type overlap_frac: float
    :param algorithm: Specifies whether to use standard cubical cover, as in
        :class:`tdamapper.cover.StandardCubicalCover` or proximity cubical
        cover, as in :class:`tdamapper.cover.ProximityCubicalCover`.
        Acceptable values are 'standard' or 'proximity'. Defaults to
        'proximity'.
    :type algorithm: str
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
        algorithm: str = "proximity",
        kind: str = "flat",
        leaf_capacity: int = 1,
        leaf_radius: Optional[float] = None,
        pivoting: Optional[str] = None,
    ):
        self.n_intervals = n_intervals
        self.overlap_frac = overlap_frac
        self.algorithm = algorithm
        self.kind = kind
        self.leaf_capacity = leaf_capacity
        self.leaf_radius = leaf_radius
        self.pivoting = pivoting

    def _get_cubical_cover(self):
        params = dict(
            n_intervals=self.n_intervals,
            overlap_frac=self.overlap_frac,
            kind=self.kind,
            leaf_capacity=self.leaf_capacity,
            leaf_radius=self.leaf_radius,
            pivoting=self.pivoting,
        )
        if self.algorithm == "proximity":
            return ProximityCubicalCover(**params)
        elif self.algorithm == "standard":
            return StandardCubicalCover(**params)
        else:
            raise ValueError(
                "The only possible values for algorithm are 'standard' and "
                "'proximity'."
            )

    def fit(self, X: ArrayLike) -> CubicalCover:
        """
        Train internal parameters.

        This method delegates to the :func:`fit` method of the internal cubical
        cover used.

        :param X: A dataset of n points.
        :type X: array-like of shape (n, m) or list-like of length n
        :return: The object itself.
        :rtype: self
        """
        self._cubical_cover = self._get_cubical_cover()
        self._cubical_cover.fit(X)
        return self

    def transform(self, X: ArrayLike) -> Generator[List[int], None, None]:
        """
        Covers the dataset using hypercubes.

        This method delegates to the `apply` method of the internal cubical
        cover used.

        :param X: A dataset of n points.
        :type X: array-like of shape (n, m) or list-like of length n
        :return: A generator of lists of ids.
        :rtype: generator of lists of ints
        """
        return self._cubical_cover.transform(X)
