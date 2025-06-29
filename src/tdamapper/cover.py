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

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

from tdamapper._common import ParamsMixin
from tdamapper.core import ArrayLike, PointLike, SpatialSearch
from tdamapper.search import (
    BallSearch,
    CubicalLandmarks,
    CubicalSearch,
    KNNSearch,
)


class ProximityNet(ABC, ParamsMixin):
    """
    Base class for proximity net algorithms, which cover data using spatial
    search algorithms.

    This class provides a common interface for applying spatial search
    algorithms to datasets, yielding the indices of covered points based on
    the search results.
    """

    landmarks_: List[PointLike]
    labels_: List[List[int]]

    @abstractmethod
    def spatial_search(self) -> SpatialSearch:
        """
        Returns a new instance of a spatial search algorithm.

        This method should be implemented by subclasses to provide
        the specific spatial search algorithm to be used.

        :return: An instance of a spatial search algorithm.
        """

    def _cover(self, X: ArrayLike) -> Generator[Tuple[int, List[int]], None, None]:
        """
        Covers the dataset using landmarks and spatial search.

        This function yields pairs of indices and lists of ids, where each
        pair represents a point in the dataset and the indices of its covered
        neighbors. The first element of the pair is the index of the point in
        the dataset, and the second element is a list of ids of the covered
        neighbors. If the index is -1, it indicates that the point is a
        landmark, and the list of ids contains the indices of its neighbors.

        :param X: A dataset of n points.
        :return: A generator of pairs of indices and lists of ids.
        :yield: A tuple containing the index of the point and a list of ids of
                its covered neighbors. If the index is -1, it indicates a
                landmark, and the list contains the indices of its neighbors.
        """
        search = self.spatial_search()
        search.fit(X)
        covered_ids = set()
        for x in self.landmarks_:
            neigh_ids = search.search(x)
            covered_ids.update(neigh_ids)
            if neigh_ids:
                yield -1, neigh_ids
        for i, xi in enumerate(X):
            if i not in covered_ids:
                neigh_ids = search.search(xi)
                covered_ids.update(neigh_ids)
                if neigh_ids:
                    yield i, neigh_ids

    def fit(self, X: ArrayLike) -> ProximityNet:
        """
        Train internal parameters.

        This method applies the spatial search algorithm to the dataset and
        stores the landmarks and labels.

        :param X: A dataset of n points.
        :return: The object itself.
        """
        self.landmarks_ = []
        self.labels_ = []
        for i, neigh_ids in self._cover(X):
            if i >= 0:
                self.landmarks_.append(X[i])
            self.labels_.append(neigh_ids)
        return self

    def transform(self, X: ArrayLike) -> Generator[List[int], None, None]:
        """
        Covers the dataset using landmarks.

        This function yields all the indices of the covered neighbors for each
        point in the dataset. The indices are the indices of the points in the
        original dataset.

        :param X: A dataset of n points.
        :return: A generator of lists of ids.
        """
        for _, neigh_ids in self._cover(X):
            yield neigh_ids

    def fit_transform(self, X: ArrayLike) -> Generator[List[int], None, None]:
        """
        Fit the model and transform the dataset.

        This function first fits the model to the dataset and then yields
        the indices of the covered neighbors for each point in the dataset.

        :param X: A dataset of n points.
        :return: A generator of lists of ids.
        """
        self.fit(X)
        for neigh_ids in self.labels_:
            yield neigh_ids


class BallCover(ProximityNet):
    """
    Cover algorithm based on `ball proximity function`, which covers data with
    open balls of fixed radius.

    An open ball is a set of points within a specified distance from a center
    point. This class maps each point to its corresponding open ball with a
    fixed radius centered on the point itself.

    :param radius: The radius of the open balls. Must be a positive value.
        Defaults to 1.0.
    :param metric: The metric used to define the distance between points.
        Accepts any value compatible with `tdamapper.metrics.get_metric`.
        Defaults to 'euclidean'.
    :param metric_params: Additional parameters for the metric function, to be
        passed to `tdamapper.metrics.get_metric`. Defaults to None.
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

    def spatial_search(self) -> BallSearch:
        return BallSearch(
            radius=self.radius,
            metric=self.metric,
            metric_params=self.metric_params,
            kind=self.kind,
            leaf_capacity=self.leaf_capacity,
            leaf_radius=self.leaf_radius,
            pivoting=self.pivoting,
        )


class KNNCover(ProximityNet):
    """
    Cover algorithm based on `KNN proximity function`, which covers data using
    k-nearest neighbors (KNN).

    This class maps each point to the set of the k nearest neighbors to the
    point itself.

    :param neighbors: The number of neighbors to use for the KNN Proximity
        function, must be positive and less than the length of the dataset.
        Defaults to 1.
    :param metric: The metric used to define the distance between points.
        Accepts any value compatible with `tdamapper.metrics.get_metric`.
        Defaults to 'euclidean'.
    :param metric_params: Additional parameters for the metric function, to be
        passed to `tdamapper.metrics.get_metric`. Defaults to None.
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

    def spatial_search(self) -> KNNSearch:
        return KNNSearch(
            neighbors=self.neighbors,
            metric=self.metric,
            metric_params=self.metric_params,
            kind=self.kind,
            leaf_capacity=self.leaf_capacity,
            leaf_radius=self.leaf_radius,
            pivoting=self.pivoting,
        )


class ProximityCubicalCover(ProximityNet):
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

    def spatial_search(self) -> CubicalSearch:
        return CubicalSearch(
            n_intervals=self.n_intervals,
            overlap_frac=self.overlap_frac,
            kind=self.kind,
            leaf_capacity=self.leaf_capacity,
            leaf_radius=self.leaf_radius,
            pivoting=self.pivoting,
        )


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

    _landmarks: CubicalLandmarks

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
        :return: A generator of lists of ids.
        """
        self.fit(X)
        lmrks_to_cover = self._landmarks.landmarks(X)
        while lmrks_to_cover:
            _, x = lmrks_to_cover.popitem()
            neigh_ids = self._landmarks.search(x)
            if neigh_ids:
                yield neigh_ids

    def fit_transform(self, X: ArrayLike) -> Generator[List[int], None, None]:
        self.fit(X)
        return self.transform(X)


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

    _cubical_cover: Union[StandardCubicalCover, ProximityCubicalCover]

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
        if self.algorithm == "standard":
            return StandardCubicalCover(**params)
        raise ValueError(
            "The only possible values for algorithm are 'standard' and 'proximity'."
        )

    def fit(self, X: ArrayLike) -> CubicalCover:
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

    def transform(self, X: ArrayLike) -> Generator[List[int], None, None]:
        """
        Covers the dataset using hypercubes.

        This method delegates to the `apply` method of the internal cubical
        cover used.

        :param X: A dataset of n points.
        :return: A generator of lists of ids.
        """
        return self._cubical_cover.transform(X)

    def fit_transform(self, X: ArrayLike) -> Generator[List[int], None, None]:
        """
        Fit the model and transform the dataset.

        This method first fits the model to the dataset and then yields
        the indices of the covered neighbors for each point in the dataset.

        :param X: A dataset of n points.
        :return: A generator of lists of ids.
        """
        self._cubical_cover = self._get_cubical_cover()
        return self._cubical_cover.fit_transform(X)
