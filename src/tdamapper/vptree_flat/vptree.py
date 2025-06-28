"""
This module implements a VP-tree (Vantage Point Tree) for efficient nearest
neighbor search. It provides a VPTree class that allows for the construction of
a VP-tree from a collection of items and supports searching for points within a
specified distance (ball search) or finding the k-nearest neighbors of a point
(k-nearest neighbors search).
"""

from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    TypeVar,
    Union,
)

from tdamapper.metrics import get_metric
from tdamapper.vptree_flat.ball_search import BallSearch
from tdamapper.vptree_flat.builder import Builder
from tdamapper.vptree_flat.common import VPArray
from tdamapper.vptree_flat.knn_search import KnnSearch

T = TypeVar("T")


class VPTree(Generic[T]):
    """
    VPTree class for constructing and searching a VP-tree.

    This class allows for the construction of a VP-tree from a collection of
    items and provides methods for searching the tree using ball search and
    k-nearest neighbors search. It supports various distance metrics and
    parameters for customizing the tree's behavior.

    :param items: An iterable of items of type T to be included in the
        VP-tree.
    :param metric: The distance metric to use for comparisons. Can be a string
        (e.g., "euclidean") or a callable function that takes two items of type
        T and returns a float distance.
    :param metric_params: Optional dictionary of parameters for the distance
        metric function.
    :param leaf_capacity: The maximum number of items allowed in a leaf node of
        the VP-tree. Defaults to 1.
    :param leaf_radius: The maximum radius of a leaf node in the VP-tree.
        Defaults to 0.0.
    :param pivoting: Optional pivoting strategy for the VP-tree. Can be
        "random" for random pivoting, "furthest" for furthest point pivoting,
        or None for no pivoting. Defaults to None.
    """

    def __init__(
        self,
        items: Iterable[T],
        metric: Union[str, Callable] = "euclidean",
        metric_params: Optional[Dict[str, Any]] = None,
        leaf_capacity: int = 1,
        leaf_radius: float = 0.0,
        pivoting: Optional[str] = None,
    ):
        self._metric = metric
        self._metric_params = metric_params
        self._leaf_capacity = leaf_capacity
        self._leaf_radius = leaf_radius
        self._pivoting = pivoting
        self._arr = Builder(self, items).build()

    def get_metric(self) -> Union[str, Callable]:
        """
        Get the distance metric used in the VP-tree.

        :return: The distance metric, which can be a string, e.g.,
            "euclidean", or a callable function that takes two items of type T
            and returns a float distance.
        """
        return self._metric

    def get_metric_params(self) -> Optional[Dict[str, Any]]:
        """
        Get the parameters for the distance metric used in the VP-tree.

        :return: A dictionary of parameters for the distance metric function,
            or None if no parameters are specified.
        """
        return self._metric_params

    @property
    def leaf_capacity(self) -> int:
        """
        Get the maximum number of items allowed in a leaf node of the VP-tree.

        :return: An integer representing the maximum number of items in a leaf
            node.
        """
        return self._leaf_capacity

    @property
    def leaf_radius(self) -> float:
        """
        Get the maximum radius of a leaf node in the VP-tree.

        :return: A float representing the maximum radius of a leaf node.
        """
        return self._leaf_radius

    @property
    def pivoting(self) -> Optional[str]:
        """
        Get the pivoting strategy used in the VP-tree.

        :return: A string indicating the pivoting strategy, which can be
            "random", "furthest", or None for no pivoting.
        """
        return self._pivoting

    @property
    def array(self) -> VPArray[T]:
        """
        Get the VPArray instance containing the dataset and distances.

        :return: An instance of VPArray containing the dataset and distances.
        """
        return self._arr

    @property
    def distance(self) -> Callable[[T, T], float]:
        """
        Get the distance function used to compute distances between points.

        :return: A callable that takes two points of type T and returns a
            float distance.
        """
        metric_params = self._metric_params or {}
        return get_metric(self._metric, **metric_params)

    def ball_search(
        self,
        point: T,
        eps: float,
        inclusive: bool = True,
    ) -> List[T]:
        """
        Perform a ball search in the VP-tree.

        This method searches for points within a specified distance (epsilon)
        from a given point in the VP-tree. It uses an iterative approach to
        traverse the tree and collect points that meet the distance criteria.

        :param point: The point from which the search is performed.
        :param eps: The distance threshold (epsilon) for the search.
        :param inclusive: If True, points exactly at distance eps are included
            in the results. If False, they are excluded. Defaults to True.
        :return: A list of points that are within the specified distance from
            the given point.
        """
        return BallSearch(self, point, eps, inclusive).search()

    def knn_search(self, point: T, k: int) -> List[T]:
        """
        Perform a k-nearest neighbors search in the VP-tree.

        This method searches for the k-nearest neighbors of a given point in
        the VP-tree. It uses an iterative approach to traverse the tree and
        collect points that are the k-nearest neighbors.

        :param point: The point from which the search is performed.
        :param k: The number of nearest neighbors to find.
        :return: A list of points that are the k-nearest neighbors of the
            given point.
        """
        return KnnSearch(self, point, k).search()
