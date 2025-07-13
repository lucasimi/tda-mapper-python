"""
A module for fast knn and range searches, depending only on a given metric.

This module contains a flat array-based implementation of a vp-tree, which is a
data structure that allows for efficient nearest neighbor and range searches in
high-dimensional spaces.
"""

from __future__ import annotations

from typing import Generic, TypeVar

from tdamapper.protocols import ArrayRead, Metric
from tdamapper.utils.vptree_flat.ball_search import BallSearch
from tdamapper.utils.vptree_flat.builder import Builder
from tdamapper.utils.vptree_flat.common import PivotingStrategy, VPArray
from tdamapper.utils.vptree_flat.knn_search import KnnSearch

T = TypeVar("T")


class VPTree(Generic[T]):
    """
    A Vantage Point Tree, or vp-tree, for fast range-queries and knn-queries.

    :param X: A dataset of n points.
    :param metric: The metric used to define the distance between points.
        Accepts any value compatible with `tdamapper.utils.metrics.get_metric`.
    :param kind: Specifies whether to use a flat or a hierarchical vantage
        point tree.
    :param leaf_capacity: The maximum number of points in a leaf node of the
        vantage point tree. Must be a positive value.
    :param leaf_radius: The radius of the leaf nodes. Must be a positive
        value.
    :param pivoting: The method used for pivoting in the vantage point tree.
    """

    metric: Metric[T]
    leaf_capacity: int
    leaf_radius: float
    pivoting: PivotingStrategy
    array: VPArray[T]

    def __init__(
        self,
        X: ArrayRead[T],
        metric: Metric[T],
        leaf_capacity: int = 1,
        leaf_radius: float = 0.0,
        pivoting: PivotingStrategy = "disabled",
    ):
        self.metric = metric
        self.leaf_capacity = leaf_capacity
        self.leaf_radius = leaf_radius
        self.pivoting = pivoting
        self.array = Builder(self, X).build()

    def ball_search(self, point: T, eps: float, inclusive: bool = True) -> list[T]:
        """
        Perform a ball search in the Vantage Point Tree.

        This method searches for all points within a specified radius from a
        given point.

        :param point: The query point from which to search for neighbors.
        :param eps: The radius within which to search for neighbors. Must be
            positive.
        :param inclusive: Whether to include points exactly at the distance
            `eps` from `point`.
        :return: A list of points within the specified radius from the given
            query point.
        """
        return BallSearch(self, point, eps, inclusive).search()

    def knn_search(self, point: T, k: int) -> list[T]:
        """
        Perform a k-nearest neighbors search in the Vantage Point Tree.

        This method searches for the k-nearest neighbors to a given query
        point.

        :param point: The point from which to search for nearest neighbors.
        :param k: The number of nearest neighbors to search for. Must be
            positive.
        :return: A list of the k-nearest neighbors to the given query point.
        """
        return KnnSearch(self, point, k).search()
