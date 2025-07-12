from __future__ import annotations

from typing import Generic, TypeVar

from tdamapper.protocols import Array, Metric
from tdamapper.utils.vptree_flat.ball_search import BallSearch
from tdamapper.utils.vptree_flat.builder import Builder
from tdamapper.utils.vptree_flat.common import PivotingStrategy, VPArray
from tdamapper.utils.vptree_flat.knn_search import KnnSearch

T = TypeVar("T")


class VPTree(Generic[T]):

    metric: Metric[T]
    leaf_capacity: int
    leaf_radius: float
    pivoting: PivotingStrategy
    array: VPArray[T]

    def __init__(
        self,
        X: Array[T],
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
        return BallSearch(self, point, eps, inclusive).search()

    def knn_search(self, point: T, k: int) -> list[T]:
        return KnnSearch(self, point, k).search()
