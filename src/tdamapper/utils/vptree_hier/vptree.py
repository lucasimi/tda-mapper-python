from typing import Any, Generic, Iterable, Optional, TypeVar, Union

import numpy as np
from numpy.typing import NDArray

from tdamapper.utils.metrics import Metric, get_metric
from tdamapper.utils.vptree_hier.ball_search import BallSearch
from tdamapper.utils.vptree_hier.builder import Builder
from tdamapper.utils.vptree_hier.common import Tree, VPArray
from tdamapper.utils.vptree_hier.knn_search import KnnSearch

T = TypeVar("T")


class VPTree(Generic[T]):

    _tree: Tree[T]
    _arr: VPArray[T]

    def __init__(
        self,
        items: Iterable[T],
        metric: Union[str, Metric[T]] = "euclidean",
        metric_params: Optional[dict[str, Any]] = None,
        leaf_capacity: int = 1,
        leaf_radius: float = 0.0,
        pivoting: Optional[str] = None,
    ):
        self.metric = metric
        self.metric_params = metric_params
        self.leaf_capacity = leaf_capacity
        self.leaf_radius = leaf_radius
        self.pivoting = pivoting
        self._tree, self._arr = Builder(self, items).build()

    def _get_tree(self) -> Tree[T]:
        return self._tree

    def _get_arr(self) -> VPArray[T]:
        return self._arr

    def _get_distance(self) -> Metric[T]:
        metric_params = self.metric_params or {}
        return get_metric(self.metric, **metric_params)

    def ball_search(self, point: T, eps: float, inclusive: bool = True) -> list[T]:
        return BallSearch(self, point, eps, inclusive).search()

    def knn_search(self, point: T, k: int) -> list[T]:
        return KnnSearch(self, point, k).search()
