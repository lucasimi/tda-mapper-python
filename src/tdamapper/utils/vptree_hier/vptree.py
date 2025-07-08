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

    _metric: Union[str, Metric[T]]
    _metric_params: Optional[dict[str, Any]]
    _leaf_capacity: int
    _leaf_radius: float
    _pivoting: Optional[str]
    _tree: Tree[T]

    def __init__(
        self,
        items: Iterable[T],
        metric: Union[str, Metric[T]] = "euclidean",
        metric_params: Optional[dict[str, Any]] = None,
        leaf_capacity: int = 1,
        leaf_radius: float = 0.0,
        pivoting: Optional[str] = None,
    ):
        self._metric = metric
        self._metric_params = metric_params
        self._leaf_capacity = leaf_capacity
        self._leaf_radius = leaf_radius
        self._pivoting = pivoting
        self._tree, self._arr = Builder(self, items).build()

    def get_metric(self) -> Union[str, Metric[T]]:
        return self._metric

    def get_metric_params(self) -> Optional[dict[str, Any]]:
        return self._metric_params

    def get_leaf_capacity(self) -> int:
        return self._leaf_capacity

    def get_leaf_radius(self) -> float:
        return self._leaf_radius

    def get_pivoting(self) -> Optional[str]:
        return self._pivoting

    def _get_tree(self):
        return self._tree

    def _get_arr(self) -> VPArray[T]:
        return self._arr

    def _get_distance(self) -> Union[Metric[T], Metric[NDArray[np.float64]]]:
        metric_params = self._metric_params or {}
        return get_metric(self._metric, **metric_params)

    def ball_search(self, point, eps, inclusive=True) -> list[T]:
        return BallSearch(self, point, eps, inclusive).search()

    def knn_search(self, point, k) -> list[T]:
        return KnnSearch(self, point, k).search()
