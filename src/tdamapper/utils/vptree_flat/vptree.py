from __future__ import annotations

from typing import Any, Optional, Union

from tdamapper._common import Array
from tdamapper.utils.metrics import Metric, MetricLiteral, get_metric
from tdamapper.utils.vptree_flat.ball_search import BallSearch
from tdamapper.utils.vptree_flat.builder import Builder
from tdamapper.utils.vptree_flat.common import PivotingStrategy, VPArray
from tdamapper.utils.vptree_flat.knn_search import KnnSearch


class VPTree:

    _metric: Union[MetricLiteral, Metric]
    _metric_params: Optional[dict[str, Any]]
    leaf_capacity: int
    leaf_radius: float
    pivoting: PivotingStrategy
    array: VPArray[Any]

    def __init__(
        self,
        X: Array[Any],
        metric: Union[MetricLiteral, Metric] = "euclidean",
        metric_params: Optional[dict[str, Any]] = None,
        leaf_capacity: int = 1,
        leaf_radius: float = 0.0,
        pivoting: PivotingStrategy = "disabled",
    ):
        self._metric = metric
        self._metric_params = metric_params
        self.metric = get_metric(metric, **(metric_params or {}))
        self.leaf_capacity = leaf_capacity
        self.leaf_radius = leaf_radius
        self.pivoting = pivoting
        self.array = Builder(self, X).build()

    def ball_search(self, point: Any, eps: float, inclusive: bool = True) -> list[Any]:
        return BallSearch(self, point, eps, inclusive).search()

    def knn_search(self, point: Any, k: int) -> list[Any]:
        return KnnSearch(self, point, k).search()
