from typing import Any, Generic, Iterable, Optional, TypeVar

from tdamapper.utils.metrics import Metric
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
        metric: Metric[T],
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
        self.tree, self.array = Builder(self, items).build()

    def ball_search(self, point: T, eps: float, inclusive: bool = True) -> list[T]:
        return BallSearch(self, point, eps, inclusive).search()

    def knn_search(self, point: T, k: int) -> list[T]:
        return KnnSearch(self, point, k).search()
