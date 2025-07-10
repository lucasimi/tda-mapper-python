from typing import Generic, Iterable, Optional, TypeVar

from tdamapper.utils.metrics import Metric
from tdamapper.utils.vptree_flat.ball_search import BallSearch
from tdamapper.utils.vptree_flat.builder import Builder
from tdamapper.utils.vptree_flat.knn_search import KnnSearch

T = TypeVar("T")


class VPTree(Generic[T]):

    def __init__(
        self,
        items: Iterable[T],
        metric: Metric[T],
        leaf_capacity: int = 1,
        leaf_radius: float = 0.0,
        pivoting: Optional[str] = None,
    ) -> None:
        self.metric = metric
        self.leaf_capacity = leaf_capacity
        self.leaf_radius = leaf_radius
        self.pivoting = pivoting
        self.array = Builder(self, items).build()

    def ball_search(self, point: T, eps: float, inclusive: bool = True) -> list[T]:
        return BallSearch(self, point, eps, inclusive).search()

    def knn_search(self, point: T, k: int) -> list[T]:
        return KnnSearch(self, point, k).search()
