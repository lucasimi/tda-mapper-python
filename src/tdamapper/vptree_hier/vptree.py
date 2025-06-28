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
from tdamapper.vptree_hier.ball_search import BallSearch
from tdamapper.vptree_hier.builder import Builder
from tdamapper.vptree_hier.common import Tree, VPArray
from tdamapper.vptree_hier.knn_search import KnnSearch

T = TypeVar("T")


class VPTree(Generic[T]):

    def __init__(
        self,
        X: Iterable[T],
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
        self._tree, self._arr = Builder(self, X).build()

    def get_metric(self) -> Union[str, Callable]:
        return self._metric

    def get_metric_params(self) -> Optional[Dict[str, Any]]:
        return self._metric_params

    @property
    def leaf_capacity(self) -> int:
        return self._leaf_capacity

    @property
    def leaf_radius(self) -> float:
        return self._leaf_radius

    @property
    def pivoting(self) -> Optional[str]:
        return self._pivoting

    @property
    def tree(self) -> Tree[T]:
        return self._tree

    @property
    def array(self) -> VPArray[T]:
        return self._arr

    @property
    def distance(self) -> Callable[[T, T], float]:
        metric_params = self._metric_params or {}
        return get_metric(self._metric, **metric_params)

    def ball_search(self, point: T, eps: float, inclusive: bool = True) -> List[T]:
        return BallSearch(self, point, eps, inclusive).search()

    def knn_search(self, point: T, k: int) -> List[T]:
        return KnnSearch(self, point, k).search()
