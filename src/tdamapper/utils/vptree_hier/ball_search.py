from typing import Callable, Generic, Iterable, Optional, TypeVar, Union

from tdamapper.utils.metrics import Metric
from tdamapper.utils.vptree_hier.common import Tree, VPArray, VPTreeType

T = TypeVar("T")


class BallSearch(Generic[T]):

    _tree: Tree[T]
    _arr: VPArray[T]
    _distance: Metric[T]
    _point: T
    _eps: float
    _inclusive: bool
    _result: list[T]

    def __init__(
        self, vpt: VPTreeType[T], point: T, eps: float, inclusive: bool = True
    ) -> None:
        self._tree = vpt._get_tree()
        self._arr = vpt._get_arr()
        self._distance = vpt._get_distance()
        self._point = point
        self._eps = eps
        self._inclusive = inclusive
        self._result = []

    def search(self) -> list[T]:
        self._result.clear()
        self._search_rec(self._tree)
        return self._result

    def _inside(self, dist: float) -> bool:
        if self._inclusive:
            return dist <= self._eps
        return dist < self._eps

    def _search_rec(self, tree: Tree[T]) -> None:
        if tree.is_terminal():
            start, end = tree.get_bounds()
            for x in self._arr.get_points(start, end):
                dist = self._distance(self._point, x)
                if self._inside(dist):
                    self._result.append(x)
        else:
            v_radius, v_point = tree.get_ball()
            dist = self._distance(v_point, self._point)
            if self._inside(dist):
                self._result.append(v_point)
            if dist <= v_radius:
                fst, snd = tree.get_left(), tree.get_right()
            else:
                fst, snd = tree.get_right(), tree.get_left()
            self._search_rec(fst)
            if abs(dist - v_radius) <= self._eps:
                self._search_rec(snd)
