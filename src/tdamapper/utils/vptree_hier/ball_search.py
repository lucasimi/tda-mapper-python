from __future__ import annotations

from typing import Callable, Generic, TypeVar

from tdamapper.utils.vptree_hier.common import Tree, VPArray, VPTreeType

T = TypeVar("T")


class BallSearch(Generic[T]):

    _tree: Tree[T]
    _array: VPArray[T]
    _distance: Callable[[T, T], float]
    _point: T
    _eps: float
    _inclusive: bool
    _result: list[T]

    def __init__(
        self, vpt: VPTreeType[T], point: T, eps: float, inclusive: bool = True
    ) -> None:
        self._tree = vpt.tree
        self._array = vpt.array
        self._distance = vpt.metric
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
        if tree.is_terminal:
            bounds = tree.bounds
            if bounds is None:
                return
            start, end = bounds
            for x in self._array.get_points(start, end):
                dist = self._distance(self._point, x)
                if self._inside(dist):
                    self._result.append(x)
        else:
            ball = tree.ball
            if ball is None:
                return
            v_radius, v_point = ball
            dist = self._distance(v_point, self._point)
            if self._inside(dist):
                self._result.append(v_point)
            if dist <= v_radius:
                fst, snd = tree.left, tree.right
            else:
                fst, snd = tree.right, tree.left
            if fst is not None:
                self._search_rec(fst)
            if abs(dist - v_radius) <= self._eps and snd is not None:
                self._search_rec(snd)
