from typing import Generic, TypeVar

from tdamapper.utils.metrics import Metric
from tdamapper.utils.vptree_flat.common import VPArray, VPTreeType, _mid

T = TypeVar("T")


class BallSearch(Generic[T]):

    _arr: VPArray[T]
    _distance: Metric[T]
    _point: T
    _eps: float
    _inclusive: bool

    def __init__(
        self, vpt: VPTreeType[T], point: T, eps: float, inclusive: bool = True
    ):
        self._arr = vpt._get_arr()
        self._distance = vpt._get_distance()
        self._point = point
        self._eps = eps
        self._inclusive = inclusive

    def search(self) -> list[T]:
        return self._search_iter()

    def _inside(self, dist: float) -> bool:
        if self._inclusive:
            return dist <= self._eps
        return dist < self._eps

    def _search_iter(self) -> list[T]:
        stack = [(0, self._arr.size())]
        result = []
        while stack:
            start, end = stack.pop()
            v_radius = self._arr.get_distance(start)
            v_point = self._arr.get_point(start)
            is_terminal = self._arr.is_terminal(start)
            if is_terminal:
                for x in self._arr.get_points(start, end):
                    dist = self._distance(self._point, x)
                    if self._inside(dist):
                        result.append(x)
            else:
                dist = self._distance(self._point, v_point)
                mid = _mid(start, end)
                if self._inside(dist):
                    result.append(v_point)
                if dist <= v_radius:
                    fst = (start + 1, mid)
                    snd = (mid, end)
                else:
                    fst = (mid, end)
                    snd = (start + 1, mid)
                if abs(dist - v_radius) <= self._eps:
                    stack.append(snd)
                stack.append(fst)
        return result
