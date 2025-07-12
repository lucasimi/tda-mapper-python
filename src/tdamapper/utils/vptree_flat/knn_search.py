from typing import Callable, Generic, TypeVar

from tdamapper.utils.heap import MaxHeap
from tdamapper.utils.vptree_flat.common import VPArray, VPTreeType, _mid

_PRE = 0
_POST = 1

T = TypeVar("T")


class KnnSearch(Generic[T]):

    _arr: VPArray[T]
    _distance: Callable[[T, T], float]
    _point: T
    _neighbors: int
    _radius: float
    _result: MaxHeap[float, T]

    def __init__(self, vpt: VPTreeType[T], point: T, neighbors: int) -> None:
        self._array = vpt.array
        self._distance = vpt.metric
        self._point = point
        self._neighbors = neighbors
        self._radius = float("inf")
        self._result = MaxHeap()

    def _get_items(self) -> list[T]:
        while len(self._result) > self._neighbors:
            self._result.pop()
        return [x for (_, x) in self._result]

    def search(self) -> list[T]:
        self._search_iter()
        return self._get_items()

    def _process(self, x: T) -> float:
        dist = self._distance(self._point, x)
        if dist >= self._radius:
            return dist
        self._result.add(dist, x)
        while len(self._result) > self._neighbors:
            self._result.pop()
        if len(self._result) == self._neighbors:
            top = self._result.top()
            if top is not None:
                self._radius, _ = top
        return dist

    def _search_iter(self) -> list[T]:
        self._result = MaxHeap()
        stack = [(0, self._array.size(), 0.0, _PRE)]
        while stack:
            start, end, thr, action = stack.pop()

            v_radius = self._array.get_distance(start)
            v_point = self._array.get_point(start)
            is_terminal = self._array.is_terminal(start)

            if is_terminal:
                for x in self._array.get_points(start, end):
                    self._process(x)
            else:
                if action == _PRE:
                    mid = _mid(start, end)
                    dist = self._process(v_point)
                    if dist <= v_radius:
                        fst_start, fst_end = start + 1, mid
                        snd_start, snd_end = mid, end
                    else:
                        fst_start, fst_end = mid, end
                        snd_start, snd_end = start + 1, mid
                    stack.append((snd_start, snd_end, abs(v_radius - dist), _POST))
                    stack.append((fst_start, fst_end, 0.0, _PRE))
                elif action == _POST:
                    if self._radius > thr:
                        stack.append((start, end, 0.0, _PRE))
        return self._get_items()
