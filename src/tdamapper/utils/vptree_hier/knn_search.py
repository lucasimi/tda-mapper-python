from typing import Callable, Generic, TypeVar

from tdamapper.utils.heap import MaxHeap
from tdamapper.utils.vptree_hier.common import Tree, VPArray, VPTreeType

T = TypeVar("T")


class KnnSearch(Generic[T]):

    _tree: Tree[T]
    _array: VPArray[T]
    _distance: Callable[[T, T], float]
    _point: T
    _neighbors: int
    _items: MaxHeap[float, T]

    def __init__(self, vpt: VPTreeType[T], point: T, neighbors: int) -> None:
        self._tree = vpt.tree
        self._array = vpt.array
        self._distance = vpt.metric
        self._point = point
        self._neighbors = neighbors
        self._items = MaxHeap()

    def _add(self, dist: float, x: T) -> None:
        self._items.add(dist, x)
        if len(self._items) > self._neighbors:
            self._items.pop()

    def _get_items(self) -> list[T]:
        while len(self._items) > self._neighbors:
            self._items.pop()
        return [x for (_, x) in self._items]

    def _get_radius(self) -> float:
        if len(self._items) < self._neighbors:
            return float("inf")
        top = self._items.top()
        if top is None:
            return float("inf")
        furthest_dist, _ = top
        return furthest_dist

    def search(self) -> list[T]:
        self._search_rec(self._tree)
        return self._get_items()

    def _search_rec(self, tree: Tree[T]) -> None:
        if tree.is_terminal:
            bounds = tree.bounds
            if bounds is None:
                return
            start, end = bounds
            for x in self._array.get_points(start, end):
                dist = self._distance(self._point, x)
                if dist < self._get_radius():
                    self._add(dist, x)
        else:
            ball = tree.ball
            if ball is None:
                return
            v_radius, v_point = ball
            dist = self._distance(v_point, self._point)
            if dist < self._get_radius():
                self._add(dist, v_point)
            if dist <= v_radius:
                fst, snd = tree.left, tree.right
            else:
                fst, snd = tree.right, tree.left
            if fst is not None:
                self._search_rec(fst)
            if abs(dist - v_radius) <= self._get_radius() and snd is not None:
                self._search_rec(snd)
