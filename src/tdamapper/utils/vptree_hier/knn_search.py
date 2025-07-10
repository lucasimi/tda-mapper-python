from typing import Generic, TypeVar

from tdamapper.utils.heap import MaxHeap
from tdamapper.utils.metrics import Metric
from tdamapper.utils.vptree_hier.common import Tree, VPArray, VPTreeType

T = TypeVar("T")


class KnnSearch(Generic[T]):

    _tree: Tree[T]
    _arr: VPArray[T]
    _distance: Metric[T]
    _items: MaxHeap[float, T]

    def __init__(self, vpt: VPTreeType[T], point: T, neighbors: int) -> None:
        self.point = point
        self.neighbors = neighbors
        self._tree = vpt._get_tree()
        self._arr = vpt._get_arr()
        self._distance = vpt._get_distance()
        self._items = MaxHeap()

    def _add(self, dist: float, x: T) -> None:
        self._items.add(dist, x)
        if len(self._items) > self.neighbors:
            self._items.pop()

    def _get_items(self) -> list[T]:
        while len(self._items) > self.neighbors:
            self._items.pop()
        return [x for (_, x) in self._items]

    def _get_radius(self) -> float:
        if len(self._items) < self.neighbors:
            return float("inf")
        furthest_dist, _ = self._items.top()
        if furthest_dist is None:
            return float("inf")
        return furthest_dist

    def search(self) -> list[T]:
        self._search_rec(self._tree)
        return self._get_items()

    def _search_rec(self, tree: Tree[T]) -> None:
        if tree.is_terminal():
            bounds = tree.get_bounds()
            if bounds is not None:
                start, end = bounds
                for x in self._arr.get_points(start, end):
                    dist = self._distance(self.point, x)
                    if dist < self._get_radius():
                        self._add(dist, x)
        else:
            ball = tree.get_ball()
            if ball is not None:
                v_radius, v_point = ball
                dist = self._distance(v_point, self.point)
                if dist < self._get_radius():
                    self._add(dist, v_point)
                if dist <= v_radius:
                    fst, snd = tree.get_left(), tree.get_right()
                else:
                    fst, snd = tree.get_right(), tree.get_left()
                if fst is not None:
                    self._search_rec(fst)
                if abs(dist - v_radius) <= self._get_radius() and snd is not None:
                    self._search_rec(snd)
