"""
A module for performing ball search in a vantage-point tree (VP-tree).
"""

from typing import Generic, TypeVar

from tdamapper.utils.metrics import Metric
from tdamapper.utils.vptree_hier.common import Tree, VPArray, VPTreeType

T = TypeVar("T")


class BallSearch(Generic[T]):
    """
    A class to perform a ball search in a vantage-point tree (VP-tree).

    This class searches for all points within a specified distance (eps)
    from a given point in the VP-tree. The search can be inclusive or exclusive
    of the boundary defined by eps.

    :param vpt: The VP-tree to search in.
    :param point: The point from which to search.
    :param eps: The distance threshold for the search.
    :param inclusive: If True, points exactly at distance eps are included in the result.
    """

    _tree: Tree[T]
    _array: VPArray[T]
    _metric: Metric[T]
    _result: list[T]

    def __init__(
        self, vpt: VPTreeType[T], point: T, eps: float, inclusive: bool = True
    ) -> None:
        self.point = point
        self.eps = eps
        self.inclusive = inclusive
        self._tree = vpt.tree
        self._array = vpt.array
        self._metric = vpt.metric
        self._result = []

    def search(self) -> list[T]:
        """
        Perform the ball search and return all points within the specified distance.

        :return: A list of points within the specified distance from the given point.
        """
        self._result.clear()
        self._search_rec(self._tree)
        return self._result

    def _inside(self, dist: float) -> bool:
        if self.inclusive:
            return dist <= self.eps
        return dist < self.eps

    def _search_rec(self, tree: Tree[T]) -> None:
        if tree.is_terminal():
            bounds = tree.get_bounds()
            if bounds is not None:
                start, end = bounds
                for x in self._array.get_points(start, end):
                    dist = self._metric(self.point, x)
                    if self._inside(dist):
                        self._result.append(x)
        else:
            ball = tree.get_ball()
            if ball is not None:
                v_radius, v_point = ball
                dist = self._metric(v_point, self.point)
                if self._inside(dist):
                    self._result.append(v_point)
                if dist <= v_radius:
                    fst, snd = tree.get_left(), tree.get_right()
                else:
                    fst, snd = tree.get_right(), tree.get_left()
                if fst is not None:
                    self._search_rec(fst)
                if abs(dist - v_radius) <= self.eps and snd is not None:
                    self._search_rec(snd)
