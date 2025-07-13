"""
This module provides a ball search implementation for a vp-tree.
"""

from __future__ import annotations

from typing import Callable, Generic, TypeVar

from tdamapper.utils.vptree_flat.common import VPArray, VPTreeType, _mid

T = TypeVar("T")


class BallSearch(Generic[T]):
    """
    A ball search implementation for a vp-tree.

    :param vpt: The vantage point tree to search.
    :param point: The point for which to find points within a certain distance.
    :param eps: The radius of the ball to search within.
    :param inclusive: Whether to include points at the boundary of the ball.
    """

    _array: VPArray[T]
    _distance: Callable[[T, T], float]
    _point: T
    _eps: float
    _inclusive: bool

    def __init__(
        self, vpt: VPTreeType[T], point: T, eps: float, inclusive: bool = True
    ) -> None:
        self._array = vpt.array
        self._distance = vpt.metric
        self._point = point
        self._eps = eps
        self._inclusive = inclusive

    def search(self) -> list[T]:
        """
        Perform a ball search in the vp-tree.

        :return: A list of points within the specified radius of the point.
        """
        return self._search_iter()

    def _inside(self, dist: float) -> bool:
        if self._inclusive:
            return dist <= self._eps
        return dist < self._eps

    def _search_iter(self) -> list[T]:
        stack = [(0, self._array.size())]
        result = []
        while stack:
            start, end = stack.pop()
            v_radius = self._array.get_distance(start)
            v_point = self._array.get_point(start)
            is_terminal = self._array.is_terminal(start)
            if is_terminal:
                for x in self._array.get_points(start, end):
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
