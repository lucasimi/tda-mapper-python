"""
VP-tree Ball Search Module.

This module provides a BallSearch class for searching points within a specified distance (epsilon)
from a given point in a VP-tree. It uses an iterative approach to traverse the VP-tree
and collect points that meet the distance criteria.
"""

from typing import Generic, List, TypeVar

from tdamapper.vptree_flat.common import VPTreeType, _mid

T = TypeVar("T")


class BallSearch(Generic[T]):
    """
    BallSearch class for searching points within a specified distance (epsilon)
    from a given point in a VP-tree.

    This class performs a search in a VP-tree to find all points that are within
    a specified distance (epsilon) from a given point. It uses an iterative
    approach to traverse the VP-tree and collect points that meet the distance
    criteria.

    :param vpt: VPTreeType instance containing distance function and parameters.
    :param point: The point from which the search is performed.
    :param eps: The distance threshold (epsilon) for the search.
    :param inclusive: If True, points exactly at distance eps are included in the
        results. If False, they are excluded. Defaults to True.
    """

    def __init__(
        self,
        vpt: VPTreeType[T],
        point: T,
        eps: float,
        inclusive: bool = True,
    ):
        self._arr = vpt.array
        self._distance = vpt.distance
        self._point = point
        self._eps = eps
        self._inclusive = inclusive

    def search(self) -> List[T]:
        """
        Perform the search for points within the specified distance from the point.

        This method initiates the search process and returns a list of points
        that are within the specified distance (epsilon) from the given point.

        :return: A list of points that are within the specified distance from the
            given point.
        """
        return self._search_iter()

    def _inside(self, dist: float) -> bool:
        if self._inclusive:
            return dist <= self._eps
        return dist < self._eps

    def _search_iter(self) -> List[T]:
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
