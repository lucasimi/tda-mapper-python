"""
VP-tree Ball Search Module.

This module provides a BallSearch class for searching points within a specified
distance (epsilon) from a given point in a VP-tree. It uses an iterative
approach to traverse the VP-tree and collect points that meet the distance
criteria.
"""

from __future__ import annotations

from typing import Generic, List, TypeVar

from tdamapper.vptree_hier.common import Tree, VPTreeType

T = TypeVar("T")


class BallSearch(Generic[T]):
    """
    BallSearch class for searching points within a specified distance (epsilon)
    from a given point in a VP-tree.

    This class performs a search in a VP-tree to find all points that are
    within a specified distance (epsilon) from a given point. It uses an
    iterative approach to traverse the VP-tree and collect points that meet the
    distance criteria.

    :param vpt: VPTreeType instance containing distance function and parameters.
    :param point: The point from which the search is performed.
    :param eps: The distance threshold (epsilon) for the search.
    :param inclusive: If True, points exactly at distance eps are included in
        the results. If False, they are excluded. Defaults to True.
    """

    _result: List[T]

    def __init__(
        self,
        vpt: VPTreeType[T],
        point: T,
        eps: float,
        inclusive: bool = True,
    ):
        self._tree = vpt.tree
        self._arr = vpt.array
        self._distance = vpt.distance
        self._point = point
        self._eps = eps
        self._inclusive = inclusive
        self._result = []

    def search(self) -> List[T]:
        """
        Perform the search for points within the specified distance from the
        point.

        This method initiates the search process and returns a list of points
        that are within the specified distance (epsilon) from the given point.

        :return: A list of points that are within the specified distance from
            the given point.
        """
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
