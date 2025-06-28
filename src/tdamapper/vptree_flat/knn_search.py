"""
This module implements a k-nearest neighbors search in a VP-tree.

It provides a KnnSearch class that allows searching for the k-nearest
neighbors of a given point in a VP-tree. The search is performed iteratively,
and the results are returned as a list of points that are the k-nearest
neighbors.
"""

from typing import Generic, List, TypeVar

from tdamapper.heap import MaxHeap
from tdamapper.vptree_flat.common import VPTreeType, _mid

_PRE = 0
_POST = 1


T = TypeVar("T")


class KnnSearch(Generic[T]):
    """
    KnnSearch class for searching k-nearest neighbors of a point in a VP-tree.

    This class performs a search in a VP-tree to find the k-nearest neighbors
    of a given point. It uses an iterative approach to traverse the VP-tree
    and collect points that are within the specified number of neighbors.

    :param vpt: VPTreeType instance containing distance function and
        parameters.
    :param point: The point from which the search is performed.
    :param neighbors: The number of nearest neighbors to find.
    """

    _result: MaxHeap[float, T]
    _radius: float

    def __init__(self, vpt: VPTreeType[T], point: T, neighbors: int):
        self._arr = vpt.array
        self._distance = vpt.distance
        self._point = point
        self._neighbors = neighbors
        self._radius = float("inf")
        self._result = MaxHeap()

    def _get_items(self) -> List[T]:
        while len(self._result) > self._neighbors:
            self._result.pop()
        return [x for (_, x) in self._result]

    def search(self) -> List[T]:
        """
        Perform the search for k-nearest neighbors of the point.
        This method initiates the search process and returns a list of points
        that are the k-nearest neighbors of the given point.

        :return: A list of points that are the k-nearest neighbors of the
            given point.
        """
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
            rad, _ = self._result.top()
            if rad is not None:
                self._radius = rad
        return dist

    def _search_iter(self) -> List[T]:
        self._result = MaxHeap()
        stack = [(0, self._arr.size(), 0.0, _PRE)]
        while stack:
            start, end, thr, action = stack.pop()

            v_radius = self._arr.get_distance(start)
            v_point = self._arr.get_point(start)
            is_terminal = self._arr.is_terminal(start)

            if is_terminal:
                for x in self._arr.get_points(start, end):
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
