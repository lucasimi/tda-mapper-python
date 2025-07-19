"""
This module contains a builder for constructing the vp-tree from a dataset.
"""

from random import randrange
from typing import Callable, Generic, TypeVar

import numpy as np

from tdamapper.protocols import ArrayRead
from tdamapper.utils.vptree_hier.common import (
    Leaf,
    Node,
    Tree,
    VPArray,
    VPTreeType,
    _mid,
)

T = TypeVar("T")


class Builder(Generic[T]):
    """
    A builder for constructing the vp-tree from a dataset.

    :param vpt: The vantage point tree to build.
    :param X: The dataset from which to build the vp-tree.
    """

    _array: VPArray[T]
    _leaf_capacity: int
    _leaf_radius: float
    _pivoting: Callable[[int, int], None]

    def __init__(self, vpt: VPTreeType[T], X: ArrayRead[T]) -> None:
        self._distance = vpt.metric

        dataset = list(X)
        indices = np.array(list(range(len(dataset))))
        distances = np.array([0.0 for _ in X])
        self._array = VPArray(dataset, distances, indices)

        self._leaf_capacity = vpt.leaf_capacity
        self._leaf_radius = vpt.leaf_radius
        pivoting = vpt.pivoting
        self._pivoting = self._pivoting_disabled
        if pivoting == "random":
            self._pivoting = self._pivoting_random
        elif pivoting == "furthest":
            self._pivoting = self._pivoting_furthest

    def _pivoting_disabled(self, start: int, end: int) -> None:
        pass

    def _pivoting_random(self, start: int, end: int) -> None:
        if end <= start:
            return
        pivot = randrange(start, end)
        if pivot > start:
            self._array.swap(start, pivot)

    def _furthest(self, start: int, end: int, i: int) -> int:
        furthest_dist = 0.0
        furthest = start
        i_point = self._array.get_point(i)
        for j in range(start, end):
            j_point = self._array.get_point(j)
            j_dist = self._distance(i_point, j_point)
            if j_dist > furthest_dist:
                furthest = j
                furthest_dist = j_dist
        return furthest

    def _pivoting_furthest(self, start: int, end: int) -> None:
        if end <= start:
            return
        rnd = randrange(start, end)
        furthest_rnd = self._furthest(start, end, rnd)
        furthest = self._furthest(start, end, furthest_rnd)
        if furthest > start:
            self._array.swap(start, furthest)

    def _update(self, start: int, end: int) -> None:
        self._pivoting(start, end)
        v_point = self._array.get_point(start)
        for i in range(start + 1, end):
            point = self._array.get_point(i)
            self._array.set_distance(i, self._distance(v_point, point))

    def build(self) -> tuple[Tree[T], VPArray[T]]:
        """
        Build the vp-tree from the dataset.

        :return: A tuple containing the constructed vp-tree and the VPArray.
        """
        if self._array.size() > 0:
            tree = self._build_rec(0, self._array.size())
        else:
            tree = Leaf(0, 0)
        return tree, self._array

    def _build_rec(self, start: int, end: int) -> Tree[T]:
        if end - start <= self._leaf_capacity:
            return Leaf(start, end)
        mid = _mid(start, end)
        self._update(start, end)
        v_point = self._array.get_point(start)
        self._array.partition(start + 1, end, mid)
        v_radius = self._array.get_distance(mid)
        self._array.set_distance(start, v_radius)
        left: Tree[T]
        right: Tree[T]
        if v_radius <= self._leaf_radius:
            left = Leaf(start + 1, mid)
            right = Leaf(mid, end)
        else:
            left = self._build_rec(start + 1, mid)
            right = self._build_rec(mid, end)
        return Node(v_radius, v_point, left, right)
