"""
VP-tree Builder Module.

This module provides a Builder class for constructing a VP-tree from a
collection of items. It supports different pivoting strategies and allows
customization of the tree's parameters.
"""

from __future__ import annotations

from random import randrange
from typing import Generic, Iterable, Tuple, TypeVar

import numpy as np

from tdamapper.vptree_hier.common import (
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
    Builder for constructing a VP-tree from a collection of items.

    This class takes a VPTreeType and an iterable of items, and builds a
    VP-tree using the specified pivoting strategy and parameters.

    :param vpt: VPTreeType instance containing distance function and
        parameters.
    :param items: Iterable of items to be included in the VP-tree.
    """

    def __init__(self, vpt: VPTreeType, items: Iterable[T]) -> None:
        self._distance = vpt.distance

        dataset = list(items)
        indices = np.array(list(range(len(dataset))))
        distances = np.array([0.0 for _ in items])
        self._arr = VPArray(dataset, distances, indices)

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
            self._arr.swap(start, pivot)

    def _furthest(self, start: int, end: int, i: int) -> int:
        furthest_dist = 0.0
        furthest = start
        i_point = self._arr.get_point(i)
        for j in range(start, end):
            j_point = self._arr.get_point(j)
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
            self._arr.swap(start, furthest)

    def _update(self, start: int, end: int) -> None:
        self._pivoting(start, end)
        v_point = self._arr.get_point(start)
        for i in range(start + 1, end):
            point = self._arr.get_point(i)
            self._arr.set_distance(i, self._distance(v_point, point))

    def build(self) -> Tuple[Tree[T], VPArray[T]]:
        """
        Build the VP-tree from the items provided during initialization.

        This method constructs the VP-tree recursively, starting from the root
        node.

        :return: A tuple containing the root of the VP-tree and the VPArray
            instance.
        """
        tree = self._build_rec(0, self._arr.size())
        return tree, self._arr

    def _build_rec(self, start: int, end: int) -> Tree[T]:
        mid = _mid(start, end)
        self._update(start, end)
        v_point = self._arr.get_point(start)
        self._arr.partition(start + 1, end, mid)
        v_radius = self._arr.get_distance(mid)
        self._arr.set_distance(start, v_radius)
        left: Tree
        right: Tree
        if (end - start <= 2 * self._leaf_capacity) or (v_radius <= self._leaf_radius):
            left = Leaf(start + 1, mid)
            right = Leaf(mid, end)
        else:
            left = self._build_rec(start + 1, mid)
            right = self._build_rec(mid, end)
        return Node(v_radius, v_point, left, right)
