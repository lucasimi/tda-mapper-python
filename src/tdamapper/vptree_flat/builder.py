"""
VP-tree Builder Module.

This module provides a Builder class for constructing a VP-tree from a
collection of items. It supports different pivoting strategies and allows
customization of the tree's parameters.
"""

from __future__ import annotations

from random import randrange
from typing import Generic, Iterable, TypeVar

import numpy as np

from tdamapper.vptree_flat.common import VPArray, VPTreeType


def _mid(start: int, end: int) -> int:
    return (start + end) // 2


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

    def __init__(self, vpt: VPTreeType[T], items: Iterable[T]):
        self._distance = vpt.distance

        dataset = list(items)
        indices = np.array(list(range(len(dataset))))
        distances = np.array([0.0 for _ in items])
        is_terminal = np.array([False for _ in items])
        self._arr = VPArray(dataset, distances, indices, is_terminal)

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
        is_terminal = self._arr.is_terminal(start)
        for i in range(start + 1, end):
            point = self._arr.get_point(i)
            self._arr.set_distance(i, self._distance(v_point, point))
            self._arr.set_terminal(i, is_terminal)

    def build(self) -> VPArray[T]:
        """
        Build the VP-tree from the items provided during initialization.

        This method constructs the VP-tree iteratively, starting from the root
        node.

        :return: The VPArray instance containing the constructed VP-tree.
        """
        self._build_iter()
        return self._arr

    def _build_iter(self) -> None:
        stack = [(0, self._arr.size())]
        while stack:
            start, end = stack.pop()
            mid = _mid(start, end)
            self._update(start, end)
            self._arr.partition(start + 1, end, mid)
            v_radius = self._arr.get_distance(mid)
            if (end - start > 2 * self._leaf_capacity) and (
                v_radius > self._leaf_radius
            ):
                self._arr.set_distance(start, v_radius)
                self._arr.set_terminal(start, False)
                stack.append((mid, end))
                stack.append((start + 1, mid))
            else:
                self._arr.set_distance(start, v_radius)
                self._arr.set_terminal(start, True)
