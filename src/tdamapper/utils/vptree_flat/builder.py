from random import randrange
from typing import Callable, Generic, Iterable, TypeVar

import numpy as np

from tdamapper.utils.metrics import Metric
from tdamapper.utils.vptree_flat.common import VPArray, VPTreeType

T = TypeVar("T")


def _mid(start, end):
    return (start + end) // 2


class Builder(Generic[T]):

    _arr: VPArray[T]
    _leaf_capacity: int
    _leaf_radius: float
    _distance: Metric[T]
    _pivoting: Callable[[int, int], None]

    def __init__(self, vpt: VPTreeType[T], items: Iterable[T]) -> None:
        self._distance = vpt._get_distance()

        dataset = [x for x in items]
        indices = np.array([i for i in range(len(dataset))])
        distances = np.array([0.0 for _ in items])
        is_terminal = np.array([False for _ in items])
        self._arr = VPArray(dataset, distances, indices, is_terminal)

        self._leaf_capacity = vpt.get_leaf_capacity()
        self._leaf_radius = vpt.get_leaf_radius()
        pivoting = vpt.get_pivoting()
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
