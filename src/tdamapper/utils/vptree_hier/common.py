from __future__ import annotations

from typing import Callable, Generic, Iterator, Literal, Optional, Protocol, TypeVar

import numpy as np
from numpy.typing import NDArray

from tdamapper._common import Array
from tdamapper.utils.quickselect import quickselect, swap_all


def _mid(start: int, end: int) -> int:
    return (start + end) // 2


T = TypeVar("T")

PivotingStrategy = Literal["disabled", "random", "furthest"]


class VPArray(Generic[T]):

    _dataset: Array[T]
    _distances: NDArray[np.float_]
    _indices: NDArray[np.int_]

    def __init__(
        self,
        dataset: Array[T],
        distances: NDArray[np.float_],
        indices: NDArray[np.int_],
    ):
        self._dataset = dataset
        self._distances = distances
        self._indices = indices

    def size(self) -> int:
        return len(self._dataset)

    def get_point(self, i: int) -> T:
        return self._dataset[self._indices[i]]

    def get_points(self, s: int, e: int) -> Iterator[T]:
        for x_index in self._indices[s:e]:
            yield self._dataset[x_index]

    def get_distance(self, i: int) -> float:
        return self._distances[i]

    def set_distance(self, i: int, dist: float) -> None:
        self._distances[i] = dist

    def swap(self, i: int, j: int) -> None:
        swap_all(self._distances, i, j, self._indices)

    def partition(self, s: int, e: int, k: int) -> None:
        quickselect(self._distances, s, e, k, self._indices)


class VPTreeType(Protocol[T]):

    @property
    def tree(self) -> Tree[T]: ...

    @property
    def array(self) -> VPArray[T]: ...

    @property
    def metric(self) -> Callable[[T, T], float]: ...

    @property
    def leaf_capacity(self) -> int: ...

    @property
    def leaf_radius(self) -> float: ...

    @property
    def pivoting(self) -> PivotingStrategy: ...


class Tree(Protocol[T]):

    @property
    def is_terminal(self) -> bool: ...

    @property
    def bounds(self) -> Optional[tuple[int, int]]: ...

    @property
    def ball(self) -> Optional[tuple[float, T]]: ...

    @property
    def left(self) -> Optional[Tree[T]]: ...

    @property
    def right(self) -> Optional[Tree[T]]: ...


class Node(Generic[T]):

    def __init__(self, radius: float, center: T, left: Tree[T], right: Tree[T]):
        self._radius = radius
        self._center = center
        self._left = left
        self._right = right

    @property
    def bounds(self) -> Optional[tuple[int, int]]:
        return None

    @property
    def ball(self) -> Optional[tuple[float, T]]:
        return self._radius, self._center

    @property
    def is_terminal(self) -> bool:
        return False

    @property
    def left(self) -> Optional[Tree[T]]:
        return self._left

    @property
    def right(self) -> Optional[Tree[T]]:
        return self._right


class Leaf(Generic[T]):

    def __init__(self, start: int, end: int):
        self._start = start
        self._end = end

    @property
    def bounds(self) -> Optional[tuple[int, int]]:
        return self._start, self._end

    @property
    def is_terminal(self) -> bool:
        return True

    @property
    def ball(self) -> Optional[tuple[float, T]]:
        return None

    @property
    def left(self) -> Optional[Tree[T]]:
        return None

    @property
    def right(self) -> Optional[Tree[T]]:
        return None
