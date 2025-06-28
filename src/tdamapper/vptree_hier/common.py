from __future__ import annotations

from typing import (
    Callable,
    Generic,
    Iterable,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
from numpy.typing import NDArray

from tdamapper.quickselect import quickselect, swap_all

T = TypeVar("T")


def _mid(start: int, end: int) -> int:
    return (start + end) // 2


class VPTreeType(Protocol, Generic[T]):

    @property
    def leaf_capacity(self) -> int: ...

    @property
    def leaf_radius(self) -> float: ...

    @property
    def pivoting(self) -> Optional[str]: ...

    @property
    def distance(self) -> Callable[[T, T], float]: ...

    @property
    def tree(self) -> Tree[T]: ...

    @property
    def array(self) -> VPArray[T]: ...


class VPArray(Generic[T]):

    def __init__(
        self,
        dataset: List[T],
        distances: NDArray[np.float64],
        indices: NDArray[np.int64],
    ):
        self._dataset = dataset
        self._distances = distances
        self._indices = indices

    def size(self) -> int:
        return len(self._dataset)

    def get_point(self, i: int) -> T:
        return self._dataset[self._indices[i]]

    def get_points(self, s: int, e: int) -> Iterable[T]:
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


class Node(Generic[T]):

    def __init__(self, radius: float, center: T, left: Tree, right: Tree):
        self._radius = radius
        self._center = center
        self._left = left
        self._right = right

    def get_ball(self) -> Tuple[float, T]:
        return self._radius, self._center

    def is_terminal(self) -> bool:
        return False

    def get_left(self) -> Tree:
        return self._left

    def get_right(self) -> Tree:
        return self._right


class Leaf:

    def __init__(self, start: int, end: int):
        self._start = start
        self._end = end

    def get_bounds(self) -> Tuple[int, int]:
        return self._start, self._end

    def is_terminal(self) -> bool:
        return True


Tree = Union[Node[T], Leaf]
