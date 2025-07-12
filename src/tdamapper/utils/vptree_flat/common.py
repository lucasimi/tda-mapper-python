from __future__ import annotations

from typing import Callable, Generic, Iterator, Literal, Protocol, TypeVar

import numpy as np
from numpy.typing import NDArray

from tdamapper._common import Array
from tdamapper.utils.quickselect import quickselect, swap_all


def _mid(start: int, end: int) -> int:
    return (start + end) // 2


PivotingStrategy = Literal["disabled", "random", "furthest"]

T = TypeVar("T")


class VPArray(Generic[T]):

    _dataset: Array[T]
    _distances: NDArray[np.float_]
    _indices: NDArray[np.int_]
    _is_terminal: NDArray[np.bool_]

    def __init__(
        self,
        dataset: Array[T],
        distances: NDArray[np.float_],
        indices: NDArray[np.int_],
        is_terminal: NDArray[np.bool_],
    ):
        self._dataset = dataset
        self._distances = distances
        self._indices = indices
        self._is_terminal = is_terminal

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

    def set_terminal(self, i: int, terminal: bool) -> None:
        self._is_terminal[i] = terminal

    def is_terminal(self, i: int) -> bool:
        return self._is_terminal[i]

    def swap(self, i: int, j: int) -> None:
        swap_all(self._distances, i, j, self._indices, self._is_terminal)

    def partition(self, s: int, e: int, k: int) -> None:
        quickselect(self._distances, s, e, k, self._indices, self._is_terminal)


class VPTreeType(Protocol[T]):

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
