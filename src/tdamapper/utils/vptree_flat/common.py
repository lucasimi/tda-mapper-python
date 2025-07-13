"""
This module provides common utilities and types for the vp-tree implementation.
"""

from __future__ import annotations

from typing import Callable, Generic, Iterator, Literal, Protocol, TypeVar

import numpy as np
from numpy.typing import NDArray

from tdamapper.protocols import Array
from tdamapper.utils.quickselect import quickselect, swap_all


def _mid(start: int, end: int) -> int:
    return (start + end) // 2


PivotingStrategy = Literal["disabled", "random", "furthest"]

T = TypeVar("T")


class VPArray(Generic[T]):
    """
    A class representing an array of points with their distances and indices.
    This is used in the vp-tree for efficient distance calculations and indexing.

    :param dataset: The dataset containing the points.
    :param distances: An array of distances corresponding to the points.
    :param indices: An array of indices mapping the points to their original positions.
    :param is_terminal: An array indicating whether each point is a terminal node.
    """

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
        """
        Get the size of the VPArray.

        :return: The number of points in the VPArray.
        """
        return len(self._dataset)

    def get_point(self, i: int) -> T:
        """
        Get the point at the specified index.

        :param i: The index of the point to retrieve.
        :return: The point at the specified index.
        """
        return self._dataset[self._indices[i]]

    def get_points(self, s: int, e: int) -> Iterator[T]:
        """
        Get points in the specified range.

        :param s: The start index of the range.
        :param e: The end index of the range.
        :return: An iterator over the points in the specified range.
        """
        for x_index in self._indices[s:e]:
            yield self._dataset[x_index]

    def get_distance(self, i: int) -> float:
        """
        Get the distance of the point at the specified index.

        :param i: The index of the point whose distance is to be retrieved.
        :return: The distance of the point at the specified index.
        """
        return self._distances[i]

    def set_distance(self, i: int, dist: float) -> None:
        """
        Set the distance of the point at the specified index.

        :param i: The index of the point whose distance is to be set.
        :param dist: The distance to set for the point at the specified index.
        """
        self._distances[i] = dist

    def set_terminal(self, i: int, terminal: bool) -> None:
        """
        Set whether the point at the specified index is a terminal node.

        :param i: The index of the point to set.
        :param terminal: True if the point is a terminal node, False otherwise.
        """
        self._is_terminal[i] = terminal

    def is_terminal(self, i: int) -> bool:
        """
        Check if the point at the specified index is a terminal node.

        :param i: The index of the point to check.
        :return: True if the point is a terminal node, False otherwise.
        """
        return self._is_terminal[i]

    def swap(self, i: int, j: int) -> None:
        """
        Swap the points and their distances at indices i and j.

        :param i: The first index to swap.
        :param j: The second index to swap.
        """
        swap_all(self._distances, i, j, self._indices, self._is_terminal)

    def partition(self, s: int, e: int, k: int) -> None:
        """
        Partition the array such that the k-th smallest distance is at index k.

        :param s: The start index of the range to partition.
        :param e: The end index of the range to partition.
        :param k: The index at which the k-th smallest distance should be placed.
        """
        quickselect(self._distances, s, e, k, self._indices, self._is_terminal)


class VPTreeType(Protocol[T]):
    """
    A protocol defining the structure of a vp-tree.
    """

    @property
    def array(self) -> VPArray[T]:
        """
        Get the VPArray associated with the vp-tree.
        """

    @property
    def metric(self) -> Callable[[T, T], float]:
        """
        Get the metric used for distance calculations in the vp-tree.
        """

    @property
    def leaf_capacity(self) -> int:
        """
        Get the maximum number of points in a leaf node of the vp-tree.
        """

    @property
    def leaf_radius(self) -> float:
        """
        Get the radius of the leaf nodes in the vp-tree.
        """

    @property
    def pivoting(self) -> PivotingStrategy:
        """
        Get the pivoting strategy used in the vp-tree.
        """
