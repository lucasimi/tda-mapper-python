"""
Common types and structures for VP-tree implementation.

This module defines the basic types and structures used in the VP-tree
implementation, including the VPArray for managing the dataset and distances,
the Node and Leaf classes for representing the tree structure, and the
VPTreeType protocol for type checking.
"""

from __future__ import annotations

from typing import (
    Callable,
    Generic,
    Iterable,
    List,
    Optional,
    Protocol,
    TypeVar,
)

import numpy as np
from numpy.typing import NDArray

from tdamapper.quickselect import quickselect, swap_all

T = TypeVar("T")


def _mid(start: int, end: int) -> int:
    return (start + end) // 2


class VPTreeType(Protocol, Generic[T]):
    """
    Protocol for a VP-tree type.
    """

    @property
    def leaf_capacity(self) -> int:
        """
        The maximum number of points in a leaf node.

        :return: An integer representing the maximum number of points in a leaf node.
        """

    @property
    def leaf_radius(self) -> float:
        """
        The maximum radius of a leaf node.

        :return: A float representing the maximum radius of a leaf node.
        """

    @property
    def pivoting(self) -> Optional[str]:
        """
        The pivoting strategy used in the VP-tree.
        Can be "random", "furthest", or None for no pivoting.

        :return: A string indicating the pivoting strategy or None.
        """

    @property
    def array(self) -> VPArray[T]:
        """
        The VPArray instance containing the dataset and distances.

        :return: An instance of VPArray containing the dataset and distances.
        """

    @property
    def distance(self) -> Callable[[T, T], float]:
        """
        The distance function used to compute distances between points.
        It should take two points of type T and return a float distance.

        :return: A callable that takes two points of type T and returns a float distance.
        """


class VPArray(Generic[T]):
    """
    VPArray class for managing a collection of points in a VP-tree.

    This class provides methods to access points, distances, and terminal
    status of points in a VP-tree. It allows for efficient retrieval and
    manipulation of points based on their indices and distances. It also
    supports partitioning and swapping of points based on distances. The class
    is initialized with a dataset, distances, indices, and terminal status of
    points.

    :param dataset: A list of points of type T.
    :param distances: A NumPy array of distances corresponding to the points.
    :param indices: A NumPy array of indices mapping points to their positions
        in the dataset.
    :param is_terminal: A NumPy array indicating whether each point is terminal
        (True) or not (False). A terminal point is a leaf node in the VP-tree.
        It is used to determine if a point is a leaf node in the VP-tree.
    """

    def __init__(
        self,
        dataset: List[T],
        distances: NDArray[np.float64],
        indices: NDArray[np.float64],
        is_terminal,
    ):
        self._dataset = dataset
        self._distances = distances
        self._indices = indices
        self._is_terminal = is_terminal

    def size(self) -> int:
        """
        Get the number of points in the dataset.

        :return: The number of points in the dataset.
        """
        return len(self._dataset)

    def get_point(self, i: int) -> T:
        """
        Get the point at index i in the dataset.

        :param i: The index of the point to retrieve.
        :return: The point at index i in the dataset.
        """
        return self._dataset[self._indices[i]]

    def get_points(self, s: int, e: int) -> Iterable[T]:
        """
        Get a slice of points from the dataset.

        :param s: The start index of the slice (inclusive).
        :param e: The end index of the slice (exclusive).
        :return: An iterable of points from the dataset in the specified
            range.
        """
        for x_index in self._indices[s:e]:
            yield self._dataset[x_index]

    def get_distance(self, i: int) -> float:
        """
        Get the distance of the point at index i.

        :param i: The index of the point whose distance is to be retrieved.
        :return: The distance of the point at index i.
        """
        return self._distances[i]

    def set_distance(self, i: int, dist: float) -> None:
        """
        Set the distance of the point at index i.

        :param i: The index of the point whose distance is to be set.
        :param dist: The distance to set for the point at index i.
        """
        self._distances[i] = dist

    def set_terminal(self, i: int, terminal: bool) -> None:
        """
        Set the terminal status of the point at index i.

        :param i: The index of the point whose terminal status is to be set.
        :param terminal: A boolean indicating whether the point is terminal
            (True) or not (False).
        """
        self._is_terminal[i] = terminal

    def is_terminal(self, i: int) -> bool:
        """
        Check if the point at index i is terminal, i.e. a leaf node in the
        VP-tree.

        :param i: The index of the point to check.
        :return: True if the point at index i is terminal, False otherwise.
        """
        return self._is_terminal[i]

    def swap(self, i: int, j: int) -> None:
        """
        Swap the points at indices i and j in the dataset, along with their
        distances and terminal status.

        :param i: The index of the first point to swap.
        :param j: The index of the second point to swap.
        """
        swap_all(self._distances, i, j, self._indices, self._is_terminal)

    def partition(self, s: int, e: int, k: int) -> None:
        """
        Partition the points in the range [s, e) such that the k-th smallest
        distance is at index k, and all points before k have distances less
        than or equal to the distance at index k, and all points after k have
        distances greater than or equal to the distance at index k.

        :param s: The start index of the range (inclusive).
        :param e: The end index of the range (exclusive).
        :param k: The index of the k-th smallest distance to partition around.
        """
        quickselect(self._distances, s, e, k, self._indices, self._is_terminal)
