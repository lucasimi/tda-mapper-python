"""
This module provides common utilities and types for the vp-tree implementation.
"""

from __future__ import annotations

from typing import Callable, Generic, Iterator, Literal, Optional, Protocol, TypeVar

import numpy as np
from numpy.typing import NDArray

from tdamapper.protocols import Array
from tdamapper.utils.quickselect import quickselect, swap_all


def _mid(start: int, end: int) -> int:
    return (start + end) // 2


T = TypeVar("T")

PivotingStrategy = Literal["disabled", "random", "furthest"]


class VPArray(Generic[T]):
    """
    A class representing an array of points with their distances and indices.
    This is used in the vp-tree for efficient distance calculations and indexing.

    :param dataset: The dataset containing the points.
    :param distances: An array of distances corresponding to the points.
    :param indices: An array of indices mapping the points to their original positions.
    """

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

    def swap(self, i: int, j: int) -> None:
        """
        Swap the points and their distances at indices i and j.

        :param i: The first index to swap.
        :param j: The second index to swap.
        """
        swap_all(self._distances, i, j, self._indices)

    def partition(self, s: int, e: int, k: int) -> None:
        """
        Partition the array such that the k-th smallest distance is at index k.

        :param s: The start index of the range to partition.
        :param e: The end index of the range to partition.
        :param k: The index at which the k-th smallest distance should be placed.
        """
        quickselect(self._distances, s, e, k, self._indices)


class VPTreeType(Protocol[T]):
    """
    A protocol defining the structure of a vp-tree.
    """

    @property
    def tree(self) -> Tree[T]:
        """
        Get the tree structure of the vp-tree.
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


class Tree(Protocol[T]):
    """
    A protocol defining the structure of a node in the vp-tree.
    This can be either a leaf node or an internal node.
    """

    @property
    def is_terminal(self) -> bool:
        """
        Check if the node is a terminal (leaf) node.

        :return: True if the node is a leaf, False otherwise.
        """

    @property
    def bounds(self) -> Optional[tuple[int, int]]:
        """
        Get the bounds of the node if it is a leaf.

        :return: A tuple containing the start and end indices of the points in
            the leaf, or None if not a leaf.
        """

    @property
    def ball(self) -> Optional[tuple[float, T]]:
        """
        Get the ball (radius and center point) of the node if it is an internal
        node.

        :return: A tuple containing the radius and the center point, or None if
            not an internal node.
        """

    @property
    def left(self) -> Optional[Tree[T]]:
        """
        Get the left child of the node.

        :return: The left child node, or None if it does not exist.
        """

    @property
    def right(self) -> Optional[Tree[T]]:
        """
        Get the right child of the node.

        :return: The right child node, or None if it does not exist.
        """


class Node(Generic[T]):
    """
    A class representing an internal node in the vp-tree.

    :param radius: The radius of the ball centered at this node.
    :param center: The center point of the ball.
    :param left: The left child of the node.
    :param right: The right child of the node.
    """

    def __init__(self, radius: float, center: T, left: Tree[T], right: Tree[T]):
        self._radius = radius
        self._center = center
        self._left = left
        self._right = right

    @property
    def bounds(self) -> Optional[tuple[int, int]]:
        """
        Get the bounds of the node if it is a leaf.

        :return: None, as this is not a leaf node.
        """
        return None

    @property
    def ball(self) -> Optional[tuple[float, T]]:
        """
        Get the ball (radius and center point) of the node.

        :return: A tuple containing the radius and the center point.
        """
        return self._radius, self._center

    @property
    def is_terminal(self) -> bool:
        """
        Check if the node is a terminal (leaf) node.

        :return: False, as this is an internal node.
        """
        return False

    @property
    def left(self) -> Optional[Tree[T]]:
        """
        Get the left child of the node.

        :return: The left child node.
        """
        return self._left

    @property
    def right(self) -> Optional[Tree[T]]:
        """
        Get the right child of the node.

        :return: The right child node.
        """
        return self._right


class Leaf(Generic[T]):
    """
    A class representing a leaf node in the vp-tree.

    :param start: The start index of the points in this leaf.
    :param end: The end index of the points in this leaf.
    """

    def __init__(self, start: int, end: int):
        self._start = start
        self._end = end

    @property
    def bounds(self) -> Optional[tuple[int, int]]:
        """
        Get the bounds of the leaf node.

        :return: A tuple containing the start and end indices of the points in
            the leaf.
        """
        return self._start, self._end

    @property
    def is_terminal(self) -> bool:
        """
        Check if the node is a terminal (leaf) node.

        :return: True, as this is a leaf node.
        """
        return True

    @property
    def ball(self) -> Optional[tuple[float, T]]:
        """
        Get the ball (radius and center point) of the node if it is an internal
        node.

        :return: None, as this is a leaf node.
        """
        return None

    @property
    def left(self) -> Optional[Tree[T]]:
        """
        Get the left child of the node.

        :return: None, as leaf nodes do not have children.
        """
        return None

    @property
    def right(self) -> Optional[Tree[T]]:
        """
        Get the right child of the node.

        :return: None, as leaf nodes do not have children.
        """
        return None
