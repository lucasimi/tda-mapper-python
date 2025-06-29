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

    @property
    def tree(self) -> Tree[T]:
        """
        The tree structure of the VP-tree.

        :return: An instance of Tree representing the VP-tree structure.
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
    """

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

    def swap(self, i: int, j: int) -> None:
        """
        Swap the points at indices i and j in the dataset, along with their
        distances and terminal status.

        :param i: The index of the first point to swap.
        :param j: The index of the second point to swap.
        """
        swap_all(self._distances, i, j, self._indices)

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
        quickselect(self._distances, s, e, k, self._indices)


class Node(Generic[T]):
    """
    Node class for representing a node in the VP-tree.

    This class represents a node in the VP-tree, which contains a pivot point,
    a radius, and references to left and right subtrees. The node can be used
    to perform ball searches or k-nearest neighbor searches in the VP-tree.

    :param radius: The radius of the ball centered at the pivot point.
    :param center: The pivot point of type T.
    :param left: The left subtree of type Tree.
    :param right: The right subtree of type Tree.
    """

    def __init__(self, radius: float, center: T, left: Tree, right: Tree):
        self._radius = radius
        self._center = center
        self._left = left
        self._right = right

    def get_ball(self) -> Tuple[float, T]:
        """
        Get the radius and center of the ball represented by this node.

        This method returns the radius of the ball and the pivot point (center)

        :return: A tuple containing the radius (float) and the center (T) of the ball.
        """
        return self._radius, self._center

    def is_terminal(self) -> bool:
        """
        Check if this node is a terminal node (leaf).

        A terminal node is a leaf node in the VP-tree, which means it does not
        have any children (left or right subtrees).

        :return: True if this node is a terminal node, False otherwise.
        """
        return False

    def get_left(self) -> Tree:
        """
        Get the left subtree of this node.

        The left subtree contains points that are closer to the pivot point
        than the radius of the ball.

        :return: The left subtree of type Tree.
        """
        return self._left

    def get_right(self) -> Tree:
        """
        Get the right subtree of this node.

        The right subtree contains points that are farther from the pivot point
        than the radius of the ball.

        :return: The right subtree of type Tree.
        """
        return self._right

    def get_bounds(self) -> Tuple[int, int]:
        """
        Get the bounds of the range of points in this node.

        Since this is a non-terminal node, it does not have bounds like a leaf
        node does. This method raises an exception to indicate that bounds are
        not applicable for this node type.

        :return: Raises an exception since this node does not have bounds.
        """
        raise NotImplementedError(
            "Node does not have bounds. Use Leaf class for leaf nodes."
        )


class Leaf:
    """
    Leaf class for representing a leaf node in the VP-tree.

    This class represents a leaf node in the VP-tree, which contains a range
    of indices representing the points in the dataset that belong to this leaf.

    :param start: The starting index of the range of points in the dataset.
    :param end: The ending index of the range of points in the dataset.
    """

    def __init__(self, start: int, end: int):
        self._start = start
        self._end = end

    def get_bounds(self) -> Tuple[int, int]:
        """
        Get the bounds of the range of points in this leaf.

        This method returns the starting and ending indices of the range of
        points in the dataset that belong to this leaf.

        :return: A tuple containing the starting index (inclusive) and the
            ending index (exclusive) of the range of points in the dataset.
        """
        return self._start, self._end

    def is_terminal(self) -> bool:
        """
        Check if this node is a terminal node (leaf).

        A terminal node is a leaf node in the VP-tree, which means it does not
        have any children (left or right subtrees).

        :return: True if this node is a terminal node, False otherwise.
        """
        return True

    def get_left(self) -> Tree:
        """
        Get the left subtree of this node.

        Since this is a leaf node, it does not have a left subtree.

        :return: Raises an exception since a leaf node does not have a left subtree.
        """
        raise NotImplementedError("Leaf nodes do not have left subtrees.")

    def get_right(self) -> Tree:
        """
        Get the right subtree of this node.

        Since this is a leaf node, it does not have a right subtree.

        :return: Raises an exception since a leaf node does not have a right subtree.
        """
        raise NotImplementedError("Leaf nodes do not have right subtrees.")

    def get_ball(self) -> Tuple[float, T]:
        """
        Get the radius and center of the ball represented by this leaf.

        Since this is a leaf node, it does not represent a ball, so this method
        raises an exception.

        :return: Raises an exception since a leaf node does not have a ball.
        """
        raise NotImplementedError("Leaf nodes do not represent balls.")


Tree = Union[Node[T], Leaf]
