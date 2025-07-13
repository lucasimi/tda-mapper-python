from __future__ import annotations

from typing import Iterator, Optional, Protocol, TypeVar

T_contra = TypeVar("T_contra", contravariant=True)
T_co = TypeVar("T_co", covariant=True)
T = TypeVar("T")


class ArrayRead(Protocol[T_co]):
    """
    Protocol for a read-only array-like structure.
    """

    def __getitem__(self, index: int) -> T_co:
        """
        Get an item from the array.

        :param index: The index of the item to retrieve.
        :return: The item at the specified index.
        """

    def __len__(self) -> int:
        """
        Get the length of the array.

        :return: The number of items in the array.
        """

    def __iter__(self) -> Iterator[T_co]:
        """
        Iterate over the array.

        :return: An iterator over the items in the array.
        """


class ArrayWrite(Protocol[T_contra]):
    """
    Protocol for a writeable array-like structure.
    """

    def __setitem__(self, index: int, value: T_contra) -> None:
        """
        Set an item in the array.

        :param index: The index at which to set the item.
        :param value: The value to set at the specified index.
        """


class Array(ArrayRead[T], ArrayWrite[T], Protocol[T]):
    """
    Protocol for an array-like structure.
    """


class Metric(Protocol[T_contra]):
    """
    Protocol for a metric function.
    """

    def __call__(self, x: T_contra, y: T_contra) -> float:
        """
        Compute the distance between two points.

        :param x: The first point.
        :param y: The second point.
        :return: The distance between the two points.
        """


class Cover(Protocol[T_contra]):
    """
    Protocol for cover algorithms.

    A cover algorithm collects open sets from a dataset such that each point
    belongs to at least one open set. The open sets are represented as lists of
    indices, where each index corresponds to a point in the dataset. The open
    sets are eventually overlapping.
    """

    def apply(self, X: ArrayRead[T_contra]) -> Iterator[list[int]]:
        """
        Covers the dataset with open sets.

        :param X: A dataset of n points.
        :return: A generator of lists of ids.
        """


class Clustering(Protocol[T_contra]):
    """
    Protocol for clustering algorithms.

    A clustering algorithm groups data points into clusters, each represented by
    an integer label. Labels are typically non-negative and may be
    non-contiguous.
    """

    labels_: list[int]

    def fit(
        self, X: ArrayRead[T_contra], y: Optional[ArrayRead[T_contra]] = None
    ) -> Clustering[T_contra]:
        """
        Fit the clustering algorithm to the data.

        :param X: A dataset of n points.
        :param y: A dataset of targets. Typically ignored and present for
            compatibility with scikit-learn's clustering interface.
        :return: The fitted clustering object.
        """


class SpatialSearch(Protocol[T_contra]):
    """
    Protocol for spatial search algorithms.

    A spatial search algorithm is a method for finding neighbors of a
    query point in a dataset.
    """

    def fit(self, X: ArrayRead[T_contra]) -> SpatialSearch[T_contra]:
        """
        Train internal parameters.

        :param X: A dataset of n points.
        :return: The object itself.
        """

    def search(self, x: T_contra) -> list[int]:
        """
        Return a list of neighbors for the query point.

        :param x: A query point for which we want to find neighbors.
        :return: A list containing all the indices of the points in the
            dataset.
        """
