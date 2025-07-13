from __future__ import annotations

from typing import Iterator, Optional, Protocol, TypeVar

T_contra = TypeVar("T_contra", contravariant=True)

T_co = TypeVar("T_co", covariant=True)

T = TypeVar("T")


class ArrayRead(Protocol[T_co]):
    """
    Abstract interface for a read-only array-like structure.
    """

    def __getitem__(self, index: int) -> T_co:
        """
        Get an item from the array.
        """

    def __len__(self) -> int:
        """
        Get the length of the array.
        """

    def __iter__(self) -> Iterator[T_co]:
        """
        Iterate over the array.
        """


class ArrayWrite(Protocol[T_contra]):
    """
    Abstract interface for a writeable array-like structure.
    """

    def __setitem__(self, index: int, value: T_contra) -> None:
        """
        Set an item in the array.
        """


class Array(ArrayRead[T], ArrayWrite[T], Protocol[T]):
    """
    Abstract interface for an array-like structure.
    """


class Metric(Protocol[T_contra]):
    """
    Abstract interface for a metric.
    """

    def __call__(self, x: T_contra, y: T_contra) -> float: ...


class Cover(Protocol[T_contra]):
    """
    Abstract interface for cover algorithms.

    This is a naive implementation. Subclasses should override the methods of
    this class to implement more meaningful cover algorithms.
    """

    def apply(self, X: ArrayRead[T_contra]) -> Iterator[list[int]]:
        """
        Covers the dataset with a single open set.

        This is a naive implementation that returns a generator producing a
        single list containing all the ids if the original dataset. This
        method should be overridden by subclasses to implement more meaningful
        cover algorithms.

        :param X: A dataset of n points.
        :return: A generator of lists of ids.
        """


class Clustering(Protocol[T_contra]):
    """
    Abstract interface for clustering algorithms.

    A clustering algorithm is a method for grouping data points into clusters.
    Each cluster is represented by a unique integer label, and the labels are
    assigned to the points in the dataset. The labels are typically non-negative
    integers, starting from zero. The labels are assigned such that the points
    in the same cluster have the same label, and the points in different clusters
    have different labels. The labels are not necessarily contiguous, and there
    may be gaps in the sequence of labels.
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
    Abstract interface for search algorithms.

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
