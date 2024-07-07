"""
Open cover construction for the Mapper algorithm.

An open cover is a collection of open subsets of a dataset whose union spans the
whole dataset. Unlike clustering, open subsets do not need to be disjoint.
Indeed, the overlaps of the open subsets define the edges of the Mapper graph.
"""

from tdamapper.proximity import (
    proximity_net,
    BallProximity,
    KNNProximity,
    CubicalProximity,
    TrivialProximity)


class Cover:
    """
    Abstract interface for cover algorithms.

    This is a naive implementation. Subclasses should override the methods of
    this class to implement more meaningful cover algorithms.
    """

    def apply(self, X):
        """
        Covers the dataset with a single open set.

        This is a naive implementation that should be overridden by subclasses
        to implement more meaningful cover algorithms.

        :param X: A dataset of n points to be covered with open subsets.
        :type X: array-like of shape (n, m) or list-like of length n
        :return: A generator that produces a single list of ints whose elements
            are the indices of the data points, ranging from 0 to n - 1.
        :rtype: generator of lists of ints
        """
        yield list(range(0, len(X)))


class ProximityCover(Cover):
    """
    Cover algorithm based on proximity-net.

    This class creates an open cover by calling the function
    :func:`tdamapper.proximity.proximity_net`.

    :param proximity: The proximity function to use.
    :type proximity: Anything compatible with class
        :class:`tdamapper.proximity.Proximity`
    """

    def __init__(self, proximity):
        self.__proximity = proximity

    def apply(self, X):
        """
        Covers the dataset using proximity-net on the specified proximity
        function.

        The proximity function is used to create an open set whenever a point is
        picked from :func:`tdamapper.proximity.proximity_net`.

        :param X: A dataset of n points to be covered with open subsets.
        :type X: array-like of shape (n, m) or list-like of length n
        :return: A generator yielding lists of ints whose elements are the
            indices of the data points.
        :rtype: generator of lists of ints
        """
        return proximity_net(X, self.__proximity)


class BallCover(ProximityCover):
    """
    Cover algorithm based on `ball proximity function` implemented as
    :class:`tdamapper.proximity.BallProximity`.

    :param radius: The radius of the open balls, must be positive.
    :type radius: float
    :param metric: The (pseudo-)metric function that defines the distance
        between points, must be symmetric, positive, and satisfy the
        triangle-inequality, i.e.
        :math:`metric(x, z) \leq metric(x, y) + metric(y, z)` for every x, y, z
        in the dataset.
    :type metric: Callable
    :param flat: A flag that indicates whether to use a flat or a hierarchical
        vantage point tree, defaults to True.
    :type flat: bool, optional
    """

    def __init__(
        self,
        radius,
        metric='euclidean',
        metric_params=None,
        kind='flat',
        leaf_capacity=1,
        leaf_radius=None,
        pivoting=None,
    ):
        prox = BallProximity(
            radius=radius,
            metric=metric,
            metric_params=metric_params,
            kind=kind,
            leaf_capacity=leaf_capacity,
            leaf_radius=leaf_radius,
            pivoting=pivoting,
        )
        super().__init__(proximity=prox)


class KNNCover(ProximityCover):
    """
    Cover algorithm based on `knn proximity function` implemented as
    :class:`tdamapper.proximity.KNNProximity`.

    :param neighbors: The number of neighbors to use for the KNN Proximity
        function, must be positive and less than the length of the dataset.
    :type neighbors: int
    :param metric: The (pseudo-)metric function that defines the distance
        between points, must be symmetric, positive, and satisfy the
        triangle-inequality, i.e.
        :math:`metric(x, z) \leq metric(x, y) + metric(y, z)` for every x, y, z
        in the dataset.
    :type metric: Callable
    :param flat: A flag that indicates whether to use a flat or a hierarchical
        vantage point tree, defaults to False.
    :type flat: bool, optional
    """

    def __init__(
        self,
        neighbors,
        metric='euclidean',
        metric_params=None,
        kind='flat',
        leaf_capacity=None,
        leaf_radius=0.0,
        pivoting=None,
    ):
        prox = KNNProximity(
            neighbors=neighbors,
            metric=metric,
            metric_params=metric_params,
            kind=kind,
            leaf_capacity=leaf_capacity,
            leaf_radius=leaf_radius,
            pivoting=pivoting,
        )
        super().__init__(proximity=prox)


class CubicalCover(ProximityCover):
    """
    Cover algorithm based on the `cubical proximity function` implemented as
    :class:`tdamapper.proximity.CubicalProximity`.

    :param n_intervals: The number of intervals to use for each dimension, must
        be positive and less than or equal to the length of the dataset.
    :type n_intervals: int
    :param overlap_frac: The fraction of overlap between adjacent intervals on
        each dimension, must be in the range (0.0, 1.0).
    :type overlap_frac: float
    :param flat: A flag that indicates whether to use a flat or a hierarchical
        vantage point tree, defaults to False.
    :type flat: bool, optional
    """

    def __init__(
        self,
        n_intervals,
        overlap_frac,
        kind='flat',
        leaf_capacity=1,
        leaf_radius=None,
        pivoting=None,
    ):
        prox = CubicalProximity(
            n_intervals=n_intervals, 
            overlap_frac=overlap_frac,
            kind=kind,
            leaf_capacity=leaf_capacity,
            leaf_radius=leaf_radius,
            pivoting=pivoting,
        )
        super().__init__(proximity=prox)


class TrivialCover(ProximityCover):
    """
    Cover algorithm based on the `trivial proximity function` implemented as
    :class:`tdamapper.proximity.TrivialProximity`.
    """

    def __init__(self):
        super().__init__(proximity=TrivialProximity())
