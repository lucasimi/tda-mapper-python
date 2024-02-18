"""
Open cover construction for the Mapper algorithm.

An open cover is a collection of subsets of a data set such that the union of
the subsets contains the whole data set.

The Mapper algorithm consists of three main steps: filtering, covering, and
clustering. First, the data points are mapped to a lower dimensional space using
a lens function. Then, the lens space is covered by overlapping open sets, using
an open cover algorithm. Finally, the data points in each open set are clustered
using a clustering algorithm, and the clusters are connected by edges if they
share points in the overlap.

The open cover construction is a key step in the Mapper algorithm that
partitions the data into overlapping subsets based on the values of a lens
function.
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
        :type X: array-like of shape (n, m) or list-like of size n
        :return: A generator yielding a single list ranging from zero to the
            length of the dataset.
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
        :type X: array-like of shape (n, m) or list-like of size n
        :return: A generator yielding a single list ranging from zero to the
            length of the dataset.
        :rtype: generator of lists of ints
        """
        return proximity_net(X, self.__proximity)


class BallCover(ProximityCover):
    """
    Cover algorithm based on :class:`tdamapper.proximity.BallProximity`.

    :param radius: The radius of the open balls, must be positive.
    :type radius: float
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

    def __init__(self, radius, metric, flat=True):
        prox = BallProximity(radius=radius, metric=metric, flat=flat)
        super().__init__(proximity=prox)


class KNNCover(ProximityCover):
    """
    Cover algorithm based on :class:`tdamapper.proximity.KNNProximity`.

    :param neighbors: The number of neighbors to use for the KNN Proximity
        function, must be positive and less than the size of the data set.
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

    def __init__(self, neighbors, metric, flat=True):
        prox = KNNProximity(neighbors=neighbors, metric=metric, flat=flat)
        super().__init__(proximity=prox)



class CubicalCover(ProximityCover):
    """
    Cover algorithm based on :class:`tdamapper.proximity.CubicalProximity`.

    :param n_intervals: The number of intervals to use for each dimension, must
        be positive and less than or equal to the size of the data set.
    :type n_intervals: int
    :param overlap_frac: The fraction of overlap between adjacent intervals on
        each dimension, must be in the range (0.0, 1.0).
    :type overlap_frac: float
    :param flat: A flag that indicates whether to use a flat or a hierarchical
        vantage point tree, defaults to False.
    :type flat: bool, optional
    """

    def __init__(self, n_intervals, overlap_frac, flat=True):
        prox = CubicalProximity(
            n_intervals=n_intervals, 
            overlap_frac=overlap_frac,
            flat=flat)
        super().__init__(proximity=prox)



class TrivialCover(ProximityCover):
    """
    Cover algorithm based on :class:`tdamapper.proximity.TrivialProximity`.
    """

    def __init__(self):
        super().__init__(proximity=TrivialProximity())
