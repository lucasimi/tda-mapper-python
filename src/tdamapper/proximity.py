"""
Proximity functions for open cover algorithms.

A proximity function is a function that maps each point into a subset of the
dataset that contains the point itself.

Proximity functions, implemented as subclasses of class
:class:`tdamapper.proximity.Proximity`, are a convenient way to implement open
cover algorithms by using the `proximity_net` construction. Proximity-net is
implemented by function :func:`tdamapper.proximity.proximity_net`, and used by
the class :class:`tdamapper.cover.ProximityCover`.
"""

import numpy as np

from tdamapper.utils.vptree_flat import VPTree as FVPT
from tdamapper.utils.vptree import VPTree as VPT


def proximity_net(X, proximity):
    """
    Compute proximity-net for a given proximity function.

    This function uses an iterative algorithm to construct the proximity-net. It
    starts with an arbitrary point and builds an open cover around it based on
    the proximity function. Then it discards the covered points and repeats the
    process on the remaining points until all points are covered.

    This function applies an iterative algorithm to create the proximity-net. It
    picks an arbitrary point and forms an open cover calling the proximity
    function on the chosen point. The points contained in the open cover are
    then marked as covered, and discarded in the following steps. The procedure
    is repeated on the leftover points until every point is eventually covered.

    This function returns a generator that yields each element of the
    proximity-net as a list of ids. The ids are the indices of the points in the
    original dataset.

    :param X: A dataset of n points.
    :type X: array-like of shape (n, m) or list-like of length n
    :param proximity: A proximity function
    :type proximity: :class:`tdamapper.cover.Proximity`
    :return: A generator of lists of ids.
    :rtype: generator of lists of ints
    """
    covered_ids = set()
    proximity.fit(X)
    for i, xi in enumerate(X):
        if i not in covered_ids:
            neigh_ids = proximity.search(xi)
            covered_ids.update(neigh_ids)
            if neigh_ids:
                yield neigh_ids


def _pullback(fun, dist):
    return lambda x, y: dist(fun(x), fun(y))


def _rho(x):
    return np.floor(x) + 0.5


def _l_infty(x, y):
    # in alternative: np.linalg.norm(x - y, ord=np.inf)
    return np.max(np.abs(x - y))


class Proximity:
    """
    Abstract interface for proximity functions.

    This is a naive implementation. Subclasses should override the methods of
    this class to implement more meaningful proximity functions.
    """

    def fit(self, X):
        """
        Train internal parameters.

        This is a naive implementation that should be overridden by subclasses
        to implement more meaningful proximity functions.

        :param X: A dataset of n points used to extract parameters and perform
            training.
        :type X: array-like of shape (n, m) or list-like of length n
        :return: The object itself.
        :rtype: self
        """
        self.__X = X
        return self

    def search(self, x):
        """
        Return a list of neighbors for the query point.

        This is a naive implementation that returns all the points in the
        dataset as neighbors. This method should be overridden by subclasses to
        implement more meaningful proximity functions.

        :param x: A query point for which we want to find neighbors.
        :type x: Any
        :return: A list containing all the indices of the points in the dataset.
        :rtype: list[int]
        """
        return list(range(0, len(self.__X)))


class BallProximity(Proximity):
    """
    Proximity function based on open balls of fixed radius.

    An open ball is a set of points placed within a certain distance from a
    center. This class maps each point to the open ball of fixed radius centered
    on the point itself.

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
        self.__metric = lambda x, y: metric(x[1], y[1])
        self.__radius = radius
        self.__data = None
        self.__vptree = None
        self.__flat = flat

    def __flat_vpt(self):
        return FVPT(self.__metric, self.__data, leaf_radius=self.__radius)

    def __vpt(self):
        return VPT(self.__metric, self.__data, leaf_radius=self.__radius)

    def fit(self, X):
        """
        Train internal parameters.

        This method creates a vptree on the dataset in order to perform fast
        range queries in the func:`tdamapper.proximity.BallProximity.search`
        method.

        :param X: A dataset of n points used to extract parameters and perform
            training.
        :type X: array-like of shape (n, m) or list-like of length n
        :return: The object itself.
        :rtype: self
        """
        self.__data = list(enumerate(X))
        self.__vptree = self.__flat_vpt() if self.__flat else self.__vpt()
        return self

    def search(self, x):
        """
        Return a list of neighbors for the query point.

        This method uses the internal vptree to perform fast range queries.

        :param x: A query point for which we want to find neighbors.
        :type x: Any
        :return: The indices of the neighbors contained in the dataset.
        :rtype: list[int]
        """
        if self.__vptree is None:
            return []
        neighs = self.__vptree.ball_search((-1, x), self.__radius)
        return [x for (x, _) in neighs]


class KNNProximity(Proximity):
    """
    Proximity function based on k-nearest neighbors (KNN).

    This class maps each point to the set of the k nearest neighbors to the
    point itself.

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

    def __init__(self, neighbors, metric, flat=True):
        self.__neighbors = neighbors
        self.__metric = _pullback(lambda x: x[1], metric)
        self.__data = None
        self.__vptree = None
        self.__flat = flat

    def __flat_vpt(self):
        return FVPT(self.__metric, self.__data, leaf_capacity=self.__neighbors)

    def __vpt(self):
        return VPT(self.__metric, self.__data, leaf_capacity=self.__neighbors)

    def fit(self, X):
        """
        Train internal parameters.

        This method creates a vptree on the dataset in order to perform fast
        KNN queries in the func:`tdamapper.proximity.BallProximity.search`
        method.

        :param X: A dataset of n points used to extract parameters and perform
            training.
        :type X: array-like of shape (n, m) or list-like of length n
        :return: The object itself.
        :rtype: self
        """
        self.__data = list(enumerate(X))
        self.__vptree = self.__flat_vpt() if self.__flat else self.__vpt()
        return self

    def search(self, x):
        """
        Return a list of neighbors for the query point.

        This method queries the internal vptree in order to perform fast KNN
        queries.

        :param x: A query point for which we want to find neighbors.
        :type x: Any
        :return: The indices of the neighbors contained in the dataset.
        :rtype: list[int]
        """
        if self.__vptree is None:
            return []
        neighs = self.__vptree.knn_search((-1, x), self.__neighbors)
        return [x for (x, _) in neighs]


class CubicalProximity(Proximity):
    """
    Proximity function based on open hypercubes of uniform size and overlap.

    A hypercube is a multidimensional generalization of a square or a cube.
    The size and overlap of the hypercubes are determined by the number of
    intervals and the overlap fraction parameters. This class maps each point to
    the hypercube with the nearest center.

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

    def __init__(self, n_intervals, overlap_frac, flat=True):
        self.__n_intervals = n_intervals
        self.__radius = 1.0 / (2.0 - 2.0 * overlap_frac)
        self.__minimum = None
        self.__maximum = None
        self.__delta = None
        metric = _pullback(self._gamma_n, _l_infty)
        self.__ball_proximity = BallProximity(self.__radius, metric, flat=flat)

    def _gamma_n(self, x):
        return self.__n_intervals * (x - self.__minimum) / self.__delta

    def _gamma_n_inv(self, x):
        return self.__minimum + self.__delta * x / self.__n_intervals

    def _phi(self, x):
        return self._gamma_n_inv(_rho(self._gamma_n(x)))

    def _set_bounds(self, data):
        if (data is None) or len(data) == 0:
            return
        minimum, maximum = data[0], data[0]
        eps = np.finfo(np.float64).eps
        for w in data:
            minimum = np.minimum(minimum, np.array(w))
            maximum = np.maximum(maximum, np.array(w))
        self.__minimum = np.nan_to_num(minimum, nan=-float(eps))
        self.__maximum = np.nan_to_num(maximum, nan=float(eps))
        delta = self.__maximum - self.__minimum
        self.__delta = np.maximum(eps, delta)

    def fit(self, X):
        """
        Train internal parameters.

        This method builds an internal :class:`tdamapper.cover.BallCover`
        attribute that allows efficient queries of the dataset.

        :param X: A dataset of n points used to extract parameters and perform
            training.
        :type X: array-like of shape (n, m) or list-like of length n
        :return: The object itself.
        :rtype: self
        """
        self._set_bounds(X)
        self.__ball_proximity.fit(X)
        return self

    def search(self, x):
        """
        Return a list of neighbors for the query point.

        This method takes a target point as input and returns the hypercube
        whose center is closest to the target point.

        :param x: A query point for which we want to find neighbors.
        :type x: Any
        :return: The indices of the neighbors contained in the dataset.
        :rtype: list[int]
        """
        return self.__ball_proximity.search(self._phi(x))


class TrivialProximity(Proximity):
    """
    Proximity function that returns the whole dataset.

    This class creates a single open set that contains all the points in the
    dataset.
    """

    def fit(self, X):
        """
        Train internal parameters.

        This method stores the dataset in an internal attribute.

        :param X: A dataset of n points used to extract parameters and perform
            training.
        :type X: array-like of shape (n, m) or list-like of length n
        :return: The object itself.
        :rtype: self
        """
        self.__data = X
        return self

    def search(self, x):
        """
        Return a list of neighbors for the query point.

        This method returns a list of all the ids ranging from zero to the
        length of the dataset.

        :param x: A query point for which we want to find neighbors.
        :type x: Any
        :return: The indices of the neighbors contained in the dataset.
        :rtype: list[int]
        """
        return list(range(len(self.__data)))

