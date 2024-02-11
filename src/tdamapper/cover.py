"""A module containing the logic for building open covers for the Mapper algorithm."""
import numpy as np

from tdamapper.utils.vptree_flat import VPTree as FVPT
from tdamapper.utils.vptree import VPTree as VPT


def _pullback(fun, dist):
    return lambda x, y: dist(fun(x), fun(y))


def _rho(x):
    return np.floor(x) + 0.5


def _l_infty(x, y):
    return np.max(np.abs(x - y))  # in alternative: np.linalg.norm(x - y, ord=np.inf)


def proximity_net(X, proximity):
    """
    Compute proximity-net for a given proximity function.

    Returns a generator where each item is a subset of ids of points from `X`.

    :param X: A dataset.
    :type X: `numpy.ndarray` or list-like.
    :param proximity: A proximity function.
    :type proximity: `tdamapper.cover.Proximity`
    """
    covered_ids = set()
    proximity.fit(X)
    for i, xi in enumerate(X):
        if i not in covered_ids:
            neigh_ids = proximity.search(xi)
            covered_ids.update(neigh_ids)
            if neigh_ids:
                yield neigh_ids


class Proximity:
    """
    This class serves as a blueprint for proximity functions used inside `proximity_net`.

    Subclasses are expected to override the methods `fit` and `search`.
    """

    def fit(self, X):
        self.__X = X
        return self

    def search(self, x):
        return [i for i, _ in enumerate(self.__X)]


class ProximityNetCover(Proximity):
    """
    This class serves as a blueprint for cover algorithm based on proximity-net.
    """

    def apply(self, X):
        return proximity_net(X, self)


class BallCover(ProximityNetCover):
    """
    Creates an open cover made of overlapping open balls of fixed radius.

    This class implements the Ball Proximity function: after calling `fit`, the `search` method
    returns all the points within a ball centered in the target point.

    :param radius: The radius of open balls
    :type radius: float.
    :param metric: The metric used to define open balls.
    :type metric: Callable.
    :param flat: Set to True to use flat vptrees.
    :type flat: `bool`
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
        self.__data = list(enumerate(X))
        self.__vptree = self.__flat_vpt() if self.__flat else self.__vpt()
        return self

    def search(self, x):
        if self.__vptree is None:
            return []
        neighs = self.__vptree.ball_search((-1, x), self.__radius)
        return [x for (x, _) in neighs]


class KNNCover(ProximityNetCover):
    """
    Creates an open cover where each open set contains a fixed number of neighbors, using KNN.

    This class implements the KNN Proximity function: after calling `fit`, the `search` method
    returns the k nearest points to the target point.

    :param neighbors: The number of neighbors.
    :type neighbors: int.
    :param metric: The metric used to search neighbors.
    :type metric: function.
    :param flat: Set to True to use flat vptrees.
    :type flat: `bool`
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
        self.__data = list(enumerate(X))
        self.__vptree = self.__flat_vpt() if self.__flat else self.__vpt()
        return self

    def search(self, x):
        if self.__vptree is None:
            return []
        neighs = self.__vptree.knn_search((-1, x), self.__neighbors)
        return [x for (x, _) in neighs]


class CubicalCover(ProximityNetCover):
    """
    Creates an open cover of hypercubes of evenly-sized sides and overlap.

    This class implements the Cubical Proximity function: after calling `fit`, the `search` method
    returns the hypercube whose center is nearest to the target point. Each hypercube is the
    product of 1-dimensional intervals with the same length and overlap.

    :param n_intervals: The number of intervals on each dimension.
    :type n_intervals: int.
    :param overlap_frac: The overlap fraction.
    :type overlap_frac: `float` in (0.0, 1.0).
    :param flat: Set to True to use flat vptrees.
    :type flat: `bool`
    """

    def __init__(self, n_intervals, overlap_frac, flat=True):
        self.__n_intervals = n_intervals
        self.__radius = 1.0 / (2.0 - 2.0 * overlap_frac)
        self.__minimum = None
        self.__maximum = None
        self.__delta = None
        metric = _pullback(self._gamma_n, _l_infty)
        self.__ball_proximity = BallCover(self.__radius, metric, flat=flat)

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
        self._set_bounds(X)
        self.__ball_proximity.fit(X)
        return self

    def search(self, x):
        return self.__ball_proximity.search(self._phi(x))


class TrivialCover(ProximityNetCover):
    """
    Creates an open cover made of a single open set that contains the whole dataset.
    """

    def fit(self, X):
        self.__data = X
        return self

    def search(self, x):
        return list(range(len(self.__data)))
