import numpy as np
from tdamapper.utils.vptree_flat import VPTree


class BallProximity:

    def __init__(self, radius, metric):
        self.__metric = lambda x, y: metric(x[1], y[1])
        self.__radius = radius
        self.__data = None
        self.__vptree = None

    def fit(self, X):
        self.__data = list(enumerate(X))
        self.__vptree = VPTree(
            self.__metric, self.__data, leaf_radius=self.__radius)
        return self

    def search(self, x):
        if self.__vptree:
            neighs = self.__vptree.ball_search((-1, x), self.__radius)
            return [x for (x, _) in neighs]
        return []


class KNNProximity:

    def __init__(self, neighbors, metric):
        self.__neighbors = neighbors
        self.__metric = lambda x, y: metric(x[1], y[1])
        self.__data = None
        self.__vptree = None

    def fit(self, X):
        self.__data = list(enumerate(X))
        self.__vptree = VPTree(self.__metric, self.__data, leaf_capacity=self.__neighbors)
        return self

    def search(self, x):
        if self.__vptree is None:
            return []
        neighs = self.__vptree.knn_search((-1, x), self.__neighbors)
        return [x for (x, _) in neighs]


class CubicalProximity:

    def __init__(self, n_intervals, overlap_frac):
        self.__n_intervals = n_intervals
        self.__overlap_frac = overlap_frac
        self.__radius = 1.0 / (2.0 - 2.0 * overlap_frac)
        self.__minimum = None
        self.__maximum = None
        self.__delta = None
        metric = self._pullback(self._gamma_n, self._l_infty)
        self.__ball_proximity = BallProximity(self.__radius, metric)

    def _l_infty(self, x, y):
        return np.max(np.abs(x - y)) # in alternative: np.linalg.norm(x - y, ord=np.inf)

    def _gamma_n(self, x):
        return self.__n_intervals * (x - self.__minimum) / self.__delta

    def _gamma_n_inv(self, x):
        return self.__minimum + self.__delta * x / self.__n_intervals

    def _rho(self, x):
        return np.floor(x) + 0.5

    def _phi(self, x):
        return self._gamma_n_inv(self._rho(self._gamma_n(x)))

    def _pullback(self, fun, dist):
        return lambda x, y: dist(fun(x), fun(y))

    def _set_bounds(self, data):
        if (data is None) or len(data) == 0:
            return
        minimum, maximum = data[0], data[0]
        eps = np.finfo(np.float64).eps
        for w in data:
            minimum = np.minimum(minimum, np.array(w))
            maximum = np.maximum(maximum, np.array(w))
        self.__minimum = np.nan_to_num(minimum, nan=-eps)
        self.__maximum = np.nan_to_num(maximum, nan=eps)
        delta = self.__maximum - self.__minimum
        eps = np.finfo(np.float64).eps
        self.__delta = np.maximum(eps, delta)

    def fit(self, X):
        self._set_bounds(X)
        self.__ball_proximity.fit(X)
        return

    def search(self, x):
        return self.__ball_proximity.search(self._phi(x))


class TrivialProximity:

    def fit(self, X):
        self.__data = X
        return self

    def search(self, x):
        return list(range(len(self.__data)))
