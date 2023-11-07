import numpy as np
from .utils.vptree_flat import VPTree


class BallNeighbors:

    def __init__(self, radius, metric):
        self.__metric = lambda x, y: metric(x[1], y[1])
        self.__radius = radius
        self.__vptree = None
        self.__data = None

    def fit(self, data):
        self.__data = list(enumerate(data))
        self.__vptree = VPTree(
            self.__metric, self.__data, leaf_radius=self.__radius)
        return self

    def search(self, point):
        if self.__vptree:
            neighs = self.__vptree.ball_search((-1, point), self.__radius)
            return [x for (x, _) in neighs]
        return []

    def get_params(self, deep=True):
        parameters = {}
        parameters['radius'] = self.__radius
        parameters['metric'] = self.__metric
        return parameters


class KNNeighbors:

    def __init__(self, k_neighbors, metric):
        self.__k_neighbors = k_neighbors
        self.__metric = lambda x, y: metric(x[1], y[1])
        self.__vptree = None
        self.__data = None

    def fit(self, data):
        self.__data = list(enumerate(data))
        self.__vptree = VPTree(self.__metric, self.__data, leaf_size=self.__k_neighbors)
        return self

    def search(self, point):
        if self.__vptree:
            neighs = self.__vptree.knn_search((-1, point), self.__k_neighbors)
            return [x for (x, _) in neighs]
        return []

    def get_params(self, deep=True):
        parameters = {}
        parameters['k_neighbors'] = self.__k_neighbors
        parameters['metric'] = self.__metric
        return parameters


class GridNeighbors:

    def __init__(self, n_intervals, overlap_frac):
        self.__n_intervals = n_intervals
        self.__overlap_frac = overlap_frac
        self.__radius = (1.0 + overlap_frac) / 2.0
        self.__minimum = None
        self.__maximum = None
        metric = self._pullback(self._gamma_n, self._l_infty)
        self.__ball_neighbors = BallNeighbors(self.__radius, metric)

    def _l_infty(self, x, y):
        return np.linalg.norm(x - y, ord=np.inf)

    def _gamma_n(self, x):
        return self.__n_intervals * (x - self.__minimum) / (self.__maximum - self.__minimum)

    def _gamma_n_inv(self, x):
        return self.__minimum + (self.__maximum - self.__minimum) * x / self.__n_intervals

    def _rho(self, x):
        return x.round()

    def _phi(self, x):
        return self._gamma_n_inv(self._rho(self._gamma_n(x)))

    def _pullback(self, fun, dist):
        return lambda x, y: dist(fun(x), fun(y))

    def _set_bounds(self, data):
        if data is None:
            return
        minimum = np.array([float('inf') for _ in data[0]])
        maximum = np.array([-float('inf') for _ in data[0]])
        eps = np.finfo(np.float64).eps
        for w in data:
            minimum = np.minimum(minimum, np.array(w))
            maximum = np.maximum(maximum, np.array(w))
        self.__minimum = np.nan_to_num(minimum, nan=-eps)
        self.__maximum = np.nan_to_num(maximum, nan=eps)

    def fit(self, data):
        self._set_bounds(data)
        self.__ball_neighbors.fit(data)
        return

    def search(self, point):
        return self.__ball_neighbors.search(self._phi(point))

    def get_params(self, deep=True):
        parameters = {}
        parameters['n_intervals'] = self.__n_intervals
        parameters['overlap_frac'] = self.__overlap_frac
        return parameters


class TrivialNeighbors:

    def __init__(self):
        self.__data = None

    def fit(self, data):
        self.__data = data
        return self

    def search(self, point=None):
        return list(range(len(self.__data)))

    def get_params(self, deep=True):
        parameters = {}
        return parameters
