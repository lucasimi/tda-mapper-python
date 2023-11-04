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


class CubicNeighbors:

    def __init__(self, n_intervals, overlap_frac):
        self.__metric = lambda x, y: np.linalg.norm(x[1] - y[1], ord=np.inf)
        self.__n_intervals = n_intervals
        self.__overlap_frac = overlap_frac
        self.__radius = 1.0 + self.__overlap_frac
        self.__vptree = None
        self.__data = None
        self.__minimum = None
        self.__maximum = None
        self.__delta = None

    def _set_bounds(self, data):
        if data is None:
            return None, None
        minimum = np.array([float('inf') for _ in data[0]])
        maximum = np.array([-float('inf') for _ in data[0]])
        for w in data:
            minimum = np.minimum(minimum, np.array(w))
            maximum = np.maximum(maximum, np.array(w))
        self.__minimum = minimum
        self.__maximum = maximum
        eps = np.finfo(np.float64).eps
        delta = (self.__maximum - self.__minimum) / self.__n_intervals
        self.__delta = np.array([max(x, eps) for x in delta])

    def _nearest_center(self, x):
        return np.round((np.array(x) - self.__minimum) / self.__delta)

    def _normalize(self, x):
        return (np.array(x) - self.__minimum) / self.__delta

    def fit(self, data):
        self._set_bounds(data)
        self.__data = [(n, self._normalize(x)) for n, x in enumerate(data)]
        self.__vptree = VPTree(
            self.__metric, self.__data, leaf_radius=self.__radius)
        return self

    def search(self, point):
        if self.__vptree:
            center = self._nearest_center(point)
            neighs = self.__vptree.ball_search((-1, center), self.__radius)
            return [x for (x, _) in neighs if x != -1]
        return []

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
