import numpy as np
from .utils.vptree_flat import VPTree


class BallProximity:

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


class KNNProximity:

    def __init__(self, neighbors, metric):
        self.neighbors = neighbors
        self.metric = lambda x, y: metric(x[1], y[1])
        self.__vptree = None
        self.__data = None

    def fit(self, data):
        self.__data = list(enumerate(data))
        self.__vptree = VPTree(self.metric, self.__data, leaf_size=self.neighbors)
        return self

    def search(self, point):
        if self.__vptree:
            neighs = self.__vptree.knn_search((-1, point), self.neighbors)
            return [x for (x, _) in neighs]
        return []

    def get_params(self, deep=True):
        parameters = {}
        parameters['neighbors'] = self.neighbors
        parameters['metric'] = self.metric
        return parameters


class GridProximity:

    def __init__(self, intervals, overlap_frac):
        self.intervals = intervals
        self.overlap_frac = overlap_frac
        self.__radius = (1.0 + overlap_frac) / 2.0
        self.__minimum = None
        self.__maximum = None
        self.__delta = None
        metric = self._pullback(self._gamma_n, self._l_infty)
        self.__ball_proximity = BallProximity(self.__radius, metric)

    def _l_infty(self, x, y):
        return np.linalg.norm(x - y, ord=np.inf)

    def _gamma_n(self, x):
        return self.intervals * (x - self.__minimum) / self.__delta

    def _gamma_n_inv(self, x):
        return self.__minimum + self.__delta * x / self.intervals

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
        delta = self.__maximum - self.__minimum
        eps = np.finfo(np.float64).eps
        self.__delta = np.array([max(x, eps) for x in delta])

    def fit(self, data):
        self._set_bounds(data)
        self.__ball_proximity.fit(data)
        return

    def search(self, point):
        return self.__ball_proximity.search(self._phi(point))

    def get_params(self, deep=True):
        parameters = {}
        parameters['intervals'] = self.intervals
        parameters['overlap_frac'] = self.overlap_frac
        return parameters


class CubicalProximity:

    def __init__(self, intervals, overlap_frac):
        self.__metric = lambda x, y: np.linalg.norm(x[1] - y[1], ord=np.inf)
        self.intervals = intervals
        self.overlap_frac = overlap_frac
        self.__radius = (1.0 + self.overlap_frac) / 2.0
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
        delta = (self.__maximum - self.__minimum) / self.intervals
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
        else:
            return []

    def get_params(self, deep=True):
        parameters = {}
        parameters['intervals'] = self.intervals
        parameters['overlap_frac'] = self.overlap_frac
        return parameters


class TrivialProximity:

    def __init__(self):
        pass

    def fit(self, data):
        self.__data = data
        return self

    def search(self, point=None):
        return list(range(len(self.__data)))

    def get_params(self, deep=True):
        return {}
