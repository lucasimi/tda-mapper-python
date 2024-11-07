"""
Open cover construction for the Mapper algorithm.

An open cover is a collection of open subsets of a dataset whose union spans
the whole dataset. Unlike clustering, open subsets do not need to be disjoint.
Indeed, the overlaps of the open subsets define the edges of the Mapper graph.
"""

import numpy as np

from tdamapper.core import Proximity
from tdamapper.utils.metrics import get_metric, chebyshev
from tdamapper.utils.vptree import VPTree

from tdamapper._common import warn_user


class _Pullback:

    def __init__(self, fun, dist):
        self.fun = fun
        self.dist = dist

    def __call__(self, x, y):
        return self.dist(self.fun(x), self.fun(y))


def _snd(x):
    return x[1]


def _rho(x):
    return np.floor(x) + 0.5


class BallCover(Proximity):
    """
    Cover algorithm based on `ball proximity function`, covering data with
    open balls of fixed radius.

    An open ball is a set of points within a specified distance from a center
    point. This class maps each point to its corresponding open ball with a
    fixed radius centered on the point itself.

    :param radius: The radius of the open balls. Must be a positive value.
        Defaults to 1.0.
    :type radius: float
    :param metric: The metric used to define the distance between points.
        Accepts any value compatible with `tdamapper.utils.metrics.get_metric`.
        Defaults to 'euclidean'.
    :type metric: str or callable
    :param metric_params: Additional parameters for the metric function, to be
        passed to `tdamapper.utils.metrics.get_metric`. Defaults to None.
    :type metric_params: dict, optional
    :param kind: Specifies whether to use a flat or a hierarchical vantage
        point tree. Acceptable values are 'flat' or 'hierarchical'. Defaults
        to 'flat'.
    :type kind: str
    :param leaf_capacity: The maximum number of points in a leaf node of the
        vantage point tree. Must be a positive value. Defaults to 1.
    :type leaf_capacity: int
    :param leaf_radius: The radius of the leaf nodes. If not specified, it
        defaults to the value of `radius`. Must be a positive value. Defaults
        to None.
    :type leaf_radius: float, optional
    :param pivoting: The method used for pivoting in the vantage point tree.
        Acceptable values are None, 'random', or 'furthest'. Defaults to None.
    :type pivoting: str or callable, optional
    """

    def __init__(
        self,
        radius=1.0,
        metric='euclidean',
        metric_params=None,
        kind='flat',
        leaf_capacity=1,
        leaf_radius=None,
        pivoting=None,
    ):
        self.radius = radius
        self.metric = metric
        self.metric_params = metric_params
        self.kind = kind
        self.leaf_capacity = leaf_capacity
        self.leaf_radius = leaf_radius
        self.pivoting = pivoting

    def fit(self, X):
        """
        Train internal parameters.

        This method creates a vptree on the dataset in order to perform fast
        range queries in the func:`tdamapper.cover.BallCover.search`
        method.

        :param X: A dataset of n points.
        :type X: array-like of shape (n, m) or list-like of length n
        :return: The object itself.
        :rtype: self
        """
        metric = get_metric(self.metric, **(self.metric_params or {}))
        self.__radius = self.radius
        self.__data = list(enumerate(X))
        self.__vptree = VPTree(
            self.__data,
            metric=_Pullback(_snd, metric),
            metric_params=None,
            kind=self.kind,
            leaf_capacity=self.leaf_capacity,
            leaf_radius=self.leaf_radius or self.radius,
            pivoting=self.pivoting,
        )
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
        neighs = self.__vptree.ball_search(
            (-1, x),
            self.__radius,
            inclusive=False
        )
        return [x for (x, _) in neighs]


class KNNCover(Proximity):
    """
    Cover algorithm based on `KNN proximity function`, covering data using
    k-nearest neighbors (KNN).

    This class maps each point to the set of the k nearest neighbors to the
    point itself.

    :param neighbors: The number of neighbors to use for the KNN Proximity
        function, must be positive and less than the length of the dataset.
        Defaults to 1.
    :type neighbors: int
    :param metric: The metric used to define the distance between points.
        Accepts any value compatible with `tdamapper.utils.metrics.get_metric`.
        Defaults to 'euclidean'.
    :type metric: str or callable
    :param metric_params: Additional parameters for the metric function, to be
        passed to `tdamapper.utils.metrics.get_metric`. Defaults to None.
    :type metric_params: dict, optional
    :param kind: Specifies whether to use a flat or a hierarchical vantage
        point tree. Acceptable values are 'flat' or 'hierarchical'. Defaults
        to 'flat'.
    :type kind: str
    :param leaf_capacity: The maximum number of points in a leaf node of the
        vantage point tree. If not specified, it defaults to the value of
        `neighbors`. Must be a positive value. Defaults to None.
    :type leaf_capacity: int, optional
    :param leaf_radius: The radius of the leaf nodes. Must be a positive value.
        Defaults to 0.0.
    :type leaf_radius: float
    :param pivoting: The method used for pivoting in the vantage point tree.
        Acceptable values are None, 'random', or 'furthest'. Defaults to None.
    :type pivoting: str or callable, optional
    """

    def __init__(
        self,
        neighbors=1,
        metric='euclidean',
        metric_params=None,
        kind='flat',
        leaf_capacity=None,
        leaf_radius=0.0,
        pivoting=None,
    ):
        self.neighbors = neighbors
        self.metric = metric
        self.metric_params = metric_params
        self.kind = kind
        self.leaf_capacity = leaf_capacity
        self.leaf_radius = leaf_radius
        self.pivoting = pivoting

    def fit(self, X):
        """
        Train internal parameters.

        This method creates a vptree on the dataset in order to perform fast
        KNN queries in the func:`tdamapper.cover.BallCover.search`
        method.

        :param X: A dataset of n points.
        :type X: array-like of shape (n, m) or list-like of length n
        :return: The object itself.
        :rtype: self
        """
        metric = get_metric(self.metric, **(self.metric_params or {}))
        self.__neighbors = self.neighbors
        self.__data = list(enumerate(X))
        self.__vptree = VPTree(
            self.__data,
            metric=_Pullback(_snd, metric),
            metric_params=None,
            kind=self.kind,
            leaf_capacity=self.leaf_capacity or self.neighbors,
            leaf_radius=self.leaf_radius,
            pivoting=self.pivoting,
        )
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


class CubicalCover(Proximity):
    """
    Cover algorithm based on the `cubical proximity function`, covering data
    with open hypercubes of uniform size and overlap.

    A hypercube is a multidimensional generalization of a square or a cube.
    The size and overlap of the hypercubes are determined by the number of
    intervals and the overlap fraction parameters. This class maps each point
    to the hypercube with the nearest center.

    :param n_intervals: The number of intervals to use for each dimension.
        Must be positive and less than or equal to the length of the dataset.
        Defaults to 1.
    :type n_intervals: int
    :param overlap_frac: The fraction of overlap between adjacent intervals on
        each dimension, must be in the range (0.0, 1.0). Defaults to 0.5.
    :type overlap_frac: float
    :param metric: The metric used to define the distance between points.
        Accepts any value compatible with `tdamapper.utils.metrics.get_metric`.
        Defaults to 'euclidean'.
    :type metric: str or callable
    :param metric_params: Additional parameters for the metric function, to be
        passed to `tdamapper.utils.metrics.get_metric`. Defaults to None.
    :type metric_params: dict, optional
    :param kind: Specifies whether to use a flat or a hierarchical vantage
        point tree. Acceptable values are 'flat' or 'hierarchical'. Defaults to
        'flat'.
    :type kind: str
    :param leaf_capacity: The maximum number of points in a leaf node of the
        vantage point tree. Must be a positive value. Defaults to 1.
    :type leaf_capacity: int
    :param leaf_radius: The radius of the leaf nodes. If not specified, it
        defaults to the value of `radius`. Must be a positive value. Defaults
        to None.
    :type leaf_radius: float, optional
    :param pivoting: The method used for pivoting in the vantage point tree.
        Acceptable values are None, 'random', or 'furthest'. Defaults to None.
    :type pivoting: str or callable, optional
    """

    def __init__(
        self,
        n_intervals=1,
        overlap_frac=None,
        kind='flat',
        leaf_capacity=1,
        leaf_radius=None,
        pivoting=None,
    ):
        self.n_intervals = n_intervals
        self.overlap_frac = overlap_frac
        self.kind = kind
        self.leaf_capacity = leaf_capacity
        self.leaf_radius = leaf_radius
        self.pivoting = pivoting
        if (self.overlap_frac <= 0.0) or (self.overlap_frac > 0.5):
            warn_user('The parameter overlap_frac is expected within range (0.0, 0.5]')

    def _gamma_n(self, x):
        return self.__n_intervals * (x - self.__min) / self.__delta

    def _gamma_n_inv(self, x):
        return self.__min + self.__delta * x / self.__n_intervals

    def _phi(self, x):
        return self._gamma_n_inv(_rho(self._gamma_n(x)))

    def _get_bounds(self, data):
        if (data is None) or len(data) == 0:
            return
        minimum, maximum = data[0], data[0]
        eps = np.finfo(np.float64).eps
        for w in data:
            minimum = np.minimum(minimum, np.array(w))
            maximum = np.maximum(maximum, np.array(w))
        _min = np.nan_to_num(minimum, nan=-float(eps))
        _max = np.nan_to_num(maximum, nan=float(eps))
        _delta = np.maximum(eps, _max - _min)
        return _min, _max, _delta

    def _convert(self, X):
        return np.asarray(X).reshape(len(X), -1).astype(float)

    def fit(self, X):
        """
        Train internal parameters.

        This method builds an internal :class:`tdamapper.cover.BallCover`
        attribute that allows efficient queries of the dataset.

        :param X: A dataset of n points.
        :type X: array-like of shape (n, m) or list-like of length n
        :return: The object itself.
        :rtype: self
        """
        self.__overlap_frac = self.overlap_frac
        self.__n_intervals = self.n_intervals
        self.__radius = 1.0 / (2.0 - 2.0 * self.__overlap_frac)
        XX = self._convert(X)
        self.__ball_proximity = BallCover(
            self.__radius,
            metric=_Pullback(self._gamma_n, chebyshev()),
            kind=self.kind,
            leaf_capacity=self.leaf_capacity,
            leaf_radius=self.leaf_radius,
            pivoting=self.pivoting
        )
        self.__min, self.__max, self.__delta = self._get_bounds(XX)
        self.__ball_proximity.fit(XX)
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
