"""
This module provides classes based on the Mapper algorithm, a technique from
topological data analysis (TDA) for extracting insights from complex data.
Each class is designed to be compatible with scikit-learn's estimator APIs,
ensuring seamless integration with existing machine learning pipelines.

Users can leverage these classes to explore high-dimensional data, visualize
relationships, and uncover meaningful structures in a manner that aligns with
scikit-learn's conventions for estimators.
"""

from tdamapper.core import _MapperAlgorithm
from tdamapper.clustering import _MapperClustering


class MapperClustering(_MapperClustering):
    """
    A clustering algorithm based on the Mapper graph.

    The Mapper algorithm constructs a graph from a dataset, where each node
    represents a cluster of points and each edge represents an overlap between
    clusters. Each point in the dataset belongs to one or more nodes in the
    graph. These nodes are therefore all connected and share the same connected
    component in the Mapper graph. This class builds clusters of points
    according to their connected component in the Mapper graph.

    This class does not compute the Mapper graph but calls the function
    :func:`tdamapper.core.mapper_connected_components`, which is faster.

    :param cover: A cover algorithm.
    :type cover: A class compatible with :class:`tdamapper.core.Cover`
    :param clustering: The clustering algorithm to apply to each subset of the
        dataset.
    :type clustering: A class compatible with scikit-learn estimators from
        :mod:`sklearn.cluster`
    :param n_jobs: The maximum number of parallel clustering jobs. This
        parameter is passed to the constructor of :class:`joblib.Parallel`.
        Defaults to 1.
    :type n_jobs: int
    """

    def __init__(
        self,
        cover=None,
        clustering=None,
        n_jobs=1,
    ):
        super().__init__(
            cover=cover,
            clustering=clustering,
            n_jobs=n_jobs,
        )

    def fit(self, X, y=None):
        """
        Fit the clustering algorithm to the data.

        :param X: A dataset of n points.
        :type X: array-like of shape (n, m) or list-like of length n
        :param y: Ignored.
        :return: self
        """
        return super().fit(X, y)


class MapperAlgorithm(_MapperAlgorithm):
    """
    A class for creating and analyzing Mapper graphs.

    This class provides two methods :func:`fit` and :func:`fit_transform`. Once
    fitted, the Mapper graph is stored in the attribute `graph_` as a
    :class:`networkx.Graph` object.

    This class adopts the same interface as scikit-learn's estimators for ease
    and consistency of use. However, it's important to note that this is not a
    proper scikit-learn estimator as it does not validata the input in the same
    way as a scikit-learn estimator is required to do. This class can work
    with datasets whose elements are arbitrary objects when feasible for the
    supplied parameters.

    :param cover: A cover algorithm. If no cover is specified,
        :class:`tdamapper.core.TrivialCover` is used, which produces a single
        open cover containing the whole dataset. Defaults to None.
    :type cover: A class compatible with :class:`tdamapper.core.Cover`
    :param clustering: The clustering algorithm to apply to each subset of the
        dataset. If no clustering is specified,
        :class:`tdamapper.core.TrivialClustering` is used, which produces a
        single cluster for each subset. Defaults to None.
    :type clustering: An estimator compatible with scikit-learn's clustering
        interface, typically from :mod:`sklearn.cluster`.
    :param failsafe: A flag that is used to prevent failures. If True, the
        clustering object is wrapped by
        :class:`tdamapper.core.FailSafeClustering`. Defaults to True.
    :type failsafe: bool, optional
    :param verbose: A flag that is used for logging, supplied to
        :class:`tdamapper.core.FailSafeClustering`. If True, clustering
        failures are logged. Set to False to suppress these messages. Defaults
        to True.
    :type verbose: bool, optional
    :param n_jobs: The maximum number of parallel clustering jobs. This
        parameter is passed to the constructor of :class:`joblib.Parallel`.
        Defaults to 1.
    :type n_jobs: int
    """

    def __init__(
        self,
        cover=None,
        clustering=None,
        failsafe=True,
        verbose=True,
        n_jobs=1,
    ):
        super().__init__(
            cover=cover,
            clustering=clustering,
            failsafe=failsafe,
            verbose=verbose,
            n_jobs=n_jobs
        )

    def fit(self, X, y=None):
        """
        Create the Mapper graph and store it for later use.

        This method stores the result of :func:`tdamapper.core.mapper_graph` in
        the attribute `graph_` and returns a reference to the calling object.

        :param X: A dataset of n points.
        :type X: array-like of shape (n, m) or list-like of length n
        :param y: Lens values for the n points of the dataset.
        :type y: array-like of shape (n, k) or list-like of length n
        :return: The object itself.
        """
        return super().fit(X, y)

    def fit_transform(self, X, y):
        """
        Create the Mapper graph.

        This method is equivalent to calling
        :func:`tdamapper.core.mapper_graph`.

        :param X: A dataset of n points.
        :type X: array-like of shape (n, m) or list-like of length n
        :param y: Lens values for the n points of the dataset.
        :type y: array-like of shape (n, k) or list-like of length n
        :return: The Mapper graph.
        :rtype: :class:`networkx.Graph`
        """
        return super().fit_transform(X, y)
