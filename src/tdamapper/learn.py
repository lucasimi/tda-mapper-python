"""
This module provides classes based on the Mapper algorithm, a technique from
topological data analysis (TDA) for extracting insights from complex data.
Each class is designed to be compatible with scikit-learn's estimator APIs,
ensuring seamless integration with existing machine learning pipelines.

Users can leverage these classes to explore high-dimensional data, visualize
relationships, and uncover meaningful structures in a manner that aligns with
scikit-learn's conventions for estimators.
"""

from __future__ import annotations

from typing import Optional

import networkx as nx

from tdamapper.clustering import _MapperClustering
from tdamapper.core import ArrayLike, Clustering, Cover, _MapperAlgorithm


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
    :param clustering: The clustering algorithm to apply to each subset of the
        dataset.
    :param n_jobs: The maximum number of parallel clustering jobs. This
        parameter is passed to the constructor of :class:`joblib.Parallel`.
        Defaults to 1.
    """

    def __init__(
        self,
        cover: Optional[Cover] = None,
        clustering: Optional[Clustering] = None,
        n_jobs: int = 1,
    ):
        super().__init__(
            cover=cover,
            clustering=clustering,
            n_jobs=n_jobs,
        )

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> MapperClustering:
        """
        Fit the clustering algorithm to the data.

        This method computes the connected components of the Mapper graph and
        assigns labels to each point in the dataset based on their connected
        component.

        :param X: The input features array.
        :param y: The target values array. If None, `x_arr` is used as `y_arr`.
        :return: The fitted MapperClustering object.
        """
        super().fit(X, y)
        return self


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
    :param clustering: The clustering algorithm to apply to each subset of the
        dataset. If no clustering is specified,
        :class:`tdamapper.core.TrivialClustering` is used, which produces a
        single cluster for each subset. Defaults to None.
    :param failsafe: A flag that is used to prevent failures. If True, the
        clustering object is wrapped by
        :class:`tdamapper.core.FailSafeClustering`. Defaults to True.
    :param verbose: A flag that is used for logging, supplied to
        :class:`tdamapper.core.FailSafeClustering`. If True, clustering
        failures are logged. Set to False to suppress these messages. Defaults
        to True.
    :param n_jobs: The maximum number of parallel clustering jobs. This
        parameter is passed to the constructor of :class:`joblib.Parallel`.
        Defaults to 1.
    """

    def __init__(
        self,
        cover: Optional[Cover] = None,
        clustering: Optional[Clustering] = None,
        failsafe: bool = True,
        verbose: bool = True,
        n_jobs: int = 1,
    ):
        super().__init__(
            cover=cover,
            clustering=clustering,
            failsafe=failsafe,
            verbose=verbose,
            n_jobs=n_jobs,
        )

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None):
        """
        Create the Mapper graph and store it for later use.

        This method stores the result of :func:`tdamapper.core.mapper_graph` in
        the attribute `graph_` and returns a reference to the calling object.

        :param X: A dataset of n points.
        :param y: Lens values for the n points of the dataset.
        :return: The object itself.
        """
        super().fit(X, y)
        return self

    def fit_transform(self, X: ArrayLike, y: Optional[ArrayLike]) -> nx.Graph:
        """
        Create the Mapper graph.

        This method is equivalent to calling
        :func:`tdamapper.core.mapper_graph`.

        :param X: A dataset of n points.
        :param y: Lens values for the n points of the dataset.
        :return: The Mapper graph.
        """
        return super().fit_transform(X, y)
