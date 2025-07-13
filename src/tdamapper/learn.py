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

from typing import Generic, Optional, TypeVar

import networkx as nx

from tdamapper._common import EstimatorMixin, ParamsMixin, clone
from tdamapper.core import (
    FailSafeClustering,
    TrivialClustering,
    TrivialCover,
    mapper_connected_components,
    mapper_graph,
)
from tdamapper.protocols import ArrayRead, Clustering, Cover

S_contra = TypeVar("S_contra", contravariant=True)

T_contra = TypeVar("T_contra", contravariant=True)


class MapperClustering(EstimatorMixin, ParamsMixin, Generic[S_contra, T_contra]):
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
    """

    labels_: list[int]

    def __init__(
        self,
        cover: Optional[Cover[T_contra]] = None,
        clustering: Optional[Clustering[S_contra]] = None,
        n_jobs: int = 1,
    ) -> None:
        self.cover = cover
        self.clustering = clustering
        self.n_jobs = n_jobs

    def fit(
        self, X: ArrayRead[S_contra], y: Optional[ArrayRead[T_contra]] = None
    ) -> MapperClustering[S_contra, T_contra]:
        """
        Fit the clustering algorithm to the data.

        :param X: A dataset of n points.
        :param y: Ignored.
        :return: self
        """
        y_ = X if y is None else y
        X, y_ = self._validate_X_y(X, y_)
        cover = TrivialCover() if self.cover is None else self.cover
        cover = clone(cover)
        clustering = TrivialClustering() if self.clustering is None else self.clustering
        clustering = clone(clustering)
        n_jobs = self.n_jobs
        itm_lbls = mapper_connected_components(
            X,
            y_,
            cover,
            clustering,
            n_jobs=n_jobs,
        )
        self.labels_ = [itm_lbls[i] for i, _ in enumerate(X)]
        self._set_n_features_in(X)
        return self


class MapperAlgorithm(EstimatorMixin, ParamsMixin, Generic[S_contra, T_contra]):
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
        open cover containing the whole dataset.
    :param clustering: The clustering algorithm to apply to each subset of the
        dataset. If no clustering is specified,
        :class:`tdamapper.core.TrivialClustering` is used, which produces a
        single cluster for each subset.
    :param failsafe: A flag that is used to prevent failures. If True, the
        clustering object is wrapped by
        :class:`tdamapper.core.FailSafeClustering`.
    :param verbose: A flag that is used for logging, supplied to
        :class:`tdamapper.core.FailSafeClustering`. If True, clustering
        failures are logged. Set to False to suppress these messages.
    :param n_jobs: The maximum number of parallel clustering jobs. This
        parameter is passed to the constructor of :class:`joblib.Parallel`.
    """

    _cover: Cover[T_contra]
    _clustering: Clustering[S_contra]
    _verbose: bool
    _failsafe: bool
    _n_jobs: int
    graph_: nx.Graph

    def __init__(
        self,
        cover: Optional[Cover[T_contra]] = None,
        clustering: Optional[Clustering[S_contra]] = None,
        failsafe: bool = True,
        verbose: bool = True,
        n_jobs: int = 1,
    ) -> None:
        self.cover = cover
        self.clustering = clustering
        self.failsafe = failsafe
        self.verbose = verbose
        self.n_jobs = n_jobs

    def fit(
        self, X: ArrayRead[S_contra], y: Optional[ArrayRead[T_contra]] = None
    ) -> MapperAlgorithm[S_contra, T_contra]:
        """
        Create the Mapper graph and store it for later use.

        This method stores the result of :func:`tdamapper.core.mapper_graph` in
        the attribute `graph_` and returns a reference to the calling object.

        :param X: A dataset of n points.
        :param y: Lens values for the n points of the dataset.
        :return: The object itself.
        """
        y_ = X if y is None else y
        X, y_ = self._validate_X_y(X, y_)
        self._cover = TrivialCover() if self.cover is None else self.cover
        self._clustering = (
            TrivialClustering() if self.clustering is None else self.clustering
        )
        self._verbose = self.verbose
        self._failsafe = self.failsafe
        if self._failsafe:
            self._clustering = FailSafeClustering(
                clustering=self._clustering,
                verbose=self._verbose,
            )
        self._cover = clone(self._cover)
        self._clustering = clone(self._clustering)
        self._n_jobs = self.n_jobs
        self.graph_ = mapper_graph(
            X,
            y_,
            self._cover,
            self._clustering,
            n_jobs=self._n_jobs,
        )
        self._set_n_features_in(X)
        return self

    def fit_transform(self, X: ArrayRead[S_contra], y: ArrayRead[T_contra]) -> nx.Graph:
        """
        Create the Mapper graph.

        This method is equivalent to calling
        :func:`tdamapper.core.mapper_graph`.

        :param X: A dataset of n points.
        :param y: Lens values for the n points of the dataset.
        :return: The Mapper graph.
        """
        self.fit(X, y)
        return self.graph_
