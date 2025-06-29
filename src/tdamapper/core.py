"""
Core tools for creating and analyzing Mapper graphs.

The Mapper algorithm is a method for exploring the shape and structure of
high-dimensional datasets, by constructing a graph representation called Mapper
graph. The algorithm has three main steps:

1. **Filtering**: Apply a lens function (also called filter) to map the data
   points to a lower-dimensional space, such as a scalar value or a 2D plane.

2. **Covering**: Arrange the lens space into overlapping open sets, using a
   cover algorithm such as uniform intervals or balls.

3. **Clustering**: Group the data points in each open set into clusters, using
   a clustering algorithm such as single-linkage or DBSCAN.

The Mapper graph consists of nodes that represent clusters of data
points, and edges that connect overlapping clusters (clusters obtained from
different open sets can possibly overlap). For more details on the Mapper
algorithm and its applications, see

    Gurjeet Singh, Facundo Mémoli and Gunnar Carlsson, "Topological Methods for
    the Analysis of High Dimensional Data Sets and 3D Object Recognition",
    Eurographics Symposium on Point-Based Graphics, 2007.

This module provides the class :class:`tdamapper.core.MapperAlgorithm`, which
encapsulates the algorithm and its parameters. The Mapper graph produced by
this module is a NetworkX graph object.
"""

from __future__ import annotations

import logging
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Protocol,
    Union,
)

import networkx as nx
import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray

from tdamapper._common import EstimatorMixin, ParamsMixin, clone, deprecated
from tdamapper.unionfind import UnionFind

ATTR_IDS = "ids"

ATTR_SIZE = "size"


_logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s %(module)s %(levelname)s: %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)

PointLike = Union[Any, NDArray[np.float64]]

ArrayLike = Union[List[Any], NDArray[np.float64]]


def mapper_labels(
    X: ArrayLike,
    y: ArrayLike,
    cover: Cover,
    clustering: Clustering,
    n_jobs: int = 1,
) -> List[List[int]]:
    """
    Identify the nodes of the Mapper graph.

    The function first covers the lens space with overlapping sets, using the
    cover algorithm provided. Then, for each set, it clusters the points of the
    dataset that have lens values within that set, using the clustering
    algorithm provided. The clusters are then labeled with unique integers,
    starting from zero for each set. The function then adds an offset to the
    cluster labels, such that the labels are distinct across all sets. The
    offset is equal to the maximum label of the previous set plus one.

    The function returns a list of node labels for each point in the dataset.
    The list at position i contains the labels of the nodes that the point at
    position i belongs to. The labels are sorted in ascending order, and there
    are no duplicates. If i < j, the labels at position i are strictly less
    than those at position j.

    :param X: A dataset of n points.
    :param y: Lens values for the n points of the dataset.
    :param cover: A cover algorithm.
    :param clustering: A clustering algorithm.
    :param n_jobs: The maximum number of parallel clustering jobs. This
        parameter is passed to the constructor of :class:`joblib.Parallel`.
        Defaults to 1.
    :return: A list of node labels for each point in the dataset.
    """

    def _run_clustering(local_ids, X_local, clust):
        local_lbls = clust.fit(X_local).labels_
        return local_ids, local_lbls

    _lbls = Parallel(n_jobs, prefer="threads")(
        delayed(_run_clustering)(
            local_ids, [X[j] for j in local_ids], clone(clustering)
        )
        for local_ids in cover.fit_transform(y)
    )
    itm_lbls: List[List[int]] = [[] for _ in X]
    max_lbl = 0
    for local_ids, local_lbls in _lbls:
        max_local_lbl = 0
        for local_id, local_lbl in zip(local_ids, local_lbls):
            if local_lbl >= 0:
                itm_lbls[local_id].append(max_lbl + local_lbl)
            if local_lbl > max_local_lbl:
                max_local_lbl = local_lbl
        max_lbl += max_local_lbl + 1
    return itm_lbls


def mapper_connected_components(
    X: ArrayLike,
    y: ArrayLike,
    cover: Cover,
    clustering: Clustering,
    n_jobs: int = 1,
) -> List[int]:
    """
    Identify the connected components of the Mapper graph.

    A connected component is a maximal set of nodes that are reachable from
    each other by following the edges. This function assigns a unique integer
    label to each point in the dataset, based on the connected component of
    the Mapper graph that it belongs to.

    This function uses a union-find data structure to efficiently keep track of
    the connected components as it scans the points of the dataset. This
    approach should be faster than computing the Mapper graph by first calling
    :func:`tdamapper.core.mapper_graph` and then calling
    :func:`networkx.connected_components` on it.

    :param X: A dataset of n points.
    :param y: Lens values for the n points of the dataset.
    :param cover: A cover algorithm.
    :param clustering: The clustering algorithm to apply to each subset of the
        dataset.
    :param n_jobs: The maximum number of parallel clustering jobs. This
        parameter is passed to the constructor of :class:`joblib.Parallel`.
        Defaults to 1.
    :return: A list of labels. The label at position i identifies the connected
        component of the point at position i in the dataset.
    """
    itm_lbls = mapper_labels(X, y, cover, clustering, n_jobs=n_jobs)
    label_values = set()
    for lbls in itm_lbls:
        label_values.update(lbls)
    uf = UnionFind(label_values)
    for lbls in itm_lbls:
        if len(lbls) > 1:
            for first, second in zip(lbls, lbls[1:]):
                uf.union(first, second)
    labels = [-1 for _ in X]
    for i, lbls in enumerate(itm_lbls):
        # assign -1 to noise points
        root = uf.find(lbls[0]) if lbls else -1
        labels[i] = root
    return labels


def mapper_graph(
    X: ArrayLike,
    y: ArrayLike,
    cover: Cover,
    clustering: Clustering,
    n_jobs: int = 1,
) -> nx.Graph:
    """
    Create the Mapper graph.

    This function first identifies the unique cluster labels that each point of
    the dataset belongs to. These labels are used to identify the nodes of the
    Mapper graph. Then the edges of the Mapper graph are created by checking if
    any two nodes share some points in their corresponding clusters.

    This function return the Mapper graph as an object of type
    :class:`networkx.Graph`. Each node has an attribute 'size' that stores the
    number of points contained in its corresponding cluster, and an attribute
    'ids' that stores the indices of the points in the dataset that are
    contained in the cluster.

    :param X: A dataset of n points.
    :param y: Lens values for the n points of the dataset.
    :param cover: A cover algorithm.
    :param clustering: The clustering algorithm to apply to each subset of the
        dataset.
    :param n_jobs: The maximum number of parallel clustering jobs. This
        parameter is passed to the constructor of :class:`joblib.Parallel`.
        Defaults to 1.
    :return: The Mapper graph.
    """
    itm_lbls = mapper_labels(X, y, cover, clustering, n_jobs=n_jobs)
    graph = nx.Graph()
    for n, lbls in enumerate(itm_lbls):
        for lbl in lbls:
            if not graph.has_node(lbl):
                graph.add_node(lbl, **{ATTR_SIZE: 0, ATTR_IDS: []})
            nodes = graph.nodes()
            nodes[lbl][ATTR_SIZE] += 1
            nodes[lbl][ATTR_IDS].append(n)
    for lbls in itm_lbls:
        lbls_len = len(lbls)
        for i in range(lbls_len):
            source_lbl = lbls[i]
            for j in range(i + 1, lbls_len):
                target_lbl = lbls[j]
                if target_lbl not in graph[source_lbl]:
                    graph.add_edge(source_lbl, target_lbl)
    return graph


def aggregate_graph(X: ArrayLike, graph: nx.Graph, agg: Callable) -> Dict:
    """
    Apply an aggregation function to the nodes of a graph.

    This function takes a dataset and a graph, and computes an aggregation
    value for each node of the graph, based on the data points that are
    associated with that node. The aggregation function can be any callable
    that takes a list of values and returns a single value, such as `sum`,
    `mean`, `max`, `min`, etc.

    The function returns a dictionary that maps each node of the graph to its
    aggregation value. The keys of the dictionary are the nodes of the graph,
    and the values are the aggregation values.

    :param X: A dataset of n points.
    :param graph: The graph to apply the aggregation function to.
    :param agg: The aggregation function to use.
    :return: A dictionary of node-aggregation pairs.
    """
    agg_values = {}
    nodes = graph.nodes()
    for node_id in nodes:
        node_values = [X[i] for i in nodes[node_id][ATTR_IDS]]
        agg_value = agg(node_values)
        agg_values[node_id] = agg_value
    return agg_values


class SpatialSearch(Protocol):
    """
    Abstract interface for spatial search algorithms.

    This interface defines the methods that a spatial search algorithm must
    implement. A spatial search algorithm is used to find the nearest neighbors
    of a point in a dataset. The algorithm should be able to handle datasets of
    arbitrary size and dimensionality, and should be efficient in terms of both
    time and space complexity.
    """

    def fit(self, X: ArrayLike) -> SpatialSearch:
        """
        Fit the spatial search algorithm to the data.

        :param X: A dataset of n points.
        :return: self
        """

    def search(self, x: PointLike) -> List[int]:
        """
        Search for the nearest neighbors of a point.

        :param x: A point to search for.
        :return: A list of indices of the nearest neighbors of the point.
        """


class Cover(Protocol):
    """
    Abstract interface for cover algorithms.

    This interface defines the methods that a cover algorithm must implement.
    A cover algorithm is used to cover the space with overlapping open sets.
    The cover algorithm should be able to handle datasets of arbitrary size and
    dimensionality, and should be efficient in terms of both time and space
    complexity.
    """

    def fit(self, X: ArrayLike) -> Cover:
        """
        Fit the cover algorithm to the data.

        :param X: A dataset of n points.
        :return: self
        """

    def fit_transform(self, X: ArrayLike) -> Generator[List[int], None, None]:
        """
        Fit the cover algorithm to the data and transform it.

        This method should yield a generator of lists, where each list contains
        the indices of the points in the dataset that belong to the open set.

        :param X: A dataset of n points.
        :yield: A generator of lists of indices.
        """

    def transform(self, X: ArrayLike) -> Generator[List[int], None, None]:
        """
        Transform the data into overlapping open sets.

        This method should yield a generator of lists, where each list contains
        the indices of the points in the dataset that belong to the open set.

        :param X: A dataset of n points.
        :yield: A generator of lists of indices.
        """


class Clustering(Protocol):
    """
    Abstract interface for clustering algorithms.

    This interface defines the methods that a clustering algorithm must
    implement. A clustering algorithm is used to group the points in the
    dataset into clusters. The clustering algorithm should be able to handle
    datasets of arbitrary size and dimensionality, and should be efficient in
    terms of both time and space complexity.

    This interface is compatible with scikit-learn's clustering
    interface, typically from :mod:`sklearn.cluster`.
    The clustering algorithm should implement the `fit` method, which takes a
    dataset as input and returns the clustering labels for each point in the
    dataset. The labels should be stored in the `labels_` attribute of the
    clustering algorithm instance. The labels should be integers starting from
    zero, and should be unique for each cluster. Points that are not assigned to
    any cluster should have a label of -1 (this is typically the case for noise
    points in clustering algorithms like DBSCAN).
    """

    labels_: List[int]

    def fit(self, X: ArrayLike, y: Any = None) -> Clustering:
        """
        Fit the clustering algorithm to the data.

        :param X: A dataset of n points.
        :param y: Ignored.
        :return: self
        """


class TrivialCover(ParamsMixin):
    """
    Cover algorithm that covers data with a single subset containing the whole
    dataset.

    This class creates a single open set that contains all the points in the
    dataset.
    """

    def fit(self, X: ArrayLike) -> TrivialCover:
        return self

    def transform(self, X: ArrayLike) -> Generator[List[int], None, None]:
        yield list(range(len(X)))

    def fit_transform(self, X: ArrayLike) -> Generator[List[int], None, None]:
        self.fit(X)
        return self.transform(X)


class _MapperAlgorithm(EstimatorMixin, ParamsMixin):

    _cover: Cover
    _clustering: Clustering
    _verbose: bool
    _failsafe: bool
    _n_jobs: int
    graph_: nx.Graph

    def __init__(
        self,
        cover: Optional[Cover] = None,
        clustering: Optional[Clustering] = None,
        failsafe: bool = True,
        verbose: bool = True,
        n_jobs: int = 1,
    ):
        self.cover = cover
        self.clustering = clustering
        self.failsafe = failsafe
        self.verbose = verbose
        self.n_jobs = n_jobs

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None):
        X, y = self._validate_X_y(X, y)
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
        y = X if y is None else y
        self.graph_ = mapper_graph(
            X,
            y,
            self._cover,
            self._clustering,
            n_jobs=self._n_jobs,
        )
        self._set_n_features_in(X)
        return self

    def fit_transform(self, X: ArrayLike, y: ArrayLike) -> nx.Graph:
        self.fit(X, y)
        return self.graph_


class MapperAlgorithm(_MapperAlgorithm):
    """
    **DEPRECATED**: This class is deprecated and will be removed in a future
    release. Use :class:`tdamapper.learn.MapperAlgorithm`.
    """

    @deprecated(
        "This class is deprecated and will be removed in a future release. "
        "Use tdamapper.learn.MapperAlgorithm."
    )
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class FailSafeClustering(ParamsMixin):
    """
    A delegating clustering algorithm that prevents failure.

    This class wraps a clustering algorithm and handles any exceptions that may
    occur during the fitting process. If the clustering algorithm fails,
    instead of throwing an exception, a single cluster containing all points is
    returned. This can be useful for robustness and debugging purposes.

    :param clustering: A clustering algorithm to delegate to.
    :param verbose: A flag to log clustering exceptions. Set to True to
        enable logging, or False to suppress it. Defaults to True.
    """

    _clustering: Clustering
    _verbose: bool
    labels_: List[int]

    def __init__(self, clustering: Optional[Clustering] = None, verbose: bool = True):
        self.clustering = clustering
        self.verbose = verbose

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None):
        self._clustering = (
            TrivialClustering() if self.clustering is None else self.clustering
        )
        self._verbose = self.verbose
        self.labels_ = []
        try:
            self._clustering.fit(X, y)
            self.labels_ = self._clustering.labels_
        except ValueError as err:
            if self._verbose:
                _logger.warning("Unable to perform clustering on local chart: %s", err)
            self.labels_ = [0 for _ in X]
        return self


class TrivialClustering(ParamsMixin):
    """
    A clustering algorithm that returns a single cluster.

    This class implements a trivial clustering algorithm that assigns all data
    points to the same cluster. It can be used as an argument of the class
    :class:`tdamapper.core.MapperAlgorithm` to skip clustering in the
    construction of the Mapper graph.
    """

    labels_: List[int]

    def __init__(self):
        pass

    def fit(self, X: ArrayLike, _: Optional[ArrayLike] = None) -> TrivialClustering:
        """
        Fit the clustering algorithm to the data.

        :param X: A dataset of n points.
        :param y: Ignored.
        :return: self
        """
        self.labels_ = [0 for _ in X]
        return self
