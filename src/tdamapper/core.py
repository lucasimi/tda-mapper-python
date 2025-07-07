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
from typing import Any, Callable, Generator, Optional, Protocol

import networkx as nx
from joblib import Parallel, delayed

from tdamapper._common import ArrayLike, ParamsMixin, clone
from tdamapper.utils.unionfind import UnionFind

ATTR_IDS = "ids"

ATTR_SIZE = "size"


_logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s %(module)s %(levelname)s: %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)


def mapper_labels(
    X: ArrayLike,
    y: ArrayLike,
    cover: Cover,
    clustering: Clustering,
    n_jobs: int = 1,
) -> list[list[int]]:
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
    :type X: array-like of shape (n, m) or list-like of length n
    :param y: Lens values for the n points of the dataset.
    :type y: array-like of shape (n, k) or list-like of length n
    :param cover: A cover algorithm.
    :type cover: A class compatible with :class:`tdamapper.core.Cover`
    :param clustering: A clustering algorithm.
    :type clustering: An estimator compatible with scikit-learn's clustering
        interface, typically from :mod:`sklearn.cluster`.
    :param n_jobs: The maximum number of parallel clustering jobs. This
        parameter is passed to the constructor of :class:`joblib.Parallel`.
        Defaults to 1.
    :type n_jobs: int
    :return: A list of node labels for each point in the dataset.
    :rtype: list[list[int]]
    """

    def _run_clustering(local_ids, X_local, clust):
        local_lbls = clust.fit(X_local).labels_
        return local_ids, local_lbls

    _lbls = Parallel(n_jobs, prefer="threads")(
        delayed(_run_clustering)(
            local_ids, [X[j] for j in local_ids], clone(clustering)
        )
        for local_ids in cover.apply(y)
    )
    itm_lbls: list[list[int]] = [[] for _ in X]
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
) -> list[int]:
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
    :type X: array-like of shape (n, m) or list-like of length n
    :param y: Lens values for the n points of the dataset.
    :type y: array-like of shape (n, k) or list-like of length n
    :param cover: A cover algorithm.
    :type cover: A class compatible with :class:`tdamapper.core.Cover`
    :param clustering: The clustering algorithm to apply to each subset of the
        dataset.
    :type clustering: An estimator compatible with scikit-learn's clustering
        interface, typically from :mod:`sklearn.cluster`.
    :param n_jobs: The maximum number of parallel clustering jobs. This
        parameter is passed to the constructor of :class:`joblib.Parallel`.
        Defaults to 1.
    :type n_jobs: int
    :return: A list of labels. The label at position i identifies the connected
        component of the point at position i in the dataset.
    :rtype: list[int]
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
    :type X: array-like of shape (n, m) or list-like of length n
    :param y: Lens values for the n points of the dataset.
    :type y: array-like of shape (n, k) or list-like of length n
    :param cover: A cover algorithm.
    :type cover: A class compatible with :class:`tdamapper.core.Cover`
    :param clustering: The clustering algorithm to apply to each subset of the
        dataset.
    :type clustering: An estimator compatible with scikit-learn's clustering
        interface, typically from :mod:`sklearn.cluster`.
    :param n_jobs: The maximum number of parallel clustering jobs. This
        parameter is passed to the constructor of :class:`joblib.Parallel`.
        Defaults to 1.
    :type n_jobs: int
    :return: The Mapper graph.
    :rtype: :class:`networkx.Graph`
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


def aggregate_graph(X: ArrayLike, graph: nx.Graph, agg: Callable) -> dict[Any, Any]:
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
    :type X: array-like of shape (n, m) or list-like of length n
    :param graph: The graph to apply the aggregation function to.
    :type graph: :class:`networkx.Graph`.
    :param agg: The aggregation function to use.
    :type agg: Callable.
    :return: A dictionary of node-aggregation pairs.
    :rtype: dict
    """
    agg_values = {}
    nodes = graph.nodes()
    for node_id in nodes:
        node_values = [X[i] for i in nodes[node_id][ATTR_IDS]]
        agg_value = agg(node_values)
        agg_values[node_id] = agg_value
    return agg_values


class Cover(Protocol):
    """
    Abstract interface for cover algorithms.
    """

    def apply(self, X: ArrayLike) -> Generator[list[int]]:
        """
        Covers the dataset with open set.

        :param X: A dataset of n points.
        :type X: array-like of shape (n, m) or list-like of length n
        :return: A generator of lists of ids.
        :rtype: generator of lists of ints
        """


class Clustering(Protocol):
    """
    Abstract interface for clustering algorithms.
    """

    labels_: list[int]

    def fit(self, X: ArrayLike, y: Any = None) -> Clustering:
        """
        Fit the clustering algorithm to the data.

        :param X: A dataset of n points.
        :type X: array-like of shape (n, m) or list-like of length n
        :param y: Ignored.
        :return: The object itself.
        :rtype: self
        """


class SpatialSearch(Protocol):
    """
    Abstract interface for spatial search algorithms.
    A spatial search algorithm is a function that returns a list of neighbors
    for a given query point. The neighbors are points in the dataset that are
    close to the query point, according to some distance metric.
    """

    def fit(self, X: ArrayLike) -> SpatialSearch:
        """
        Train internal parameters.

        :param X: A dataset of n points.
        :type X: array-like of shape (n, m) or list-like of length n
        :return: The object itself.
        :rtype: self
        """

    def search(self, x: Any) -> list[int]:
        """
        Return a list of neighbors for the query point.

        :param x: A query point for which we want to find neighbors.
        :type x: Any
        :return: A list containing the indices of the points in the dataset
            that are neighbors of the query point.
        :rtype: list[int]
        """


def proximity_net(search: SpatialSearch, X: ArrayLike) -> Generator[list[int]]:
    """
    Create a proximity-net for the dataset.

    This function applies an iterative algorithm to create the proximity-net.
    It picks an arbitrary point and forms an open cover calling the proximity
    function on the chosen point. The points contained in the open cover are
    then marked as covered, and discarded in the following steps. The procedure
    is repeated on the leftover points until every point is eventually covered.

    :param search: A spatial search algorithm.
    :type search: A class compatible with :class:`tdamapper.core.SpatialSearch`
    :param X: A dataset of n points.
    :type X: array-like of shape (n, m) or list-like of length n
    :return: A generator of lists of ids.
    :rtype: generator of lists of ints
    """
    covered_ids = set()
    search.fit(X)
    for i, xi in enumerate(X):
        if i not in covered_ids:
            neigh_ids = search.search(xi)
            covered_ids.update(neigh_ids)
            if neigh_ids:
                yield neigh_ids


class Proximity(SpatialSearch):

    def apply(self, X: ArrayLike) -> Generator[list[int]]:
        """
        Create a proximity-net for the dataset.

        This method is an alias for the `proximity_net` function. It allows
        using the Proximity class as a spatial search algorithm.

        :param X: A dataset of n points.
        :type X: array-like of shape (n, m) or list-like of length n
        :return: A generator of lists of ids.
        :rtype: generator of lists of ints
        """
        return proximity_net(self, X)


class TrivialCover(ParamsMixin):
    """
    Cover algorithm that covers data with a single subset containing the whole
    dataset.

    This class creates a single open set that contains all the points in the
    dataset.
    """

    def apply(self, X: ArrayLike) -> Generator[list[int]]:
        """
        Covers the dataset with a single open set.

        :param X: A dataset of n points.
        :type X: array-like of shape (n, m) or list-like of length n
        :return: A generator of lists of ids.
        :rtype: generator of lists of ints
        """
        yield list(range(0, len(X)))


class FailSafeClustering(ParamsMixin):
    """
    A delegating clustering algorithm that prevents failure.

    This class wraps a clustering algorithm and handles any exceptions that may
    occur during the fitting process. If the clustering algorithm fails,
    instead of throwing an exception, a single cluster containing all points is
    returned. This can be useful for robustness and debugging purposes.

    :param clustering: A clustering algorithm to delegate to.
    :type clustering: An estimator compatible with scikit-learn's clustering
        interface, typically from :mod:`sklearn.cluster`.
    :param verbose: A flag to log clustering exceptions. Set to True to
        enable logging, or False to suppress it. Defaults to True.
    :type verbose: bool, optional.
    """

    _clustering: Clustering
    _verbose: bool
    labels_: list[int]

    def __init__(
        self,
        clustering: Optional[Clustering] = None,
        verbose: bool = True,
    ):
        self.clustering = clustering
        self.verbose = verbose

    def fit(self, X: ArrayLike, y: Any = None) -> FailSafeClustering:
        """
        Fit the clustering algorithm to the data.

        This method attempts to fit the clustering algorithm to the data. If it
        fails, it logs a warning (if verbose is True) and assigns all points to
        a single cluster (label 0).

        :param X: A dataset of n points.
        :type X: array-like of shape (n, m) or list-like of length n
        :param y: Ignored.
        :return: The object itself.
        :rtype: self
        """
        self._clustering = (
            TrivialClustering() if self.clustering is None else self.clustering
        )
        self._verbose = self.verbose
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

    labels_: list[int]

    def __init__(self):
        pass

    def fit(self, X: ArrayLike, _y: Any = None) -> TrivialClustering:
        """
        Fit the clustering algorithm to the data.

        :param X: A dataset of n points.
        :type X: array-like of shape (n, m) or list-like of length n
        :param _y: Ignored.
        :return: self
        """
        self.labels_ = [0 for _ in X]
        return self
