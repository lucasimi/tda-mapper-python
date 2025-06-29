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
from typing import Callable, Dict, Generator, List, Optional, Protocol

import networkx as nx
from joblib import Parallel, delayed

from tdamapper._common import (
    ArrayLike,
    EstimatorMixin,
    ParamsMixin,
    PointLike,
    clone,
    deprecated,
)
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


def mapper_labels(
    x_arr: ArrayLike,
    y_arr: ArrayLike,
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

    :param x_arr: A dataset of n points.
    :param y_arr: Lens values for the n points of the dataset.
    :param cover: A cover algorithm.
    :param clustering: A clustering algorithm.
    :param n_jobs: The maximum number of parallel clustering jobs. This
        parameter is passed to the constructor of :class:`joblib.Parallel`.
        Defaults to 1.
    :return: A list of node labels for each point in the dataset.
    """

    def _run_clustering(local_ids, x_arr_local, clust):
        local_lbls = clust.fit(x_arr_local).labels_
        return local_ids, local_lbls

    _lbls = Parallel(n_jobs, prefer="threads")(
        delayed(_run_clustering)(
            local_ids, [x_arr[j] for j in local_ids], clone(clustering)
        )
        for local_ids in cover.fit_transform(y_arr)
    )
    itm_lbls: List[List[int]] = [[] for _ in x_arr]
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
    x_arr: ArrayLike,
    y_arr: ArrayLike,
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

    :param x_arr: A dataset of n points.
    :param y_arr: Lens values for the n points of the dataset.
    :param cover: A cover algorithm.
    :param clustering: The clustering algorithm to apply to each subset of the
        dataset.
    :param n_jobs: The maximum number of parallel clustering jobs. This
        parameter is passed to the constructor of :class:`joblib.Parallel`.
        Defaults to 1.
    :return: A list of labels. The label at position i identifies the connected
        component of the point at position i in the dataset.
    """
    itm_lbls = mapper_labels(x_arr, y_arr, cover, clustering, n_jobs=n_jobs)
    label_values = set()
    for lbls in itm_lbls:
        label_values.update(lbls)
    uf = UnionFind(label_values)
    for lbls in itm_lbls:
        if len(lbls) > 1:
            for first, second in zip(lbls, lbls[1:]):
                uf.union(first, second)
    labels = [-1 for _ in x_arr]
    for i, lbls in enumerate(itm_lbls):
        # assign -1 to noise points
        root = uf.find(lbls[0]) if lbls else -1
        labels[i] = root
    return labels


def mapper_graph(
    x_arr: ArrayLike,
    y_arr: ArrayLike,
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

    :param x_arr: A dataset of n points.
    :param y_arr: Lens values for the n points of the dataset.
    :param cover: A cover algorithm.
    :param clustering: The clustering algorithm to apply to each subset of the
        dataset.
    :param n_jobs: The maximum number of parallel clustering jobs. This
        parameter is passed to the constructor of :class:`joblib.Parallel`.
        Defaults to 1.
    :return: The Mapper graph.
    """
    itm_lbls = mapper_labels(x_arr, y_arr, cover, clustering, n_jobs=n_jobs)
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


def aggregate_graph(arr: ArrayLike, graph: nx.Graph, agg: Callable) -> Dict:
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

    :param x_arr: A dataset of n points.
    :param graph: The graph to apply the aggregation function to.
    :param agg: The aggregation function to use.
    :return: A dictionary of node-aggregation pairs.
    """
    agg_values = {}
    nodes = graph.nodes()
    for node_id in nodes:
        node_values = [arr[i] for i in nodes[node_id][ATTR_IDS]]
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

    def fit(self, arr: ArrayLike) -> SpatialSearch:
        """
        Fit the spatial search algorithm to the data.

        :param arr: A dataset of n points.
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

    def fit(self, arr: ArrayLike) -> Cover:
        """
        Fit the cover algorithm to the data.

        :param arr: A dataset of n points.
        :return: self
        """

    def fit_transform(self, arr: ArrayLike) -> Generator[List[int], None, None]:
        """
        Fit the cover algorithm to the data and transform it.

        This method should yield a generator of lists, where each list contains
        the indices of the points in the dataset that belong to the open set.

        :param arr: A dataset of n points.
        :yield: A generator of lists of indices.
        """

    def transform(self, arr: ArrayLike) -> Generator[List[int], None, None]:
        """
        Transform the data into overlapping open sets.

        This method should yield a generator of lists, where each list contains
        the indices of the points in the dataset that belong to the open set.

        :param arr: A dataset of n points.
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

    def fit(self, x_arr: ArrayLike, y_arr: Optional[ArrayLike] = None) -> Clustering:
        """
        Fit the clustering algorithm to the data.

        :param x_arr: A dataset of n points.
        :param y_arr: Ignored.
        :return: self
        """


class TrivialCover(ParamsMixin):
    """
    Cover algorithm that covers data with a single subset containing the whole
    dataset.

    This class creates a single open set that contains all the points in the
    dataset.
    """

    def fit(self, _x_arr: ArrayLike) -> TrivialCover:
        """
        Fit the cover algorithm to the data.

        :param _x_arr: A dataset of n points.
        :return: self
        """
        return self

    def transform(self, x_arr: ArrayLike) -> Generator[List[int], None, None]:
        """
        Transform the data into overlapping open sets.

        This method yields a generator that produces a single list containing
        the indices of all points in the dataset.

        :param x_arr: A dataset of n points.
        :yield: A generator yielding a single list of indices.
        """
        if len(x_arr) > 0:
            yield list(range(len(x_arr)))

    def fit_transform(self, x_arr: ArrayLike) -> Generator[List[int], None, None]:
        """
        Fit the cover algorithm to the data and transform it.

        This method fits the cover algorithm to the data and then yields a
        generator that produces a single list containing the indices of all
        points in the dataset.

        :param x_arr: A dataset of n points.
        :return: A generator yielding a single list of indices.
        """
        self.fit(x_arr)
        return self.transform(x_arr)


class _MapperAlgorithm(EstimatorMixin, ParamsMixin):
    """
    Mapper algorithm for constructing Mapper graphs.

    This class implements the Mapper algorithm for constructing Mapper graphs
    from a dataset. It allows the user to specify the cover and clustering
    algorithms to use, as well as various parameters such as verbosity and
    failsafe behavior. The Mapper graph is constructed by first covering the
    lens space with overlapping open sets, then clustering the points in each
    open set, and finally connecting the clusters that share points.

    The Mapper graph is represented as a NetworkX graph object, where each node
    corresponds to a cluster of points, and edges connect nodes that share
    points. The nodes have attributes 'size' (the number of points in the
    cluster) and 'ids' (the indices of the points in the dataset that belong to
    the cluster).

    :param cover: A cover algorithm to use for covering the lens space.
        If None, a trivial cover that contains all points in a single open set
        is used. Defaults to None.
    :param clustering: A clustering algorithm to use for clustering the points
        in each open set. If None, a trivial clustering that assigns all points
        to a single cluster is used. Defaults to None.
    :param failsafe: A flag to enable failsafe behavior. If True, the clustering
        algorithm is wrapped in a failsafe clustering that prevents failure by
        returning a single cluster containing all points if the clustering
        algorithm fails. Defaults to True.
    :param verbose: A flag to enable verbose logging. If True, the algorithm
        logs information about the clustering process. Defaults to True.
    :param n_jobs: The maximum number of parallel clustering jobs. This
        parameter is passed to the constructor of :class:`joblib.Parallel`.
        Defaults to 1.
    """

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

    def fit(self, x_arr: ArrayLike, y_arr: Optional[ArrayLike] = None):
        """
        Fit the Mapper algorithm to the data.

        This method fits the Mapper algorithm to the provided dataset and
        constructs the Mapper graph. It uses the cover and clustering algorithms
        specified in the constructor to cover the lens space and cluster the
        points in each open set.

        :param x_arr: A dataset of n points.
        :param y_arr: Lens values for the n points of the dataset. If None,
            the lens values are assumed to be the same as the dataset points.
            Defaults to None.
        :return: self
        :raises ValueError: If the input arrays are not valid or if the cover or
            clustering algorithms are not set properly.
        """
        y_arr_ = x_arr if y_arr is None else y_arr
        x_arr_, y_arr_ = self._validate_x_y(x_arr, y_arr_)
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
            x_arr_,
            y_arr_,
            self._cover,
            self._clustering,
            n_jobs=self._n_jobs,
        )
        self._set_n_features_in(x_arr_)
        return self

    def fit_transform(self, x_arr: ArrayLike, y_arr: Optional[ArrayLike]) -> nx.Graph:
        """
        Fit the Mapper algorithm to the data and return the Mapper graph.

        This method fits the Mapper algorithm to the provided dataset and
        returns the Mapper graph as a NetworkX graph object. The graph is built
        using the cover and clustering algorithms specified in the constructor.

        :param x_arr: A dataset of n points.
        :param y_arr: Lens values for the n points of the dataset.
        :return: The Mapper graph as a NetworkX graph object.
        """
        self.fit(x_arr, y_arr)
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

    def fit(self, x_arr: ArrayLike, y_arr: Optional[ArrayLike] = None):
        """
        Fit the clustering algorithm to the data.

        This method attempts to fit the clustering algorithm to the provided
        dataset. If the clustering algorithm raises a ValueError, it logs a
        warning message and assigns all points to a single cluster (label 0).

        :param x_arr: A dataset of n points.
        :param y_arr: Ignored.
        :return: self
        """
        self._clustering = (
            TrivialClustering() if self.clustering is None else self.clustering
        )
        self._verbose = self.verbose
        self.labels_ = []
        try:
            self._clustering.fit(x_arr, y_arr)
            self.labels_ = self._clustering.labels_
        except ValueError as err:
            if self._verbose:
                _logger.warning("Unable to perform clustering on local chart: %s", err)
            self.labels_ = [0 for _ in x_arr]
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

    def fit(
        self, x_arr: ArrayLike, _y_arr: Optional[ArrayLike] = None
    ) -> TrivialClustering:
        """
        Fit the clustering algorithm to the data.

        :param x_arr: A dataset of n points.
        :param _y_arr: Ignored.
        :return: self
        """
        self.labels_ = [0 for _ in x_arr]
        return self
