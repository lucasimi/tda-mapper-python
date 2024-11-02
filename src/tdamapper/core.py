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

import logging
import networkx as nx

from tdamapper.cover import TrivialCover
from tdamapper.utils.unionfind import UnionFind
from tdamapper._common import ParamsMixin


ATTR_IDS = 'ids'
ATTR_SIZE = 'size'


_logger = logging.getLogger(__name__)

logging.basicConfig(
    format='%(asctime)s %(module)s %(levelname)s: %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler()
    ]
)


def mapper_labels(X, y, cover, clustering):
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

    :param X: The dataset to be mapped.
    :type X: array-like of shape (n, m) or list-like of length n
    :param y: The lens values for each point in the dataset.
    :type y: array-like of shape (n, k) or list-like of length n
    :param cover: The cover algorithm to apply to lens space.
    :type cover: A class compatible with :class:`tdamapper.cover.Cover`
    :param clustering: The clustering algorithm to apply to each subset of the
        dataset.
    :type clustering: A class from :mod:`tdamapper.clustering`, or a class from
        :mod:`sklearn.cluster`
    :return: A list of node labels for each point in the dataset.
    :rtype: list[list[int]]
    """
    itm_lbls = [[] for _ in X]
    max_lbl = 0
    for local_ids in cover.apply(y):
        local_lbls = clustering.fit([X[j] for j in local_ids]).labels_
        max_local_lbl = 0
        for local_id, local_lbl in zip(local_ids, local_lbls):
            if local_lbl >= 0:
                itm_lbls[local_id].append(max_lbl + local_lbl)
            if local_lbl > max_local_lbl:
                max_local_lbl = local_lbl
        max_lbl += max_local_lbl + 1
    return itm_lbls


def mapper_connected_components(X, y, cover, clustering):
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

    :param X: The dataset to be mapped.
    :type X: array-like of shape (n, m) or list-like of length n
    :param y: The lens values for each point in the dataset.
    :type y: array-like of shape (n, k) or list-like of length n
    :param cover: The cover algorithm to apply to lens space.
    :type cover: A class compatible with :class:`tdamapper.cover.Cover`
    :param clustering: The clustering algorithm to apply to each subset of the
        dataset.
    :type clustering: A class from :mod:`tdamapper.clustering`, or a class from
        :mod:`sklearn.cluster`
    :return: A list of labels. The label at position i identifies the connected
        component of the point at position i in the dataset.
    :rtype: list[int]
    """
    itm_lbls = mapper_labels(X, y, cover, clustering)
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


def mapper_graph(X, y, cover, clustering):
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

    :param X: The dataset to be mapped.
    :type X: array-like of shape (n, m) or list-like of length n
    :param y: The lens values for each point in the dataset.
    :type y: array-like of shape (n, k) or list-like of length n
    :param cover: The cover algorithm to apply to lens space.
    :type cover: A class compatible with :class:`tdamapper.cover.Cover`
    :param clustering: The clustering algorithm to apply to each subset of the
        dataset.
    :type clustering: A class from :mod:`tdamapper.clustering`, or a class from
        :mod:`sklearn.cluster`
    :return: The Mapper graph.
    :rtype: :class:`networkx.Graph`
    """
    itm_lbls = mapper_labels(X, y, cover, clustering)
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


def aggregate_graph(X, graph, agg):
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

    :param X: The dataset to be aggregated.
    :type X: array-like of shape (n, m) or list-like
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


class MapperAlgorithm(ParamsMixin):
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

    :param cover: The cover algorithm to apply to lens space.
    :type cover: A class compatible with :class:`tdamapper.cover.Cover`
    :param clustering: The clustering algorithm to apply to each subset of the
        dataset. If no clustering is specified,
        :class:`tdamapper.core.TrivialClustering` is used, which produces a
        single cluster for each subset. Defaults to None.
    :type clustering: A class from :mod:`tdamapper.clustering`, or a class from
        :mod:`sklearn.cluster`, optional
    :param failsafe: A flag that is used to prevent failures. If True, the
        clustering object is wrapped by
        :class:`tdamapper.core.FailSafeClustering`. Defaults to True.
    :type failsafe: bool, optional
    :param verbose: A flag that is used for logging, supplied to
        :class:`tdamapper.core.FailSafeClustering`. If True, clustering
        failures are logged. Set to False to suppress these messages. Defaults
        to True.
    :type verbose: bool, optional
    """

    def __init__(self, cover=None, clustering=None, failsafe=True, verbose=True):
        self.cover = cover
        self.clustering = clustering
        self.failsafe = failsafe
        self.verbose = verbose

    def fit(self, X, y=None):
        """
        Create the Mapper graph and store it for later use.

        This method stores the result of :func:`tdamapper.core.mapper_graph` in
        the attribute `graph_` and returns a reference to the calling object.

        :param X: The dataset to be mapped.
        :type X: array-like of shape (n, m) or list-like of length n
        :param y: The lens values for each point in the dataset.
        :type y: array-like of shape (n, k) or list-like of length n
        :return: The object itself.
        """
        self.__cover = TrivialCover() if self.cover is None \
            else self.cover
        self.__clustering = TrivialClustering() if self.clustering is None \
            else self.clustering
        self.__verbose = self.verbose
        self.__failsafe = self.failsafe
        if self.__failsafe:
            self.__clustering = FailSafeClustering(
                clustering=self.__clustering,
                verbose=self.__verbose
            )
        y = X if y is None else y
        self.graph_ = mapper_graph(X, y, self.__cover, self.__clustering)
        return self

    def fit_transform(self, X, y):
        """
        Create the Mapper graph.

        This method is equivalent to calling
        :func:`tdamapper.core.mapper_graph`.

        :param X: The dataset to be mapped.
        :type X: array-like of shape (n, m) or list-like of length n
        :param y: The lens values for each point in the dataset.
        :type y: array-like of shape (n, k) or list-like of length n
        :return: The Mapper graph.
        :rtype: :class:`networkx.Graph`
        """
        self.fit(X, y)
        return self.graph_


class FailSafeClustering(ParamsMixin):
    """
    A delegating clustering algorithm that prevents failure.

    This class wraps a clustering algorithm and handles any exceptions that may
    occur during the fitting process. If the clustering algorithm fails,
    instead of throwing an exception, a single cluster containing all points is
    returned. This can be useful for robustness and debugging purposes.

    :param clustering: A clustering algorithm to delegate to.
    :type clustering: Anything compatible with a :mod:`sklearn.cluster` class.
    :param verbose: A flag to log clustering exceptions. Set to True to
        enable logging, or False to suppress it. Defaults to True.
    :type verbose: bool, optional.
    """

    def __init__(self, clustering=None, verbose=True):
        self.clustering = clustering
        self.verbose = verbose

    def fit(self, X, y=None):
        self.__clustering = TrivialClustering() if self.clustering is None \
            else self.clustering
        self.__verbose = self.verbose
        self.labels_ = None
        try:
            self.__clustering.fit(X, y)
            self.labels_ = self.__clustering.labels_
        except ValueError as err:
            if self.__verbose:
                _logger.warning('Unable to perform clustering on local chart: %s', err)
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

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """
        Fit the clustering algorithm to the data.

        :param X: The dataset to be mapped.
        :type X: array-like of shape (n, m) or list-like of length n
        :param y: Ignored.
        :return: self
        """
        self.labels_ = [0 for _ in X]
        return self
