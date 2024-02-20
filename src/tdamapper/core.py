"""
Core tools for creating and analyzing Mapper graphs.

The Mapper algorithm is a method for exploring the shape and structure of
high-dimensional datasets, by constructing a graph representation called Mapper
graph. The algorithm has three main steps:

1. **Filtering**: Apply a lens function (also called filter) to map the data points
   to a lower-dimensional space, such as a scalar value or a 2D plane.

2. **Covering**: Arrange the lens space into overlapping open sets, using a
   cover algorithm such as uniform intervals or balls.

3. **Clustering**: Group the data points in each open set into clusters, using a
   clustering algorithm such as single-linkage or DBSCAN.

The Mapper graph consists of nodes that represent clusters of data
points, and edges that connect overlapping clusters (clusters obtained from
different open sets can possibly overlap). For more details on the Mapper
algorithm and its applications, see

    Gurjeet Singh, Facundo Mémoli and Gunnar Carlsson, "Topological Methods for
    the Analysis of High Dimensional Data Sets and 3D Object Recognition",
    Eurographics Symposium on Point-Based Graphics, 2007.

This module provides the class :class:`tdamapper.core.MapperAlgorithm`, which
encapsulates the algorithm and its parameters. The Mapper graph produced by this
module is a NetworkX graph object.
"""

import networkx as nx

from tdamapper.utils.unionfind import UnionFind


ATTR_IDS = 'ids'
ATTR_SIZE = 'size'


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
    are no duplicates. If i < j, the labels at position i are strictly less than
    those at position j.

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

    A connected component is a maximal set of nodes that are reachable from each
    other by following the edges. This function assigns a unique integer label
    to each point in the dataset, based on the connected component of the Mapper
    graph that it belongs to.

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

    This function takes a dataset and a graph, and computes an aggregation value
    for each node of the graph, based on the data points that are associated
    with that node. The aggregation function can be any callable that takes a
    list of values and returns a single value, such as `sum`, `mean`, `max`,
    `min`, etc.

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


class MapperAlgorithm:
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
        dataset.
    :type clustering: A class from :mod:`tdamapper.clustering`, or a class from
        :mod:`sklearn.cluster`
    """

    def __init__(self, cover, clustering):
        self.__cover = cover
        self.__clustering = clustering
        self.graph_ = None

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
        self.graph_ = self.fit_transform(X, y)
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
        return mapper_graph(X, y, self.__cover, self.__clustering)
