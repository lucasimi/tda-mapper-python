"""This module implements the main tools for creating and analyzing Mapper graphs.

The Mapper graph is a simplified representation of the shape and structure of the data,
composed of nodes and edges. Each node corresponds to a cluster of points having similar
values in the lens space. Each edge connects two nodes that share some points in their
corresponding clusters.

The Mapper algorithm consists of three main steps: filtering, covering, and clustering.
First, the data points are mapped to a lower-dimensional space using a lens function.
Then, the lens space is covered by overlapping open sets, using an open cover algorithm.
Finally, the data points in each open set are clustered using a clustering algorithm,
and the clusters are connected by edges if they share points in the overlap.

The module provides a class that encapsulates the algorithm and its parameters,
and a fit_transform method that takes a dataset and returns a NetworkX graph object.
For more details on the Mapper algorithm and its applications, see

    Gurjeet Singh, Facundo Mémoli and Gunnar Carlsson,
    "Topological Methods for the Analysis of High Dimensional Data Sets and 3D Object Recognition",
    Eurographics Symposium on Point-Based Graphics, 2007

"""

import networkx as nx

from tdamapper.utils.unionfind import UnionFind


ATTR_IDS = 'ids'
ATTR_SIZE = 'size'


def mapper_labels(X, y, cover, clustering):
    """Identify the nodes of the Mapper graph.

    The function first covers the lens space with overlapping sets, using the cover
    algorithm provided. Then, for each set, it clusters the points of the dataset
    that have lens values within that set, using the clustering algorithm provided.
    The clusters are labeled with unique integers, starting from zero for each set.
    The function then adds an offset to the cluster labels, such that the labels are
    distinct across all sets. The offset is equal to the maximum label of the
    previous set plus one.

    The function returns a list of node labels for each point in the dataset. The list at
    position i contains the labels of the nodes that the point at position i belongs to.
    The labels are sorted in ascending order, and there are no duplicates. If i < j, the
    labels at position i are strictly less than those at position j.

    :param X: The dataset to be mapped.
    :type X: array-like of shape (n_samples, n_features) or list-like
    :param y: The lens values for each point in the dataset.
    :type y: array-like of shape (n_samples,) or list-like
    :param cover: The cover algorithm to apply to the lens.
    :type cover: An instance of a cover algorithm from `tdamapper.cover`.
    :param clustering: The clustering algorithm to apply to each subset of the dataset.
    :type clustering: A class from `tdamapper.clustering` or a class from `sklearn.cluster`.
    :return: A list of node labels for each point in the dataset.
    :rtype: `list[list[int]]`

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
    """Identify the connected components of the Mapper graph.

    This function assigns a unique integer label to each point in the dataset, based on
    the connected component of the Mapper graph that it belongs to. A connected component
    is a maximal set of nodes that are reachable from each other by following the edges.

    The function uses a union-find data structure to efficiently keep track of the
    connected components as it scans the nodes of the Mapper graph. This approach
    should be faster than computing the Mapper graph by first calling `mapper_graph`
    and then calling `networkx.connected_components` on it.

    :param X: The dataset to be mapped.
    :type X: array-like of shape (n_samples, n_features) or list-like
    :param y: The lens values for each point in the dataset.
    :type y: array-like of shape (n_samples,) or list-like
    :param cover: The cover algorithm to apply to the lens.
    :type cover: An instance of a cover algorithm from `tdamapper.cover`.
    :param clustering: The clustering algorithm to apply to each subset of the dataset.
    :type clustering: A class from `tdamapper.clustering` or a class from `sklearn.cluster`.
    :return: A list of labels. The label at position i identifies the connected component
        of the point at position i in the dataset.
    :rtype: `list[int]`

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
    """Create the Mapper graph.

    This function creates a Mapper graph, which is a simplified representation of the
    shape and structure of the data, composed of nodes and edges. Each node corresponds
    to a cluster of points having similar values in the lens space. Each edge connects
    two nodes that share some points in their corresponding clusters.

    The function first covers the lens space with overlapping sets, using the cover
    algorithm provided. Then, for each set, it clusters the points of the dataset
    that have lens values within that set, using the clustering algorithm provided.
    The clusters are labeled with unique integers, and the nodes of the Mapper graph are
    created from these labels. The edges of the Mapper graph are created by checking if
    any two nodes share some points in their clusters.

    The function returns a networkx.Graph object that represents the Mapper graph. The
    node of the graph are identified by the integers corresponding to cluster labels.
    Each node has an attribute 'size' that stores the number of points contained in its
    corresponding cluster, and an attribute 'ids' that stores the indices of the points
    in the dataset that are contained in the cluster.

    :param X: The dataset to be mapped.
    :type X: array-like of shape (n_samples, n_features) or list-like
    :param y: The lens values for each point in the dataset.
    :type y: array-like of shape (n_samples,) or list-like
    :param cover: The cover algorithm to apply to the lens.
    :type cover: An instance of a cover algorithm from `tdamapper.cover`.
    :param clustering: The clustering algorithm to apply to each subset of the dataset.
    :type clustering: A class from `tdamapper.clustering` or a class from `sklearn.cluster`.
    :return: The Mapper graph.
    :rtype: `networkx.Graph`

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


def aggregate_graph(y, graph, agg):
    """Apply an aggregation function to the nodes of a graph.

    This function takes a dataset and a graph, and computes an aggregation value for each
    node of the graph, based on the data points that are associated with that node. The
    aggregation function can be any callable that takes a list of values and returns a
    single value, such as `sum`, `mean`, `max`, `min`, etc.

    The function returns a dictionary that maps each node of the graph to its aggregation
    value. The keys of the dictionary are the nodes of the graph, and the values are the
    aggregation values.

    :param y: The dataset to be aggregated.
    :type y: array-like of shape (n_samples,) or list-like
    :param graph: The graph to apply the aggregation function to.
    :type graph: `networkx.Graph`.
    :param agg: The aggregation function to use.
    :type agg: Callable.
    :return: A dictionary of node-aggregation pairs.
    :rtype: `dict`

    """
    agg_values = {}
    nodes = graph.nodes()
    for node_id in nodes:
        node_values = [y[i] for i in nodes[node_id][ATTR_IDS]]
        agg_value = agg(node_values)
        agg_values[node_id] = agg_value
    return agg_values


class MapperAlgorithm:
    """A class for creating and analyzing Mapper graphs.

    :param cover: The cover algorithm to apply to the lens.
    :type cover: An instance of a cover algorithm from `tdamapper.cover`.
    :param clustering: The clustering algorithm to apply to each subset of the dataset.
    :type clustering: A class from `tdamapper.clustering` or a class from `sklearn.cluster`.

    """

    def __init__(self, cover, clustering):
        self.__cover = cover
        self.__clustering = clustering
        self.graph_ = None

    def fit(self, X, y=None):
        """Create the Mapper graph.

        This method creates a Mapper graph, which is a simplified representation of the
        shape and structure of the data, composed of nodes and edges. Each node corresponds
        to a cluster of points having similar values in the lens space. Each edge connects
        two nodes that share some points in their corresponding clusters.

        This method stores the result of `mapper_graph` in the attribute `graph_` and
        returns a reference to the calling object.

        :param X: The dataset to be mapped.
        :type X: array-like of shape (n_samples, n_features) or list-like
        :param y: The lens values for each point in the dataset.
        :type y: array-like of shape (n_samples,) or list-like
        :return: `self`.

        """
        self.graph_ = self.fit_transform(X, y)
        return self

    def fit_transform(self, X, y):
        """Create the Mapper graph.

        This method creates a Mapper graph, which is a simplified representation of the
        shape and structure of the data, composed of nodes and edges. Each node corresponds
        to a cluster of points having similar values in the lens space. Each edge connects
        two nodes that share some points in their corresponding clusters.

        This method is equivalent to calling `mapper_graph`.

        :param X: The dataset to be mapped.
        :type X: array-like of shape (n_samples, n_features) or list-like
        :param y: The lens values for each point in the dataset.
        :type y: array-like of shape (n_samples,) or list-like
        :return: The Mapper graph.
        :rtype: `networkx.Graph`

        """
        return mapper_graph(X, y, self.__cover, self.__clustering)
