'''A module containing the main implementation logic for the Mapper algorithm.'''

import networkx as nx

from tdamapper.utils.unionfind import UnionFind


ATTR_IDS = 'ids'
ATTR_SIZE = 'size'


def mapper_labels(X, y, cover, clustering):
    '''
    Computes the open cover, then perform local clustering on each open set from the cover.

    :param X: A dataset.
    :type X: `numpy.ndarray` or list-like.
    :param y: Lens values.
    :type y: `numpy.ndarray` or list-like.
    :param cover: A cover algorithm.
    :type cover: A class from `tdamapper.cover`.
    :param clustering: A clustering algorithm.
    :type clustering: A class from `tdamapper.clustering` or a class from `sklearn.cluster`.
    :return: A list where each item is a sorted list of ints with no duplicate.
    The list at position `i` contains the cluster labels to which the point at position `i` in `X`
    belongs to. If `i < j`, the labels at position `i` are strictly less then those at position `j`.
    :rtype: `list[list[int]]`.
    '''
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
    ''' 
    Computes the connected components of the Mapper graph.
    The algorithm computes the connected components using a union-find data structure.
    This approach should be faster than computing the Mapper graph by first calling
    `tdamapper.core.mapper_graph` and then calling `networkx.connected_components` on it.

    :param X: A dataset.
    :type X: `numpy.ndarray` or list-like.
    :param y: Lens values.
    :type y: `numpy.ndarray` or list-like.
    :param cover: A cover algorithm.
    :type cover: A class from `tdamapper.cover`.
    :param clustering: A clustering algorithm.
    :type clustering: A class from `tdamapper.clustering` or a class from `sklearn.cluster`.
    :return: A list of labels, where the value at position `i` identifies
    the connected component of the point `X[i]`.
    :rtype: `list[int]`.
    '''
    itm_lbls = mapper_labels(X, y, cover, clustering)
    label_values = set()
    for lbls in itm_lbls:
        label_values.update(lbls)
    uf = UnionFind(label_values)
    labels = [-1 for _ in X]
    for lbls in itm_lbls:
        len_lbls = len(lbls)
        root = -1
        # noise points
        if len_lbls == 1:
            root = uf.find(lbls[0])
        elif len_lbls > 1:
            for first, second in zip(lbls, lbls[1:]):
                root = uf.union(first, second)
        labels.append(root)
    return labels


def mapper_graph(X, y, cover, clustering):
    ''' 
    Computes the Mapper graph.

    :param X: A dataset.
    :type X: `numpy.ndarray` or list-like.
    :param y: Lens values.
    :type y: `numpy.ndarray` or list-like.
    :param cover: A cover algorithm.
    :type cover: A class from `tdamapper.cover`.
    :param clustering: A clustering algorithm.
    :type clustering: A class from `tdamapper.clustering` or a class from `sklearn.cluster`.
    :return: The Mapper graph.
    :rtype: `networkx.Graph`.
    '''
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
    ''' 
    Computes an aggregation on the nodes of a graph.

    :param y: A dataset.
    :type y: `numpy.ndarray` or list-like.
    :param graph: A graph.
    :type graph: `networkx.Graph`.
    :param agg: An aggregation function.
    :type agg: Callable.
    :return: A dict of values, where each node is mapped to its aggregation.
    :rtype: `dict`.
    '''
    agg_values = {}
    nodes = graph.nodes()
    for node_id in nodes:
        node_values = [y[i] for i in nodes[node_id][ATTR_IDS]]
        agg_value = agg(node_values)
        agg_values[node_id] = agg_value
    return agg_values


class MapperAlgorithm:
    ''' 
    Main class for performing the Mapper Algorithm.

    :param cover: A cover algorithm.
    :type cover: A class from `tdamapper.cover`.
    :param clustering: A clustering algorithm.
    :type clustering: A class from `tdamapper.clustering` or a class from `sklearn.cluster`.
    '''

    def __init__(self, cover, clustering):
        self.__cover = cover
        self.__clustering = clustering
        self.graph_ = None

    def fit(self, X, y=None):
        ''' 
        Computes the Mapper Graph

        :param X: A dataset.
        :type X: `numpy.ndarray` or list-like.
        :param y: Lens values.
        :type y: `numpy.ndarray` or list-like.
        :return: `self`.
        '''
        self.graph_ = self.fit_transform(X, y)
        return self

    def fit_transform(self, X, y):
        ''' 
        Computes the Mapper Graph.

        :param X: A dataset.
        :type X: `numpy.ndarray` or list-like.
        :param y: Lens values.
        :type y: `numpy.ndarray` or list-like.
        :return: The Mapper graph.
        :rtype: `networkx.Graph`
        '''
        return mapper_graph(X, y, self.__cover, self.__clustering)
