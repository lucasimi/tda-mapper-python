import networkx as nx


ATTR_IDS = 'ids'
ATTR_SIZE = 'size'


def cover_labels(X, cover_ids, clustering):
    '''
    Perform local clustering on subsets of a dataset open cover.

    :param X: A dataset
    :type X: numpy.ndarray or list-like
    :param cover_ids: An open cover, expressed as a list of lists. 
    Each item in cover_ids is a list of ids of points from X.
    :type cover_ids: list[list[int]]
    :param clustering: A clustering algorithm
    :type clustering: A class from tdamapper.clustering or a class from sklearn.cluster
    :return: A list where each item is a list of labels.
    If i < j, the labels at position i are strictly less then those at position j.
    :rtype: list[list[int]]
    '''
    def get_labels(ids):
        return clustering.fit([X[j] for j in ids]).labels_
    lbls = [get_labels(ids) for ids in cover_ids]
    max_lbl = 0
    for local_lbls in lbls:
        global_lbls = [lbl + max_lbl for lbl in local_lbls] #TODO: handle when l < 0
        max_local_lbl = 0
        for lbl in local_lbls:
            if lbl > max_local_lbl:
                max_local_lbl = lbl
        max_lbl += max_local_lbl + 1
        yield global_lbls


def item_labels(X, y, cover, clustering):
    '''
    Computes the open cover, then perform local clustering on each open set from the cover.

    :param X: A dataset
    :type X: numpy.ndarray or list-like
    :param y: lens values
    :type y: numpy.ndarray or list-like
    :param cover: A cover algorithm
    :type cover: A class from tdamapper.cover
    :param clustering: A clustering algorithm
    :type clustering: A class from tdamapper.clustering or a class from sklearn.cluster
    :return: A list where each item is a sorted list of ints with no duplicate.
    The list at position i contains the cluster labels to which the point at position i in X belongs to. 
    :rtype: list[list[int]]
    '''
    cover_ids = list(cover.apply(y))
    cover_lbls = list(cover_labels(X, cover_ids, clustering))
    itm_lbls = [[] for _ in X]
    for ids, lbls in zip(cover_ids, cover_lbls):
        for itm_id, itm_lbl in zip(ids, lbls):
            itm_lbls[itm_id].append(itm_lbl)
    return itm_lbls


def mapper_graph(X, y, cover, clustering):
    ''' 
    Computes the Mapper graph

    :param X: A dataset
    :type X: numpy.ndarray or list-like
    :param y: Lens values
    :type y: numpy.ndarray or list-like
    :param cover: A cover algorithm
    :type cover: A class from tdamapper.cover
    :param clustering: A clustering algorithm
    :type clustering: A class from tdamapper.clustering or a class from sklearn.cluster
    :return: The Mapper graph
    :rtype: networkx.Graph
    '''
    itm_lbls = item_labels(X, y, cover, clustering)
    graph = nx.Graph()
    for n, lbls in enumerate(itm_lbls):
        for lbl in lbls:
            if not graph.has_node(lbl):
                graph.add_node(lbl, **{ATTR_SIZE: 0, ATTR_IDS: []})
            nodes = graph.nodes
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

    :param cover: A cover algorithm
    :type cover: A class from tdamapper.cover
    :param clustering: A clustering algorithm
    :type clustering: A class from tdamapper.clustering or a class from sklearn.cluster
    '''

    def __init__(self, cover, clustering):
        self.__cover = cover
        self.__clustering = clustering
        self.graph_ = None

    def fit(self, X, y=None):
        ''' 
        Computes the Mapper Graph

        :param X: A dataset
        :type X: numpy.ndarray or list-like
        :param y: Lens values
        :type y: numpy.ndarray or list-like
        :return: self
        '''
        self.graph_ = self.fit_transform(X, y)
        return self

    def fit_transform(self, X, y):
        ''' 
        Computes the Mapper Graph

        :param X: A dataset
        :type X: numpy.ndarray or list-like
        :param y: Lens values
        :type y: numpy.ndarray or list-like
        :return: The Mapper Graph
        :rtype: networkx.Graph
        '''
        return mapper_graph(X, y, self.__cover, self.__clustering)
