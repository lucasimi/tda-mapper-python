import networkx as nx
from joblib import Parallel, delayed


ATTR_IDS = 'ids'
ATTR_SIZE = 'size'

_ID_IDS = 0
_ID_NEIGHS = 1


class RefinedNet:

    def __init__(self, X, proximity_net, clustering, n_jobs):
        self.__X = X
        self.__proximity_net = proximity_net
        self.__clustering = clustering
        self.__n_jobs = n_jobs

    def __labels(self, ids):
        x_data = [self.__X[j] for j in ids]
        return self.__clustering.fit(x_data).labels_
        #lbls = [l + max_lbl for l in labels] #TODO: handle when l < 0

    def __reindex(self, lbls):
        max_lbl = 0
        for local_lbls in lbls:
            global_lbls = [lbl + max_lbl for lbl in local_lbls] #TODO: handle when l < 0
            max_local_lbl = 0
            for lbl in local_lbls:
                if lbl > max_local_lbl:
                    max_local_lbl = lbl
            max_lbl += max_local_lbl + 1
            yield global_lbls

    def __iter__(self):
        parallel = Parallel(n_jobs=self.__n_jobs)
        lbls = parallel(delayed(self.__labels)(ids) for ids in self.__proximity_net)
        return self.__reindex(lbls)


def build_labels_par(X, y, cover, clustering, n_jobs):
    '''
    Computes the local cluster labels for each element of the dataset and stores them in a list.
    Each item in the returned list is a sorted list of ints with no duplicate.
    The list at position i contains the cluster ids to which the point at position i belongs to. 

    :param X: A dataset
    :type X: numpy.ndarray or list-like
    :param y: lens values
    :type y: numpy.ndarray or list-like
    :param cover: A cover algorithm
    :type cover: A class from tdamapper.cover
    :param clustering: A clustering algorithm
    :type clustering: A class from tdamapper.clustering or a class from sklearn.cluster
    :param n_jobs: The number of parallel jobs for clustering
    :type n_jobs: int
    :return: The labels list
    :rtype: list[list[int]]
    '''
    net = list(cover.build(y))
    net_lbls = list(RefinedNet(X, net, clustering, n_jobs))
    itm_lbls = [[] for _ in X]
    for ids, lbls in zip(net, net_lbls):
        for itm_id, itm_lbl in zip(ids, lbls):
            itm_lbls[itm_id].append(itm_lbl)
    return itm_lbls


def build_adjaciency(labels):
    '''
    Takes a list of lists of items, returns a dict where each item is
    mapped to a couple. Inside each couple the first entry is the list 
    of positions where the item is present, the second entry is the list
    of items which appear in any of the lists where the key is present.

    :param labels: A list of lists 
    :type labels: list[list[int]]
    '''
    adj = {}
    for n, clusters in enumerate(labels):
        for label in clusters:
            if label not in adj:
                adj[label] = ([], [])
            adj[label][_ID_IDS].append(n)
    edges = set()
    for clusters in labels:
        clusters_len = len(clusters)
        for i in range(clusters_len):
            source = clusters[i]
            for j in range(i + 1, clusters_len):
                target = clusters[j]
                if (source, target) not in edges:
                    target = clusters[j]
                    adj[source][_ID_NEIGHS].append(target)
                    edges.add((source, target))
                    adj[target][_ID_NEIGHS].append(source)
                    edges.add((target, source))
    return adj


def build_graph(X, y, cover, clustering, n_jobs=1):
    ''' 
    Computes the Mapper Graph

    :param X: A dataset
    :type X: numpy.ndarray or list-like
    :param y: Lens values
    :type y: numpy.ndarray or list-like
    :return: The Mapper Graph
    :rtype: networkx.Graph
    '''
    labels = build_labels_par(X, y, cover, clustering, n_jobs)
    adjaciency = build_adjaciency(labels)
    graph = nx.Graph()
    for source, (items, _) in adjaciency.items():
        graph.add_node(source, **{ATTR_SIZE: len(items), ATTR_IDS: items})
    edges = set()
    for source, (_, target_ids) in adjaciency.items():
        for target in target_ids:
            if (source, target) not in edges:
                graph.add_edge(source, target)
                edges.add((source, target))
                graph.add_edge(target, source)
                edges.add((target, source))
    return graph


def build_connected_components(graph):
    '''
    Takes a dataset and a graph, where each node represents a sets of elements
    from the dataset, returns a list of integers, where position i is the id
    of the connected component of the graph where the element at position i 
    from the dataset lies.

    :param graph: Any graph
    :type graph: networkx.Graph
    '''
    cc_id = 1
    item_cc = {}
    for connected_component in nx.connected_components(graph):
        for node in connected_component:
            for item_id in graph.nodes[node][ATTR_IDS]:
                item_cc[item_id] = cc_id
        cc_id += 1
    return item_cc


def compute_local_interpolation(y, graph, agg):
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

    def __init__(self, cover, clustering, n_jobs=1):
        self.__cover = cover
        self.__clustering = clustering
        self.__n_jobs = n_jobs
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
        return build_graph(X, y, self.__cover, self.__clustering, self.__n_jobs)
