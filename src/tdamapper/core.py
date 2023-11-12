import logging
import networkx as nx
from joblib import Parallel, delayed


_ATTR_IDS = 'ids'
_ATTR_SIZE = 'size'

_ID_IDS = 0
_ID_NEIGHS = 1


_logger = logging.getLogger(__name__)

logging.basicConfig(
    format = '%(asctime)s %(module)s %(levelname)s: %(message)s',
    datefmt = '%m/%d/%Y %I:%M:%S %p',
    level = logging.INFO)


def build_labels_par(X, y, cover, clustering, verbose=False, permissive=True, n_jobs=1):
    def _lbls(i, x):
        try:
            x_data = [X[j] for j in x]
            return x, clustering.fit(x_data).labels_
        except ValueError as err:
            if verbose:
                _logger.warning('Unable to perform clustering on local chart %d: %s', i, err)
            if not permissive:
                raise err
            return x, [0 for _ in x]
    itr = enumerate(cover.proximity_net(y))
    par = Parallel(n_jobs=n_jobs)(delayed(_lbls)(i, ids) for i, ids in itr)
    max_lbl = 0
    lbls = [[] for _ in X]
    for neigh_ids, neigh_lbls in par:
        max_neigh_lbl = 0
        for neigh_id, neigh_lbl in zip(neigh_ids, neigh_lbls):
            if neigh_lbl != -1:
                if neigh_lbl > max_neigh_lbl:
                    max_neigh_lbl = neigh_lbl
                lbls[neigh_id].append(max_lbl + neigh_lbl)
        max_lbl += max_neigh_lbl + 1
    return lbls


def build_labels(X, y, cover, clustering, verbose=False, permissive=True):
    '''
    Takes a dataset, returns a list of lists, where the list at position i
    contains the cluster ids to which the item at position i belongs to.
    * Each list in the output is a sorted list of ints with no duplicate.
    '''
    max_lbl = 0
    lbls = [[] for _ in X]
    for i, neigh_ids in enumerate(cover.proximity_net(y)):
        neigh_data = [X[j] for j in neigh_ids]
        try:
            neigh_lbls = clustering.fit(neigh_data).labels_
        except ValueError as err:
            neigh_lbls = [0 for _ in neigh_data]
            if verbose:
                _logger.warning('Unable to perform clustering on local chart %d: %s', i, err)
            if not permissive:
                raise err
        max_neigh_lbl = 0
        for neigh_id, neigh_lbl in zip(neigh_ids, neigh_lbls):
            if neigh_lbl != -1:
                if neigh_lbl > max_neigh_lbl:
                    max_neigh_lbl = neigh_lbl
                lbls[neigh_id].append(max_lbl + neigh_lbl)
        max_lbl += max_neigh_lbl + 1
    return lbls


def build_adjaciency(labels):
    '''
    Takes a list of lists of items, returns a dict where each item is
    mapped to a couple. Inside each couple the first entry is the list 
    of positions where the item is present, the second entry is the list
    of items which appear in any of the lists where the key is present.
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


def build_graph(X, y, cover, clustering, verbose=False, permissive=True, n_jobs=1):
    labels = build_labels_par(X, y, cover, clustering,
        verbose=verbose,
        permissive=permissive,
        n_jobs=n_jobs)
    adjaciency = build_adjaciency(labels)
    graph = nx.Graph()
    for source, (items, _) in adjaciency.items():
        graph.add_node(source, **{_ATTR_SIZE: len(items), _ATTR_IDS: items})
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
    '''
    cc_id = 1
    item_cc = {}
    for connected_component in nx.connected_components(graph):
        for node in connected_component:
            for item_id in graph.nodes[node][_ATTR_IDS]:
                item_cc[item_id] = cc_id
        cc_id += 1
    return item_cc


def compute_local_interpolation(y, graph, agg):
    agg_values = {}
    nodes = graph.nodes()
    for node_id in nodes:
        node_values = [y[i] for i in nodes[node_id][_ATTR_IDS]]
        agg_value = agg(node_values)
        agg_values[node_id] = agg_value
    return agg_values


class MapperAlgorithm:

    def __init__(self, cover, clustering, verbose=False, permissive=True, n_jobs=1):
        self.__cover = cover
        self.__clustering = clustering
        self.__verbose = verbose
        self.__permissive = permissive
        self.__n_jobs = n_jobs
        self.graph_ = None

    def fit(self, X, y=None):
        self.graph_ = self.fit_transform(X, y)
        return self

    def fit_transform(self, X, y):
        return build_graph(X, y, self.__cover, self.__clustering, 
            self.__verbose, 
            self.__permissive, 
            self.__n_jobs)


class MapperClassifier:

    def __init__(self, mapper_algo):
        self.__mapper_algo = mapper_algo
        self.labels_ = None

    def fit(self, X, y=None):
        self.labels_ = self.fit_predict(X, y)
        return self

    def fit_predict(self, X, y):
        graph = self.__mapper_algo.fit_transform(X, y)
        ccs = build_connected_components(graph)
        return [ccs[i] for i, _ in enumerate(X)]
