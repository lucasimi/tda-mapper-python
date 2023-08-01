import networkx as nx
from sklearn.utils import check_X_y, check_array

from .search import BallSearch, KnnSearch, TrivialSearch
from .utils.unionfind import UnionFind


ATTR_IDS = 'ids'
ATTR_SIZE = 'size'

_ID_IDS = 0
_ID_NEIGHS = 1


def build_labels(X, cover, clustering):
    '''
    Takes a dataset, returns a list of lists, where the list at position i
    contains the cluster ids to which the item at position i belongs to.
    * Each list in the output is a sorted list of ints with no duplicate.
    '''
    max_label = 0
    labels = [[] for _ in X]
    for neigh_ids in cover.charts(X):
        neigh_data = [X[j] for j in neigh_ids]
        neigh_labels = clustering.fit(neigh_data).labels_
        max_neigh_label = 0
        for (neigh_id, neigh_label) in zip(neigh_ids, neigh_labels):
            if neigh_label != -1:
                if neigh_label > max_neigh_label:
                    max_neigh_label = neigh_label
                labels[neigh_id].append(max_label + neigh_label)
        max_label += max_neigh_label + 1
    return labels
        

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


def build_graph(X, cover, clustering):
    labels = build_labels(X, cover, clustering)
    adjaciency = build_adjaciency(labels)
    graph = nx.Graph()
    for source, (items, _) in adjaciency.items():
        graph.add_node(source, size=len(items), ids=items)
    edges = set()
    for source, (_, target_ids) in adjaciency.items():
        for target in target_ids:
            if (source, target) not in edges:
                graph.add_edge(source, target)
                edges.add((source, target))
                graph.add_edge(target, source)
                edges.add((target, source))
    return graph


def generate_charts(X, search):
    covered = set()
    search.fit(X)
    for i in range(len(X)):
        if i not in covered:
            xi = X[i]
            neigh_ids = search.neighbors(xi)
            covered.update(neigh_ids)
            yield neigh_ids


def compute_connected_components(X, graph):
    cc_id = 1
    item_cc = {}
    for cc in nx.connected_components(graph):
        for node in cc:
            for item_id in node[ATTR_IDS]:
                item_cc[item_id] = cc_id
        cc_id += 1
    return item_cc


class MapperAlgorithm:

    def __init__(self, cover, clustering):
        self.__cover = cover
        self.__clustering = clustering
            
    def build_graph(self, X):
        return build_graph(X, self.__cover, self.__clustering)


class BallCover:

    def __init__(self, radius, metric): 
        self.__radius = radius 
        self.__metric = metric 

    def charts(self, X): 
        search = BallSearch(self.__radius, self.__metric)
        return generate_charts(X, search)


class KnnCover:

    def __init__(self, neighbors, metric): 
        self.__neighbors = neighbors 
        self.__metric = metric 

    def charts(self, X): 
        search = KnnSearch(self.__neighbors, self.__metric)
        return generate_charts(X, search)


class TrivialCover:

    def __init__(self): 
        pass

    def charts(self, X): 
        search = TrivialSearch()
        return generate_charts(X, search)


class TrivialClustering:

    def TrivialClustering(self):
        pass

    def get_params(self, deep=True):
        return {}
    
    def set_params(self, **parameters):
        return self

    def _set_n_features_in_(self, X):
        self.n_features_in_ = len(X[0])

    def _check_input(self, X, y):
        if y is None:
            X = check_array(X)
        else:
            X, y = check_X_y(X, y)
        return X, y

    def fit(self, X, y=None):
        X, y = self._check_input(X, y)
        self.labels_ = [0 for _ in X]
        self._set_n_features_in_(X)
        return self


class CoverClustering:

    def __init__(self, cover):
        self.__cover = cover

    def _check_params(self):
        if not self.__cover:
            cover = TrivialCover()
        else:
            cover = self.__cover
        return cover

    def get_params(self, deep=True):
        parameters = {}
        parameters['cover'] = self.__cover
        if deep:
            if self.__cover:
                for k, v in self.search.get_params().items():
                    parameters[f'cover__{k}'] = v
        return parameters
    
    def set_params(self, **parameters):
        for k, v in parameters.items():
            setattr(self, k, v)
        return self

    def _set_n_features_in_(self, X):
        self.n_features_in_ = len(X[0])

    def _check_input(self, X, y):
        if y is None:
            X = check_array(X)
        else:
            X, y = check_X_y(X, y)
        return X, y

    def fit(self, X, y=None):
        X, y = self._check_input(X, y)
        multilabels = build_labels(X, self.__cover, TrivialClustering())
        label_values = set()
        for labels in multilabels:
            label_values.update(labels)
        uf = UnionFind(label_values)
        self.labels_ = []
        for labels in multilabels:
            if len(labels) > 1:
                for first, second in zip(labels, labels[1:]):
                    root = uf.union(first, second)
            else:
                root = uf.find(labels[0])
            self.labels_.append(root)
        self._set_n_features_in_(X)
        return self