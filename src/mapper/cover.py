"""A collection of functions to build open covers"""
import networkx as nx
from sklearn.utils import check_X_y, check_array

from .search import TrivialSearch
from .utils.unionfind import UnionFind

ATTR_IDS = 'ids'
ATTR_SIZE = 'size'

class CoverAlgorithm:
    
    def __init__(self, search=None, clustering=None):
        self.search = search
        self.clustering = clustering

    def _check_params(self):
        if not self.search:
            search = TrivialSearch()
        else:
            search = self.search
        if not self.clustering:
            clustering = TrivialClustering()
        else:
            clustering = self.clustering
        return search, clustering

    def get_params(self, deep=True):
        parameters = {}
        parameters['search'] = self.search
        parameters['clustering'] = self.clustering
        if deep:
            if self.search:
                for k, v in self.search.get_params().items():
                    parameters[f'search__{k}'] = v
            if self.clustering:
                for k, v in self.clustering.get_params().items():
                    parameters[f'clustering__{k}'] = v
        return parameters
    
    def set_params(self, **parameters):
        for k, v in parameters.items():
            setattr(self, k, v)
        return self

    def _check_input(self, X, y):
        if y is None:
            X = check_array(X)
        else:
            X, y = check_X_y(X, y)
        return X, y

    def _set_n_features_in_(self, X):
        self.n_features_in_ = len(X[0])

    def fit(self, X, y=None):
        search, clustering = self._check_params()
        X, y = self._check_input(X, y)
        cluster_count = 0
        search.fit(X)
        self.labels_ = [[] for _ in X]
        for i, cover_i in enumerate(self.labels_):
            cover_i = self.labels_[i]
            if not cover_i:
                neighs_ids = search.neighbors(X[i])
                neighs = [X[j] for j in neighs_ids]
                #neighs = np.take(X, neighs_ids, axis=0)
                labels = clustering.fit(neighs).labels_
                max_label = 0
                for (n, label) in zip(neighs_ids, labels):
                    if label != -1:
                        if label > max_label:
                            max_label = label
                        self.labels_[n].append(cluster_count + label)
                cluster_count += max_label + 1
        self._set_n_features_in_(X)
        return self

    def build_graph(self):
        graph = nx.Graph()
        clusters = set()
        sizes = {}
        point_ids = {}
        for point_id, point_labels in enumerate(self.labels_):
            for label in point_labels:
                if label not in clusters:
                    clusters.add(label)
                    graph.add_node(label)
                    sizes[label] = 0
                    point_ids[label] = []
                sizes[label] += 1
                point_ids[label].append(point_id)
        nx.set_node_attributes(graph, sizes, ATTR_SIZE)
        nx.set_node_attributes(graph, point_ids, ATTR_IDS)
        edges = set()
        for labels in self.labels_:
            for s in labels:
                for t in labels:
                    if s != t and (s, t) not in edges:
                        graph.add_edge(s, t, weight=1) # TODO: compute weight correctly
                        edges.add((s, t))
                        graph.add_edge(t, s, weight=1) # TODO: compute weight correctly
                        edges.add((t, s))
        return graph


class CoverGraph:

    def __init__(self):
        pass

    def build(self, subsets):
        graph = nx.Graph()
        # a map where:
        # key   = item
        # value = set of subsets containing the key
        items_subsets = {} 
        for n, subset in enumerate(subsets): 
            graph.add_node(n)
            for x in subset:
                if x not in items_subsets:
                    items_subsets[x] = set()
                items_subsets[x].add(n)

        sizes = [len(s) for s in subsets]
        nx.set_node_attributes(graph, sizes, ATTR_SIZE)
        edges = set()
        for item, subset_ids in items_subsets:
            for source in subset_ids:
                for target in subset_ids:
                    if target > source and (source, target) not in edges:
                        graph.add_edge(source, target, weight=1) # TODO: compute weight correctly
                        graph.add_edge(target, source, weight=1) # TODO: compute weight correctly
                        edges.add((source, target))
        return graph



class SearchClustering:

    def __init__(self, search=None):
        self.search = search

    def _check_params(self):
        if not self.search:
            search = TrivialSearch()
        else:
            search = self.search
        return search

    def get_params(self, deep=True):
        parameters = {}
        parameters['search'] = self.search
        if deep:
            if self.search:
                for k, v in self.search.get_params().items():
                    parameters[f'search__{k}'] = v
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
        multilabels = CoverAlgorithm(search=self.search).fit(X).labels_
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
