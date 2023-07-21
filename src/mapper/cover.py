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

    def __build_dual_map(self, groups):
        '''
        Takes a list of groups of items, 
        returns a dict where each item is mapped to the set of ids of groups containing the key.
        Each id is the position of the corresponding group in the input
        '''
        dual_map = {} 
        for n, subset in enumerate(groups): 
            for x in subset:
                if x not in dual_map:
                    dual_map[x] = set()
                dual_map[x].add(n)
        return dual_map 

    def build_dict(self, groups):
        '''
        Takes a list of groups of items, 
        returns a dict where each group id is mapped to the set of ids of intersecting groups.
        Each id is the position of the corresponding group in the input
        '''
        adjaciency_map = {}
        dual_map = self.__build_dual_map(groups)
        for subset_id, _ in enumerate(groups):
            adjaciency_map[subset_id] = set()
        edges = set()
        for item, subset_ids in dual_map.items():
            for source in subset_ids:
                for target in subset_ids:
                    if target > source and (source, target) not in edges:
                        adjaciency_map[source].add(target)
                        adjaciency_map[target].add(source)
                        edges.add((source, target))
        return adjaciency_map

    def build_nx(self, groups):
        '''
        Takes a list of groups of items, 
        returns a networkx graph where a vertex corresponds to a group, 
        and whenever two groups intersect, an edge is drawn between their corresponding vertices.
        '''
        graph = nx.Graph()
        dual_map = self.__build_dual_map(groups)
        sizes = [len(s) for s in groups]
        nx.set_node_attributes(graph, sizes, ATTR_SIZE)
        for subset_id, _ in enumerate(groups):
            graph.add_node(subset_id)
        edges = set()
        for item, subset_ids in dual_map.items():
            for source in subset_ids:
                for target in subset_ids:
                    if target > source and (source, target) not in edges:
                        graph.add_edge(source, target)
                        graph.add_edge(target, source)
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
