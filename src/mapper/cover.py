"""A collection of functions to build open covers"""
import networkx as nx
from sklearn.utils import check_X_y, check_array

from .search import BallSearch, KnnSearch, TrivialSearch
from .utils.unionfind import UnionFind


class SearchCoverCharts: 

    def __init__(self, X, search):
        self.__X = X
        self.__search = search

    def generate(self):
        covered = set()
        self.__search.fit(self.__X)
        for i in range(len(self.__X)):
            if i not in covered:
                xi = self.__X[i]
                neigh_ids = self.__search.neighbors(xi)
                covered.update(neigh_ids)
                yield neigh_ids


class CoverGraph:

    def build_labels(self, groups):
        '''
        Takes a list of groups of items, returns a dict where each item 
        is mapped to the list of ids of groups containing the key.
        * Each id is the position of the corresponding group in the input.
        * Each key maps to a sorted list of ints with no duplicate element.
        '''
        labels = {} 
        for n, group in enumerate(groups): 
            for x in set(group):
                if x not in labels:
                    labels[x] = []
                labels[x].append(n)
        return labels

    def build_adjaciency(self, groups):
        '''
        Takes a list of groups of items, returns a dict where each group id
        is mapped to the set of ids of intersecting groups.
        * Each id is the position of the corresponding group in the input.
        '''
        labels = self.build_labels(groups)
        adjaciency = {n: [] for n, _ in enumerate(groups)}
        edges = set()
        for item, group_ids in labels.items():
            for i in range(len(group_ids)):
                source = group_ids[i]
                for j in range(i + 1, len(group_ids)):
                    target = group_ids[j]
                    if (source, target) not in edges:
                        adjaciency[source].append(target)
                        edges.add((source, target))
                        adjaciency[target].append(source)
                        edges.add((target, source))
        return adjaciency

    def build_nx(self, adjaciency):
        '''
        Takes a list of groups of items, returns a networkx graph where a vertex
        corresponds to a group. Whenever two groups intersect, an edge is drawn 
        between their corresponding vertices.
        '''
        graph = nx.Graph()
        for node_id in adjaciency:
            graph.add_node(node_id)
        edges = set()
        for node_id, node_ids in adjaciency.items():
            for i in range(len(node_ids)):
                source = node_ids[i]
                for j in range(i + 1, len(node_ids)):
                    target = node_ids[j]
                    if (source, target) not in edges:
                        graph.add_edge(source, target)
                        edges.add((source, target))
                        graph.add_edge(target, source)
                        edges.add((target, source))
        return graph

'''
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
'''


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


class BallCover:

    def __init__(self, radius, metric): 
        self.__radius = radius 
        self.__metric = metric 

    def get_charts_iter(self, X): 
        search = BallSearch(self.__radius, self.__metric)
        return SearchCoverCharts(X, search)


class KnnCover:

    def __init__(self, neighbors, metric): 
        self.__neighbors = neighbors 
        self.__metric = metric 

    def get_charts_iter(self, X): 
        search = KnnSearch(self.__neighbors, self.__metric)
        return SearchCoverCharts(X, search)


class TrivialCover:

    def __init__(self): 
        pass

    def get_charts_iter(self, X): 
        search = TrivialSearch()
        return SearchCoverCharts(X, search)
