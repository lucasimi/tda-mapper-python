"""A collection of functions to build open covers"""
import networkx as nx
from .search import TrivialSearch
from .utils.unionfind import UnionFind

ATTR_IDS = 'ids'
ATTR_SIZE = 'size'

class CoverAlgorithm:
    
    def __init__(self, search_algo=None, clustering_algo=None):
        if not search_algo:
            self.__search_algo = TrivialSearch()
        else:
            self.__search_algo = search_algo
        if not clustering_algo:
            self.__clustering_algo = TrivialClustering()
        else:
            self.__clustering_algo = clustering_algo
        self.labels_ = []

    def fit(self, X):
        cluster_count = 0
        self.__search_algo.fit(X)
        multilabels = [[] for _ in X]
        for i, cover_i in enumerate(multilabels):
            cover_i = multilabels[i]
            if not cover_i:
                neighs_ids = self.__search_algo.neighbors(X[i])
                neighs = [X[j] for j in neighs_ids]
                labels = self.__clustering_algo.fit(neighs).labels_
                max_label = 0
                for (n, label) in zip(neighs_ids, labels):
                    if label != -1:
                        if label > max_label:
                            max_label = label
                        multilabels[n].append(cluster_count + label)
                cluster_count += max_label + 1
        self.labels_ = multilabels
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


class SearchClustering:

    def __init__(self, search_algo=None):
        if not search_algo:
            self.__search_algo = TrivialSearch()
        else:
            self.__search_algo = search_algo
        self.labels_ = []

    def fit(self, X):
        cover_algo = CoverAlgorithm(search_algo=self.__search_algo)
        multilabels = cover_algo.fit(X).labels_
        label_values = set()
        for labels in multilabels:
            label_values.update(labels)
        uf = UnionFind(label_values)
        cc = [None for _ in X]
        for labels in multilabels:
            if len(labels) > 1:
                for first, second in zip(labels, labels[1:]):
                    root = uf.union(first, second)
            else:
                root = uf.find(labels[0])
            cc.append(root)
        self.labels_ = cc
        return self


class TrivialClustering:

    def TrivialClustering(self):
        self.labels_ = []

    def fit(self, X):
        self.labels_ = [0 for _ in X]
        return self
