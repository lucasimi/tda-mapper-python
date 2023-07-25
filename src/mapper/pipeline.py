"""A module for the exact mapper algorithm"""
import time
import networkx as nx

#from .cover import CoverAlgorithm


def compute_connected_components(graph):
    cc_id = 1
    vert_cc = {}
    for cc in nx.connected_components(graph):
        for node in cc:
            vert_cc[node] = cc_id
        cc_id += 1
    return vert_cc


class MapperAlgorithm:

    def __init__(self, cover, clustering):
        self.__cover = cover
        self.__clustering = clustering

    def build_labels(self, X):
        '''
        Takes a dataset, a search algorithm and a clustering algortithm, 
        returns a list of lists, where the list at position i contains
        the cluster ids to which the item at position i belongs to.
        * Each list in the output is a sorted list of ints with no duplicate.
        '''
        max_label = 0
        labels = [[] for _ in X]
        for neigh_ids in self.__cover.get_charts_iter(X).generate():
            neigh_data = [X[j] for j in neigh_ids]
            neigh_labels = self.__clustering.fit(neigh_data).labels_
            max_neigh_label = 0
            for (neigh_id, neigh_label) in zip(neigh_ids, neigh_labels):
                if neigh_label != -1:
                    if neigh_label > max_neigh_label:
                        max_neigh_label = neigh_label
                    labels[neigh_id].append(max_label + neigh_label)
            max_label += max_neigh_label + 1
        return labels
            

    def build_adjaciency(self, labels):
        adj = {}
        for clusters in labels:
            for label in clusters:
                if label not in adj:
                    adj[label] = []
        edges = set()
        for clusters in labels:
            clusters_len = len(clusters)
            for i in range(clusters_len):
                source = clusters[i]
                for j in range(i + 1, clusters_len):
                    target = clusters[j]
                    if (source, target) not in edges:
                        target = clusters[j]
                        adj[source].append(target)
                        edges.add((source, target))
                        adj[target].append(source)
                        edges.add((target, source))
        return adj


    def build_graph(self, X):
        labels = self.build_labels(X)
        adjaciency = self.build_adjaciency(labels)
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


class MapperPipeline:

    def __init__(self, search=None, clustering=None):
        self.cover = CoverAlgorithm(
            search=search,
            clustering=clustering
        )
        self.__graph = None

    def fit(self, X):
        self.__graph = self.cover.fit(X).build_graph()
        return self

    def get_graph(self):
        return self.__graph    

