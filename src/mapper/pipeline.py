"""A module for the exact mapper algorithm"""
import numpy as np
import networkx as nx

from .cover import TrivialClustering, TrivialCover

ATTR_IDS = 'ids'
ATTR_SIZE = 'size'


def compute_connected_components(graph):
    cc_id = 1
    vert_cc = {}
    for cc in nx.connected_components(graph):
        for node in cc:
            vert_cc[node] = cc_id
        cc_id += 1
    return vert_cc


def aggregate(graph, data, fun, agg=np.nanmean):
    nodes = graph.nodes()
    values = {}
    for node_id in nodes:
        node_values = [fun(data[i]) for i in nodes[node_id][ATTR_IDS]]
        values[node_id] = agg(node_values)
    return values


class MapperPipeline:

    def __init__(self, cover_algo=TrivialCover(), clustering_algo=TrivialClustering()):
        self.__cover_algo = cover_algo
        self.__clustering_algo = clustering_algo

    def _build_graph(self, cover_arr):
        graph = nx.Graph()
        added_clusters = set()
        sizes = {}
        point_ids = {}
        for point_id, point_clusters in enumerate(cover_arr):
            for cluster in point_clusters:
                if cluster not in added_clusters:
                    added_clusters.add(cluster)
                    graph.add_node(cluster)
                    sizes[cluster] = 0
                    point_ids[cluster] = []
                sizes[cluster] += 1
                point_ids[cluster].append(point_id)
        nx.set_node_attributes(graph, sizes, ATTR_SIZE)
        nx.set_node_attributes(graph, point_ids, ATTR_IDS)
        for clusters in cover_arr:
            for s in clusters:
                for t in clusters:
                    if s != t:
                        graph.add_edge(s, t, weight=1) # TODO: compute weight correctly
        return graph

    def fit(self, data):
        cover_arr = self.__cover_algo.cover_points(data, self.__clustering_algo)
        graph = self._build_graph(cover_arr)
        return graph
