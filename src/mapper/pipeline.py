"""A module for the exact mapper algorithm"""
import math
import numpy as np
import pandas as pd
import networkx as nx

from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse

from .cover import TrivialClustering, TrivialCover


ATTR_IDS = 'ids'
ATTR_SIZE = 'size'


def _rmse(x, y):
    return math.sqrt(mse(x, y))


KPIS = {
    'mape': mape,
    'mae': mae,
    'rmse': _rmse
}


def compute_connected_components(graph):
    cc_id = 1
    vert_cc = {}
    for cc in nx.connected_components(graph):
        for node in cc:
            vert_cc[node] = cc_id
        cc_id += 1
    return vert_cc


class MapperKpis:

    def __init__(self, agg=lambda x: np.nanmean(x, axis=0), colormap=np.nanmean, kpis=None):
        self.__agg = agg
        self.__colormap = colormap
        self.__kpis = KPIS if not kpis else kpis
        self.__col_values = {}
        self.__agg_values = {}
        self.__kpi_values = {}

    def aggregate(self, graph, data, fun, attribute=None):
        nodes = graph.nodes()
        self.__agg_values = {}
        self.__col_values = {}
        self.__kpi_values = {kpi_name:{} for kpi_name, _ in self.__kpis.items()}
        for node_id in nodes:
            node_data = [data[i] for i in nodes[node_id][ATTR_IDS]]
            node_values = [fun(x) for x in node_data]
            agg_value = self.__agg(node_values)
            self.__agg_values[node_id] = agg_value
            col_value = self.__colormap(agg_value)
            self.__col_values[node_id] = col_value
            for kpi_name, kpi_fun in self.__kpis.items():
                self.__kpi_values[kpi_name][node_id] = kpi_fun(node_values, [agg_value for _ in node_data])
        if attribute is not None:
            nx.set_node_attributes(graph, self.__col_values, attribute)

    def get_colors(self):
        return self.__col_values

    def get_kpis(self):
        return self.__kpi_values

    def get_aggregations(self):
        return self.__agg_values


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
