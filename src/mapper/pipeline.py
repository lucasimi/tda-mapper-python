"""A module for the exact mapper algorithm"""
import time
import networkx as nx

from .cover import CoverAlgorithm


def compute_connected_components(graph):
    cc_id = 1
    vert_cc = {}
    for cc in nx.connected_components(graph):
        for node in cc:
            vert_cc[node] = cc_id
        cc_id += 1
    return vert_cc


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
