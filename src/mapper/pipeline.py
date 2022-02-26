"""A module for the exact mapper algorithm"""
import numpy as np
import networkx as nx

from .graph import Vertex, Edge, Graph
from .network import Network
from .cover import TrivialClustering, TrivialCover


class MapperPipeline:

    def __init__(self, cover_algo=TrivialCover(), clustering_algo=TrivialClustering()):
        self.__cover_algo = cover_algo
        self.__clustering_algo = clustering_algo

    def fit(self, data):
        cluster_arr = self.__cover_algo.cover_points(data, self.__clustering_algo)
        graph = Graph()
        vertices = {}
        for i, clusters in enumerate(cluster_arr):
            for c in clusters:
                if c not in vertices:
                    vertices[c] = []
                vertices[c].append(i)
        for cluster, cluster_points in vertices.items():
            v = Vertex(cluster_points)
            graph.add_vertex(cluster, v)
        for p in cluster_arr:
            for s in p:
                for t in p:
                    if s != t:
                        edge = Edge(1, 1, 0) #compute this correctly
                        graph.add_edge(s, t, edge)
        return graph

        
                         


