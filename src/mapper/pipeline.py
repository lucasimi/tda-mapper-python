"""A module for the exact mapper algorithm"""
from .graph import CoverGraph
from .cover import TrivialClustering, TrivialCover


class MapperPipeline:

    def __init__(self, cover_algo=TrivialCover(), clustering_algo=TrivialClustering()):
        self.__cover_algo = cover_algo
        self.__clustering_algo = clustering_algo

    def fit(self, data):
        cluster_arr = self.__cover_algo.cover_points(data, self.__clustering_algo)
        return CoverGraph(cluster_arr)
