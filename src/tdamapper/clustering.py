"""
Clustering module for the Mapper algorithm.

This module implements some tools for the clustering step of Mapper algorithm,
which groups the data points in each open set into clusters using a clustering
algorithm of choice. The clusters are then used to form the nodes of the Mapper
graph, and are connected by edges if they share points in the overlap.
"""

from tdamapper.core import mapper_connected_components
from tdamapper.cover import TrivialCover
import tdamapper.core


class TrivialClustering(tdamapper.core.TrivialClustering):
    pass


class FailSafeClustering(tdamapper.core.FailSafeClustering):
    pass


class MapperClustering:
    """
    A clustering algorithm based on the Mapper graph.

    The Mapper algorithm constructs a graph from a dataset, where each node
    represents a cluster of points and each edge represents an overlap between
    clusters. Each point in the dataset belongs to one or more nodes in the
    graph. These nodes are therefore all connected and share the same connected
    component in the Mapper graph. This class clusters point according to their
    connected component in the Mapper graph calling the function
    :func:`tdamapper.core.mapper_connected_components`.

    :param cover: The cover algorithm to apply to lens space.
    :type cover: A class from :mod:`tdamapper.cover`
    :param clustering: The clustering algorithm to apply to each subset of the
        dataset.
    :type clustering: A class from :mod:`tdamapper.clustering`, or a class from
        :mod:`sklearn.cluster`
    """

    def __init__(self, cover=None, clustering=None):
        self.cover = cover
        self.clustering = clustering

    def fit(self, X, y=None):
        cover = self.cover if self.cover else TrivialCover()
        clustering = self.clustering if self.clustering else TrivialClustering()
        itm_lbls = mapper_connected_components(X, y, cover, clustering)
        self.labels_ = [itm_lbls[i] for i, _ in enumerate(X)]
        return self
