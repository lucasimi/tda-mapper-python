"""
Clustering module for the Mapper algorithm.

This module implements some tools for the clustering step of Mapper algorithm,
which groups the data points in each open set into clusters using a clustering
algorithm of choice. The clusters are then used to form the nodes of the Mapper
graph, and are connected by edges if they share points in the overlap.
"""

import logging

from tdamapper.core import mapper_connected_components
from tdamapper.cover import TrivialCover

_logger = logging.getLogger(__name__)

logging.basicConfig(
    format='%(asctime)s %(module)s %(levelname)s: %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO)


class TrivialClustering:
    """
    A clustering algorithm that returns a single cluster.

    This class implements a trivial clustering algorithm that assigns all data
    points to the same cluster. It can be used as an argument of the class
    :class:`tdamapper.core.MapperAlgorithm` to skip clustering in the
    construction of the Mapper graph.
    """

    def __init__(self):
        self.labels_ = None

    def fit(self, X, y=None):
        """
        Fit the clustering algorithm to the data.

        :param X: The dataset to be mapped.
        :type X: array-like of shape (n, m) or list-like of size n
        :param y: Ignored.
        :return: self
        """
        self.labels_ = [0 for _ in X]
        return self


class FailSafeClustering:
    """
    A delegating clustering algorithm that prevents failure.

    This class wraps a clustering algorithm and handles any exceptions that may
    occur during the fitting process. If the clustering algorithm fails, instead
    of throwing an exception, a single cluster containing all points is
    returned. This can be useful for robustness and debugging purposes.
    
    :param clustering: A clustering algorithm to delegate to.
    :type clustering: Anything compatible with a :mod:`sklearn.cluster` class.
    :param verbose: Set to `True` to log exceptions. The default is `True`.
    :type verbose: bool
    """

    def __init__(self, clustering, verbose=True):
        self.__clustering = clustering
        self.__verbose = verbose
        self.labels_ = None

    def fit(self, X, y=None):
        try:
            self.__clustering.fit(X, y)
            self.labels_ = self.__clustering.labels_
        except ValueError as err:
            if self.__verbose:
                _logger.warning('Unable to perform clustering on local chart: %s', err)
            self.labels_ = [0 for _ in X]
        return self


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
