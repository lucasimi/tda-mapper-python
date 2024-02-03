'''A module containing the logic related to clustering for the Mapper algorithm.'''
import logging

from tdamapper.core import mapper_connected_components
from tdamapper.cover import TrivialCover


_logger = logging.getLogger(__name__)

logging.basicConfig(
    format = '%(asctime)s %(module)s %(levelname)s: %(message)s',
    datefmt = '%m/%d/%Y %I:%M:%S %p',
    level = logging.INFO)


class TrivialClustering:
    '''
    A clustering algorithm that returns a single cluster.
    '''

    def __init__(self):
        self.labels_ = None

    def fit(self, X, y=None):
        self.labels_ = [0 for _ in X]
        return self


class FailSafeClustering:
    '''
    A delegating clustering algorithm that prevents failure.
    When clustering fails, instead of throwing an exception,
    a single cluster, containing all points, is returned.

    :param clustering: A clustering algorithm to delegate to.
    :type clustering: Anything compatible with a `sklearn.cluster` class.
    :param verbose: Set to `True` to log exceptions.
    :type verbose: `bool`
    '''

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
    '''
    A clustering algorithm based on the Mapper graph.
    The Mapper algorithm returns a graph where each point is eventually contained 
    in multiple nodes. In this case all those nodes are connected in the Mapper graph,
    therefore they share the same connected component. For this reason the notion of
    connected component is well-defined for any point of the dataset. This class 
    clusters point according to their connected component in the Mapper graph.

    :type cover: A cover algorithm.
    :type cover: Anything compatible with a `tdamapper.cover` class.
    :param clustering: A clustering algorithm.
    :type clustering: Anything compatible with a `sklearn.cluster` class.
    '''

    def __init__(self, cover=None, clustering=None):
        self.cover = cover
        self.clustering = clustering

    def fit(self, X, y=None):
        cover = self.cover if self.cover else TrivialCover()
        clustering = self.clustering if self.clustering else TrivialClustering()
        itm_lbls = mapper_connected_components(X, y, cover, clustering)
        self.labels_ = [itm_lbls[i] for i, _ in enumerate(X)]
        return self
