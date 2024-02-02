import logging

import numpy as np

from tdamapper.core import mapper_connected_components
from tdamapper.cover import TrivialCover


_logger = logging.getLogger(__name__)

logging.basicConfig(
    format = '%(asctime)s %(module)s %(levelname)s: %(message)s',
    datefmt = '%m/%d/%Y %I:%M:%S %p',
    level = logging.INFO)


def euclidean(x, y):
    return np.linalg.norm(x - y)


class TrivialClustering:

    def __init__(self):
        self.labels_ = None

    def fit(self, X, y=None):
        self.labels_ = [0 for _ in X]
        return self


class PermissiveClustering:

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

    def __init__(self, cover=None, clustering=None):
        self.cover = cover
        self.clustering = clustering

    def fit(self, X, y=None):
        cover = self.cover if self.cover else TrivialCover()
        clustering = self.clustering if self.clustering else TrivialClustering()
        itm_lbls = mapper_connected_components(X, y, cover, clustering)
        self.labels_ = [itm_lbls[i] for i, _ in enumerate(X)]
        return self
