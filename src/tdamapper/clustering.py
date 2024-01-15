import logging

from tdamapper.core import build_labels_par
from tdamapper.utils.unionfind import UnionFind
from tdamapper.cover import TrivialCover

_logger = logging.getLogger(__name__)

logging.basicConfig(
    format = '%(asctime)s %(module)s %(levelname)s: %(message)s',
    datefmt = '%m/%d/%Y %I:%M:%S %p',
    level = logging.INFO)


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


class CoverClustering:

    def __init__(self, cover=None):
        self.cover = cover
        self.labels_ = None

    def _check_params(self):
        if not self.cover:
            cover = TrivialCover()
        else:
            cover = self.cover
        return cover

    def fit(self, X, y=None):
        if self.cover:
            cover = self.cover
        else:
            cover = TrivialCover()
        multilabels = build_labels_par(X, X, cover, TrivialClustering(), 1)
        label_values = set()
        for labels in multilabels:
            label_values.update(labels)
        uf = UnionFind(label_values)
        self.labels_ = []
        for labels in multilabels:
            if len(labels) > 1:
                for first, second in zip(labels, labels[1:]):
                    root = uf.union(first, second)
            else:
                root = uf.find(labels[0])
            self.labels_.append(root)
        return self
