import logging
import networkx as nx
import numpy as np
from tdamapper.core import ATTR_IDS, item_labels, MapperAlgorithm
from tdamapper.utils.unionfind import UnionFind
from tdamapper.cover import TrivialCover, CubicalCover, BallCover, KNNCover


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
        itm_lbls = item_labels(X, X, cover, TrivialClustering())
        label_values = set()
        for lbls in itm_lbls:
            label_values.update(lbls)
        uf = UnionFind(label_values)
        self.labels_ = []
        for lbls in itm_lbls:
            if len(lbls) > 1:
                for first, second in zip(lbls, lbls[1:]):
                    root = uf.union(first, second)
            else:
                root = uf.find(lbls[0])
            self.labels_.append(root)
        return self


class MapperGraphClustering:

    def __init__(self,
            cover='cubical',
            n_intervals=10,
            overlap_frac=0.25,
            radius=0.5,
            neighbors=5,
            metric=euclidean):
        self.cover = cover
        self.n_intervals = n_intervals
        self.overlap_frac = overlap_frac
        self.radius = radius
        self.neighbors = neighbors
        self.metric = metric

    def fit(self, X, y=None):
        self.labels_ = self.fit_predict(X, y)
        return self

    def __get_cover(self):
        if self.cover == 'trivial':
            return TrivialCover()
        elif self.cover == 'cubical':
            return CubicalCover(n_intervals=self.n_intervals, overlap_frac=self.overlap_frac)
        elif self.cover == 'ball':
            return BallCover(radius=self.radius, metric=self.metric)
        elif self.cover == 'knn':
            return KNNCover(neighbors=self.neighbors, metric=self.metric)
        else:
            raise ValueError(f'Unknown cover algorithm {self.cover}')

    def fit_predict(self, X, y):
        cover = self.__get_cover()
        clustering = CoverClustering(self.__get_cover())
        mapper_algo = MapperAlgorithm(cover=cover, clustering=clustering)
        graph = mapper_algo.fit_transform(X, y)

        cc_id = 1
        item_cc = {}
        for cc in nx.connected_components(graph):
            for node in cc:
                for itm_id in graph.nodes[node][ATTR_IDS]:
                    item_cc[itm_id] = cc_id
            cc_id += 1
        return [item_cc[i] for i, _ in enumerate(X)]
