from sklearn.utils import check_X_y
import numpy as np

from tdamapper.core import MapperAlgorithm, build_connected_components
from tdamapper.cover import TrivialCover, GridCover, BallCover, KNNCover
from tdamapper.clustering import TrivialClustering, CoverClustering


def euclidean(x, y):
    return np.linalg.norm(x - y)


class MapperEstimator:

    def __init__(self,
            cover='grid',
            n_intervals=10,
            overlap_frac=0.25,
            radius=0.5,
            neighbors=5,
            metric=euclidean,
            n_jobs=1):
        self.cover = cover
        self.n_intervals = n_intervals
        self.overlap_frac = overlap_frac
        self.radius = radius
        self.neighbors = neighbors
        self.metric = metric
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        self.labels_ = self.fit_predict(X, y)
        self.n_features_in_ = X.shape[1]
        return self

    def __get_cover(self):
        if self.cover == 'trivial':
            return TrivialCover()
        elif self.cover == 'grid':
            return GridCover(n_intervals=self.n_intervals, overlap_frac=self.overlap_frac)
        elif self.cover == 'ball':
            return BallCover(radius=self.radius, metric=self.metric)
        elif self.cover == 'knn':
            return KNNCover(neighbors=self.neighbors, metric=self.metric)
        else:
            raise ValueError(f'Unknown cover algorithm {self.cover}')

    def fit_predict(self, X, y):
        X, y = check_X_y(X, y)
        cover = self.__get_cover()
        clustering = CoverClustering(self.__get_cover())
        mapper_algo = MapperAlgorithm(cover=cover, clustering=clustering, n_jobs=self.n_jobs)
        graph = mapper_algo.fit_transform(X, y)
        ccs = build_connected_components(graph)
        return [ccs[i] for i, _ in enumerate(X)]

    def get_params(self, deep=True):
        params = {}
        params['cover'] = self.cover
        params['n_intervals'] = self.n_intervals
        params['overlap_frac'] = self.overlap_frac
        params['radius'] = self.radius
        params['neighbors'] = self.neighbors
        params['metric'] = self.metric
        params['n_jobs'] = self.n_jobs
        return params

    def set_params(self, **parameters):
        self.cover = parameters.get('cover', 'grid')
        self.n_intervals = parameters.get('n_intervals', 10)
        self.overlap_frac = parameters.get('overlap_frac', 0.25)
        self.radius = parameters.get('radius', 0.5)
        self.neighbors = parameters.get('neighbors', 5)
        self.metric = parameters.get('metric', euclidean)
        self.n_jobs = parameters.get('n_jobs', 1)
        return self
