import time

import pandas as pd
import numpy as np

from sklearn.base import ClusterMixin, clone
from sklearn.base import BaseEstimator, TransformerMixin

from tdamapper.clustering import TrivialClustering

import tdamapper as tm
import gtda.mapper as gm
import kmapper as km
from sklearn.preprocessing import FunctionTransformer
from gtda.mapper import Projection


def segment(cardinality, dimension, noise=0.1, start=None, end=None):
    if start is None:
        start = np.zeros(dimension)
    if end is None:
        end = np.ones(dimension)
    coefficients = np.random.rand(cardinality, 1)
    points = start + coefficients * (end - start)
    noise = np.random.normal(0, noise, size=(cardinality, dimension))
    return points + noise


class EstimatorWrapper(ClusterMixin):

    def get_params(self, deep=True):
        return {}

    def set_params(self, **parmeters):
        return self

    def fit(self, X, y=None):
        clust = TrivialClustering()
        self.labels_ = clust.fit(X, y).labels_
        return self


class IdentityTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


def run_gm(X, n, p):
    t0 = time.time()
    pipe = gm.make_mapper_pipeline(
        filter_func=IdentityTransformer(),
        cover=gm.CubicalCover(
            n_intervals=n,
            overlap_frac=p),
        clusterer=EstimatorWrapper(),
    )
    mapper_graph = pipe.fit_transform(X)
    t1 = time.time()
    return t1 - t0


def run_tm(X, n, p):
    t0 = time.time()
    mapper_graph = tm.core.MapperAlgorithm(
        cover=tm.cover.CubicalCover(
            n_intervals=n,
            overlap_frac=p),
        clustering=EstimatorWrapper(),
    ).fit_transform(X, X)
    t1 = time.time()
    return t1 - t0


def run_km(X, n, p):
    t0 = time.time()
    mapper = km.KeplerMapper(verbose=0)
    projected = mapper.fit_transform(X, projection=lambda x: x)
    cover = km.Cover(
        n_cubes=n,
        perc_overlap=p
    )
    graph = mapper.map(projected, X, cover=cover)
    t1 = time.time()
    return t1 - t0


def run_bench(benches, dimensions, overlaps, intervals):
    df_bench = pd.DataFrame({
        'bench': [],
        'p': [],
        'n': [],
        'k': [],
        'time': [],
    })
    launch_time = int(time.time())
    for bench_name, bench in benches:
        for k in dimensions:
            X = segment(10000, k, 0.1)
            for p in overlaps:
                for n in intervals:
                    t = bench(X, n, p)
                    df_delta = pd.DataFrame({
                        'bench': bench_name,
                        'p': p,
                        'n': n,
                        'k': k,
                        'time': t,
                      }, index=[0])
                    print(df_delta)
                    df_bench = pd.concat([df_bench, df_delta], ignore_index=True)
                    df_bench.to_csv(f'./benchmark_line_{launch_time}.csv', index=False)


if __name__ == '__main__':
    run_tm(segment(1000, 1, 0.1), 1, 0.5) # fist run to jit-compile numba decorated functions
    run_bench(
        overlaps=[
            0.125,
            0.25,
            0.5
        ],
        intervals=[
            10,
        ],
        dimensions=[
            1,
            2,
            3,
            4,
            5,
        ],
        benches = [
            ('tda-mapper', run_tm),
            ('kepler-mapper', run_km),
            ('giotto-tda', run_gm),
        ],
    )
