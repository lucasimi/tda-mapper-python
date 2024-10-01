import time

import pandas as pd

from sklearn.decomposition import PCA
from sklearn.base import ClusterMixin, clone

import tdamapper as tm
from tdamapper.clustering import TrivialClustering
import gtda.mapper as gm
import kmapper as km

from sklearn.datasets import fetch_openml, load_digits


def digits():
    X_digits, _ = load_digits(return_X_y=True)
    return X_digits


def _load_openml(name):
    XX, _ = fetch_openml(name=name, return_X_y=True)
    return XX.to_numpy()


def mnist():
    return _load_openml('mnist_784')


def cifar10():
    return _load_openml('CIFAR_10')


def fashion_mnist():
    return _load_openml('Fashion-MNIST')


def lens(n):
    return PCA(n)


class EstimatorWrapper(ClusterMixin):

    def get_params(self, deep=True):
        return {}

    def set_params(self, **parmeters):
        return self

    def fit(self, X, y=None):
        clust = TrivialClustering()
        self.labels_ = clust.fit(X, y).labels_
        return self


def run_gm(X, n, p, k):
    t0 = time.time()
    pipe = gm.make_mapper_pipeline(
        filter_func=lens(k),
        cover=gm.CubicalCover(
            n_intervals=n,
            overlap_frac=p),
        clusterer=EstimatorWrapper(),
        verbose=False,
    )
    mapper_graph = pipe.fit_transform(X)
    t1 = time.time()
    return t1 - t0


def run_tm(X, n, p, k):
    t0 = time.time()
    y = lens(k).fit_transform(X)
    mapper_graph1 = tm.core.MapperAlgorithm(
        cover=tm.cover.CubicalCover(
            n_intervals=n,
            overlap_frac=p),
        clustering=EstimatorWrapper(),
    ).fit_transform(X, y)
    t1 = time.time()
    return t1 - t0


def run_km(X, n, p, k):
    t0 = time.time()
    mapper = km.KeplerMapper(verbose=0)
    projected = mapper.fit_transform(X, projection=lens(k))
    cover = km.Cover(
        n_cubes=n,
        perc_overlap=p
    )
    graph = mapper.map(projected, X, cover=cover)
    t1 = time.time()
    return t1 - t0


def run_bench(benches, datasets, overlaps, intervals, dimensions):
    df_bench = pd.DataFrame({
        'bench': [],
        'dataset': [],
        'p': [],
        'n': [],
        'k': [],
        'time': [],
    })
    launch_time = int(time.time())
    for bench_name, bench in benches:
        for dataset_name, dataset in datasets:
            X = dataset()
            for p in overlaps:
                for n in intervals:
                  for k in dimensions:
                      t = bench(X, n, p, k)
                      df_delta = pd.DataFrame({
                          'bench': bench_name,
                          'dataset': dataset_name,
                          'p': p,
                          'n': n,
                          'k': k,
                          'time': t,
                      }, index=[0])
                      print(df_delta)
                      df_bench = pd.concat([df_bench, df_delta], ignore_index=True)
                      df_bench.to_csv(f'./benchmark_pca_{launch_time}.csv', index=False)


if __name__ == '__main__':
    run_tm(digits(), 1, 0.5, 1) # first run for jit-compiling numba decorated functions
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
        benches=[
            ('tda-mapper', run_tm),
            ('kepler-mapper', run_km),
            ('giotto-tda', run_gm),
        ],
        datasets=[
            ('digits', digits),
            ('mnist', mnist),
            ('cifar10', cifar10),
            ('fashion_mnist', fashion_mnist)
        ],
    )
