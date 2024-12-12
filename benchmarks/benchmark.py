import time

import pandas as pd
import numpy as np

from umap import UMAP

from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml, load_digits
from sklearn.base import ClusterMixin

from tdamapper.core import TrivialClustering
from tdamapper.utils.metrics import chebyshev
from tdamapper.cover import BallCover, ProximityCubicalCover
from tdamapper.learn import MapperAlgorithm

import gtda.mapper as gm
import kmapper as km


random_state = 42


def _segment(cardinality, dimension, noise=0.1, start=None, end=None):
    if start is None:
        start = np.zeros(dimension)
    if end is None:
        end = np.ones(dimension)
    coefficients = np.random.rand(cardinality, 1)
    points = start + coefficients * (end - start)
    noise = np.random.normal(0, noise, size=(cardinality, dimension))
    return points + noise


def _fetch_openml(name):
    XX, _ = fetch_openml(name=name, return_X_y=True, parser='auto')
    return XX.to_numpy()


def line(k):
    return _segment(100000, k, 0.01)


def load_digits_pca(k):
    X_digits, _ = load_digits(return_X_y=True)
    pca = PCA(n_components=k, random_state=random_state)
    return pca.fit_transform(X_digits)


def load_digits_umap(k):
    X_digits, _ = load_digits(return_X_y=True)
    um = UMAP(n_components=k, random_state=random_state)
    return um.fit_transform(X_digits)


def load_mnist_pca(k):
    X = _fetch_openml('mnist_784')
    pca = PCA(n_components=k, random_state=random_state)
    return pca.fit_transform(X)


def load_mnist_umap(k):
    X = _fetch_openml('mnist_784')
    um = UMAP(n_components=k, random_state=random_state)
    return um.fit_transform(X)


def load_cifar10_pca(k):
    X = _fetch_openml('CIFAR_10')
    pca = PCA(n_components=k, random_state=random_state)
    return pca.fit_transform(X)


def load_cifar10_umap(k):
    X = _fetch_openml('CIFAR_10')
    um = UMAP(n_components=k, random_state=random_state)
    return um.fit_transform(X)


def load_fmnist_pca(k):
    X = _fetch_openml('Fashion-MNIST')
    pca = PCA(n_components=k, random_state=random_state)
    return pca.fit_transform(X)


def load_fmnist_umap(k):
    X = _fetch_openml('Fashion-MNIST')
    um = UMAP(n_components=k, random_state=random_state)
    return um.fit_transform(X)


# wrapper class to supply trivial clustering to giotto-tda
class TrivialEstimator(ClusterMixin):

    def get_params(self, deep=True):
        return {}

    def set_params(self, **parmeters):
        return self

    def fit(self, X, y=None):
        clust = TrivialClustering()
        self.labels_ = clust.fit(X, y).labels_
        return self


def run_gm(X, n, p):
    t0 = time.time()
    pipe = gm.make_mapper_pipeline(
        filter_func=lambda x: x,
        cover=gm.CubicalCover(
            n_intervals=n,
            overlap_frac=p,
        ),
        clusterer=TrivialEstimator(),
    )
    mapper_graph = pipe.fit_transform(X)
    t1 = time.time()
    return t1 - t0


def run_tm_cubical(X, n, p):
    t0 = time.time()
    mapper_graph = MapperAlgorithm(
        cover=ProximityCubicalCover(
            n_intervals=n,
            overlap_frac=p,
        ),
        clustering=TrivialEstimator(),
        verbose=False,
    ).fit_transform(X, X)
    t1 = time.time()
    return t1 - t0


def run_tm_ball(X, n, p):
    r = 1.0 / (2.0 - 2.0 * p)
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    X_range = X_max - X_min
    cheby = chebyshev()

    def _scale(x):
        return n * (x - X_min) / X_range

    def _scaled_chebyshev(x, y):
        return cheby(_scale(x), _scale(y))
    t0 = time.time()
    mapper_graph = MapperAlgorithm(
        cover=BallCover(
            radius=r,
            metric=_scaled_chebyshev,
        ),
        clustering=TrivialEstimator(),
        verbose=False,
    ).fit_transform(X, X)
    t1 = time.time()
    return t1 - t0


def run_km(X, n, p):
    t0 = time.time()
    mapper = km.KeplerMapper(verbose=0)
    graph = mapper.map(
        lens=X,
        X=X,
        cover=km.Cover(
            n_cubes=n,
            perc_overlap=p
        ),
        clusterer=TrivialEstimator(),
    )
    t1 = time.time()
    return t1 - t0


def run_bench(benches, datasets, dimensions, overlaps, intervals):
    df_bench = pd.DataFrame({
        'bench': [],
        'dataset': [],
        'p': [],
        'n': [],
        'k': [],
        'time': [],
    })
    launch_time = int(time.time())
    for dataset_name, dataset in datasets:
        for p in overlaps:
            for n in intervals:
                for k in dimensions:
                    X = dataset(k)
                    for bench_name, bench in benches:
                        t = bench(X, n, p)
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
                        df_bench.to_csv(f'./benchmark_{launch_time}.csv', index=False)


if __name__ == '__main__':
    # fist run to jit-compile numba decorated functions
    run_tm_cubical(line(1), 1, 0.5)
    run_tm_ball(line(1), 1, 0.5)

    bench_params = dict(
        overlaps=[
            0.125,
            0.25,
            0.5
        ],
        datasets=[
            ('line', line),
            ('digits_pca', load_digits_pca),
            ('mnist_pca', load_mnist_pca),
            ('cifar10_pca', load_cifar10_pca),
            ('fashion_mnist_pca', load_fmnist_pca),
            ('digits_umap', load_digits_umap),
            ('mnist_umap', load_mnist_umap),
            ('cifar10_umap', load_cifar10_umap),
            ('fashion_mnist_umap', load_fmnist_umap),
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
            ('tda-mapper-cubical', run_tm_cubical),
            ('tda-mapper-ball', run_tm_ball),
            ('kepler-mapper', run_km),
            ('giotto-tda', run_gm),
        ],
    )

    run_bench(**bench_params)
