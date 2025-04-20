import time

import gtda.mapper as gm
import kmapper as km
import numpy as np
import pandas as pd
from sklearn.base import ClusterMixin
from sklearn.datasets import fetch_openml, load_digits
from sklearn.decomposition import PCA

import tdamapper as tm
from tdamapper.clustering import TrivialClustering


def _segment(cardinality, dimension, noise=0.1, start=None, end=None):
    if start is None:
        start = np.zeros(dimension)
    if end is None:
        end = np.ones(dimension)
    coefficients = np.random.rand(cardinality, 1)
    points = start + coefficients * (end - start)
    noise = np.random.normal(0, noise, size=(cardinality, dimension))
    return points + noise


def _load_openml(name):
    XX, _ = fetch_openml(name=name, return_X_y=True)
    return XX.to_numpy()


def line(k):
    return _segment(100000, k, 0.01)


def digits(k):
    X_digits, _ = load_digits(return_X_y=True)
    return PCA(k).fit_transform(X_digits)


def mnist(k):
    X = _load_openml("mnist_784")
    return PCA(k).fit_transform(X)


def cifar10(k):
    X = _load_openml("CIFAR_10")
    return PCA(k).fit_transform(X)


def fashion_mnist(k):
    X = _load_openml("Fashion-MNIST")
    return PCA(k).fit_transform(X)


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
        cover=gm.CubicalCover(n_intervals=n, overlap_frac=p),
        clusterer=TrivialEstimator(),
    )
    mapper_graph = pipe.fit_transform(X)
    t1 = time.time()
    return t1 - t0


def run_tm(X, n, p):
    t0 = time.time()
    mapper_graph = tm.core.MapperAlgorithm(
        cover=tm.cover.CubicalCover(
            n_intervals=n,
            overlap_frac=p,
            # leaf_capacity=1000,
            # leaf_radius=1.0 / (2.0 - 2.0 * p),
            # kind='hierarchical',
            # pivoting='random',
        ),
        clustering=TrivialEstimator(),
    ).fit_transform(X, X)
    t1 = time.time()
    return t1 - t0


def run_km(X, n, p):
    t0 = time.time()
    mapper = km.KeplerMapper(verbose=0)
    graph = mapper.map(
        lens=X,
        X=X,
        cover=km.Cover(n_cubes=n, perc_overlap=p),
        clusterer=TrivialEstimator(),
    )
    t1 = time.time()
    return t1 - t0


def run_bench(benches, datasets, dimensions, overlaps, intervals):
    df_bench = pd.DataFrame(
        {
            "bench": [],
            "dataset": [],
            "p": [],
            "n": [],
            "k": [],
            "time": [],
        }
    )
    launch_time = int(time.time())
    for bench_name, bench in benches:
        for dataset_name, dataset in datasets:
            for k in dimensions:
                X = dataset(k)
                for p in overlaps:
                    for n in intervals:
                        t = bench(X, n, p)
                        df_delta = pd.DataFrame(
                            {
                                "bench": bench_name,
                                "dataset": dataset_name,
                                "p": p,
                                "n": n,
                                "k": k,
                                "time": t,
                            },
                            index=[0],
                        )
                        print(df_delta)
                        df_bench = pd.concat([df_bench, df_delta], ignore_index=True)
                        df_bench.to_csv(f"./benchmark_{launch_time}.csv", index=False)


if __name__ == "__main__":
    run_tm(line(1), 1, 0.5)  # fist run to jit-compile numba decorated functions

    run_bench(
        overlaps=[0.125, 0.25, 0.5],
        datasets=[
            ("line", line),
            ("digits", digits),
            ("mnist", mnist),
            ("cifar10", cifar10),
            ("fashion_mnist", fashion_mnist),
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
            ("tda-mapper", run_tm),
            ("kepler-mapper", run_km),
            ("giotto-tda", run_gm),
        ],
    )
