import unittest
import timeit

import pandas as pd
import numpy as np

import tdamapper.utils.metrics as metrics


def euclidean_numpy(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def euclidean_numpy_linalg(a, b):
    return np.linalg.norm(a - b)


def manhattan_numpy(a, b):
    return np.sum(np.abs(a - b))


def manhattan_numpy_linalg(a, b):
    return np.linalg.norm(a - b, ord=1)


def chebyshev_numpy(a, b):
    return np.max(np.abs(a - b))


def chebyshev_numpy_linalg(a, b):
    return np.linalg.norm(a - b, ord=np.inf)


def eval_dist(X, d):
    for i in range(X.shape[0] - 1):
        d(X[i], X[i+1])


def run_dist_bench(X, d):
    return timeit.timeit(lambda: eval_dist(X, d), number=30)


def run_euclidean_bench(X):
    t_numpy = run_dist_bench(X, euclidean_numpy)
    t_numpy_linalg = run_dist_bench(X, euclidean_numpy_linalg)
    t_tdamapper = run_dist_bench(X, metrics.euclidean())
    return {
        'metric': 'euclidean',
        'numpy': t_numpy,
        'numpy_linalg': t_numpy_linalg,
        'tdamapper': t_tdamapper,
    }


def run_chebyshev_bench(X):
    t_numpy = run_dist_bench(X, chebyshev_numpy)
    t_numpy_linalg = run_dist_bench(X, chebyshev_numpy_linalg)
    t_tdamapper = run_dist_bench(X, metrics.chebyshev())
    return {
        'metric': 'chebyshev',
        'numpy': t_numpy,
        'numpy_linalg': t_numpy_linalg,
        'tdamapper': t_tdamapper,
    }


def run_manhattan_bench(X):
    t_numpy = run_dist_bench(X, manhattan_numpy)
    t_numpy_linalg = run_dist_bench(X, manhattan_numpy_linalg)
    t_tdamapper = run_dist_bench(X, metrics.manhattan())
    return {
        'metric': 'manhattan',
        'numpy': t_numpy,
        'numpy_linalg': t_numpy_linalg,
        'tdamapper': t_tdamapper,
    }


def merge(d, d_part):
    for k, v in d_part.items():
        if k not in d:
            d[k] = []
        d[k].append(v)
    return d


def run_bench(X):
    d = {}
    d_part = run_euclidean_bench(X)
    merge(d, d_part)
    d_part = run_chebyshev_bench(X)
    merge(d, d_part)
    d_part = run_manhattan_bench(X)
    merge(d, d_part)
    return pd.DataFrame(d)


class TestBenchMetrics(unittest.TestCase):

    def test_bench(self):
        X = np.random.rand(1000, 1000)
        df_bench = run_bench(X)
        print(df_bench)
