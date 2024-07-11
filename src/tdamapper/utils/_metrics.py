import numpy as np
from numba import njit


@njit(fastmath=True)
def euclidean(x, y):
    return np.linalg.norm(x - y)


@njit(fastmath=True)
def manhattan(x, y):
    return np.linalg.norm(x - y, ord=1)


@njit(fastmath=True)
def chebyshev(x, y):
    return np.linalg.norm(x - y, ord=np.inf)


@njit(fastmath=True)
def minkowski(p, x, y):
    return np.linalg.norm(x - y, ord=p)


@njit(fastmath=True)
def cosine(x, y):
    xy = np.dot(x, y)
    xx = np.linalg.norm(x)
    yy = np.linalg.norm(y)
    similarity = xy / (xx * yy)
    return np.sqrt(2.0 * (1.0 - similarity))
