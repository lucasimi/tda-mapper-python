import numpy as np
from numba import njit
from numpy.typing import NDArray


@njit(fastmath=True)  # type: ignore
def euclidean(
    x: NDArray[np.float_], y: NDArray[np.float_]
) -> np.float_:  # pragma: no cover
    return np.linalg.norm(x - y)


@njit(fastmath=True)  # type: ignore
def manhattan(
    x: NDArray[np.float_], y: NDArray[np.float_]
) -> np.float_:  # pragma: no cover
    return np.linalg.norm(x - y, ord=1)


@njit(fastmath=True)  # type: ignore
def chebyshev(
    x: NDArray[np.float_], y: NDArray[np.float_]
) -> np.float_:  # pragma: no cover
    return np.linalg.norm(x - y, ord=np.inf)


@njit(fastmath=True)  # type: ignore
def minkowski(
    p: int, x: NDArray[np.float_], y: NDArray[np.float_]
) -> np.float_:  # pragma: no cover
    return np.linalg.norm(x - y, ord=p)


@njit(fastmath=True)  # type: ignore
def cosine(
    x: NDArray[np.float_], y: NDArray[np.float_]
) -> np.float_:  # pragma: no cover
    xy = np.dot(x, y)
    xx = np.linalg.norm(x)
    yy = np.linalg.norm(y)
    similarity = xy / (xx * yy)
    res = np.sqrt(2.0 * (1.0 - similarity))
    return np.float_(res)
