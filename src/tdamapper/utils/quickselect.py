from numba import njit


@njit
def swap(arr, i, j):
    arr[i], arr[j] = arr[j], arr[i]


@njit
def swap_all(arr, i, j, extra1=None, extra2=None):
    swap(arr, i, j)
    if extra1 is not None:
        swap(extra1, i, j)
    if extra2 is not None:
        swap(extra2, i, j)


@njit
def partition(data, start, end, p_ord, extra1=None, extra2=None):
    higher = start
    for j in range(start, end):
        j_ord = data[j]
        if j_ord < p_ord:
            swap_all(data, higher, j, extra1, extra2)
            higher += 1
    return higher


@njit
def quickselect(data, start, end, k, extra1=None, extra2=None):
    if (k < start) or (k >= end):
        return
    start_, end_, higher = start, end, None
    while higher != k + 1:
        p = data[k]
        swap_all(data, start_, k, extra1, extra2)
        higher = partition(data, start_ + 1, end_, p, extra1, extra2)
        swap_all(data, start_, higher - 1, extra1, extra2)
        if k <= higher - 1:
            end_ = higher
        else:
            start_ = higher
