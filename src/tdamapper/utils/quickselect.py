"""
This module provides utility functions for quickselect and partitioning
arrays using the Numba library for performance optimization. It includes
functions to swap elements in an array, partition an array based on a pivot,
and perform the quickselect algorithm to find the k-th smallest element.
It also supports optional additional arrays for extra data that can be swapped
along with the main array.
"""

import numpy as np
from numba import njit

_ARR = np.zeros(1)


@njit  # pragma: no cover
def swap(arr, i, j):
    arr[i], arr[j] = arr[j], arr[i]


@njit  # pragma: no cover
def _swap_all(arr, i, j, extra1, use_extra1, extra2, use_extra2):
    swap(arr, i, j)
    if use_extra1:
        swap(extra1, i, j)
    if use_extra2:
        swap(extra2, i, j)


@njit  # pragma: no cover
def _partition(data, start, end, p_ord, extra1, use_extra1, extra2, use_extra2):
    higher = start
    for j in range(start, end):
        j_ord = data[j]
        if j_ord < p_ord:
            _swap_all(data, higher, j, extra1, use_extra1, extra2, use_extra2)
            higher += 1
    return higher


@njit  # pragma: no cover
def _quickselect(data, start, end, k, extra1, use_extra1, extra2, use_extra2):
    if (k < start) or (k >= end):
        return
    start_, end_, higher = start, end, -1
    while higher != k + 1:
        p = data[k]
        _swap_all(data, start_, k, extra1, use_extra1, extra2, use_extra2)
        higher = _partition(
            data, start_ + 1, end_, p, extra1, use_extra1, extra2, use_extra2
        )
        _swap_all(data, start_, higher - 1, extra1, use_extra1, extra2, use_extra2)
        if k <= higher - 1:
            end_ = higher
        else:
            start_ = higher


def _to_array(extra1=None, extra2=None):
    extra1_arr = _ARR if extra1 is None else extra1
    extra2_arr = _ARR if extra2 is None else extra2
    return extra1_arr, extra2_arr


def _use_array(extra1=None, extra2=None):
    use_extra1 = extra1 is not None
    use_extra2 = extra2 is not None
    return use_extra1, use_extra2


def swap_all(arr, i, j, extra1=None, extra2=None):
    """
    Swap elements at indices i and j in the array arr, and optionally swap
    corresponding elements in extra1 and extra2 if they are provided.

    :param arr: The array in which elements will be swapped.
    :param i: The index of the first element to swap.
    :param j: The index of the second element to swap.
    :param extra1: Optional additional array for extra data to swap.
    :param extra2: Optional additional array for another set of extra data to swap.
    :return: None. The function modifies the arrays in place.
    """
    extra1_arr, extra2_arr = _to_array(extra1, extra2)
    use_extra1, use_extra2 = _use_array(extra1, extra2)
    _swap_all(
        arr,
        i,
        j,
        extra1=extra1_arr,
        use_extra1=use_extra1,
        extra2=extra2_arr,
        use_extra2=use_extra2,
    )


def partition(data, start, end, p_ord, extra1=None, extra2=None):
    """
    Partition the data array into two parts based on the pivot value p_ord.
    Elements less than p_ord will be moved to the left, and elements greater
    than or equal to p_ord will be moved to the right.

    :param data: The array of data to be partitioned.
    :param start: The starting index of the range to consider.
    :param end: The ending index of the range to consider.
    :param p_ord: The pivot value used for partitioning.
    :param extra1: Optional additional array for extra data.
    :param extra2: Optional additional array for another set of extra data.
    :return: The index of the first element in the right partition.
    """
    extra1_arr, extra2_arr = _to_array(extra1, extra2)
    use_extra1, use_extra2 = _use_array(extra1, extra2)
    return _partition(
        data,
        start,
        end,
        p_ord,
        extra1=extra1_arr,
        use_extra1=use_extra1,
        extra2=extra2_arr,
        use_extra2=use_extra2,
    )


def quickselect(data, start, end, k, extra1=None, extra2=None):
    """
    Perform the quickselect algorithm to find the k-th smallest element in data.

    :param data: The array of data to be processed.
    :param start: The starting index of the range to consider.
    :param end: The ending index of the range to consider.
    :param k: The index of the element to find (0-based).
    :param extra1: Optional additional array for extra data.
    :param extra2: Optional additional array for another set of extra data.
    :return: None. The data array is modified in place to place the k-th smallest
        element at index k.
    """
    extra1_arr, extra2_arr = _to_array(extra1, extra2)
    use_extra1, use_extra2 = _use_array(extra1, extra2)
    _quickselect(
        data,
        start,
        end,
        k,
        extra1=extra1_arr,
        use_extra1=use_extra1,
        extra2=extra2_arr,
        use_extra2=use_extra2,
    )
