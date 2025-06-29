"""
Module for quickselect algorithm and related utilities.

This module provides functions for performing quickselect operations on arrays,
including partitioning and selecting the k-th smallest element. It uses Numba
to compile the functions for performance optimization. The module also includes
utility functions for swapping elements in arrays and managing additional
arrays that may be used during the quickselect process.
"""

import numpy as np
from numba import njit

_ARR = np.zeros(1)


@njit  # pragma: no cover
def swap(arr, i, j):
    """
    Swap two elements in an array.

    :param arr: The array in which to swap elements.
    :param i: The index of the first element to swap.
    :param j: The index of the second element to swap.
    :return: None. The array is modified in place.
    """
    arr[i], arr[j] = arr[j], arr[i]


@njit  # pragma: no cover
def _swap_all(arr, i, j, extra1, use_extra1, extra2, use_extra2):
    """
    Swap elements in the main array and optionally in two additional arrays.

    This function swaps elements at indices `i` and `j` in the main array,
    and if `use_extra1` is True, it also swaps elements in `extra1` at the same indices.
    Similarly, if `use_extra2` is True, it swaps elements in `extra2` at the same indices.

    :param arr: The main array in which to swap elements.
    :param i: The index of the first element to swap.
    :param j: The index of the second element to swap.
    :param extra1: An optional additional array to swap elements in.
    :param use_extra1: A boolean indicating whether to swap in `extra1`.
    :param extra2: Another optional additional array to swap elements in.
    :param use_extra2: A boolean indicating whether to swap in `extra2`.
    :return: None. The arrays are modified in place.
    """
    swap(arr, i, j)
    if use_extra1:
        swap(extra1, i, j)
    if use_extra2:
        swap(extra2, i, j)


@njit  # pragma: no cover
def _partition(data, start, end, p_ord, extra1, use_extra1, extra2, use_extra2):
    """
    Partition the array into two parts based on a pivot value.

    Elements less than the pivot are moved to the left, and elements greater
    than or equal to the pivot are moved to the right. The pivot is chosen as
    the element at index `start`.

    :param data: The array to partition.
    :param start: The starting index for the partitioning.
    :param end: The ending index for the partitioning.
    :param p_ord: The pivot value used for partitioning.
    :param extra1: An optional additional array to swap elements in.
    :param use_extra1: A boolean indicating whether to swap in `extra1`.
    :param extra2: Another optional additional array to swap elements in.
    :param use_extra2: A boolean indicating whether to swap in `extra2`.
    :return: The index of the first element greater than or equal to the pivot.
        This index indicates the boundary between elements less than the pivot
        and those greater than or equal to it.
    """
    higher = start
    for j in range(start, end):
        j_ord = data[j]
        if j_ord < p_ord:
            _swap_all(data, higher, j, extra1, use_extra1, extra2, use_extra2)
            higher += 1
    return higher


@njit  # pragma: no cover
def _quickselect(data, start, end, k, extra1, use_extra1, extra2, use_extra2):
    """
    Perform the quickselect algorithm to find the k-th smallest element in the array.

    This function modifies the array in place to bring the k-th smallest element
    to the k-th position. It uses a partitioning approach similar to quicksort.

    :param data: The array to perform quickselect on.
    :param start: The starting index for the quickselect operation.
    :param end: The ending index for the quickselect operation.
    :param k: The index of the element to select (0-based).
    :param extra1: An optional additional array to swap elements in.
    :param use_extra1: A boolean indicating whether to swap in `extra1`.
    :param extra2: Another optional additional array to swap elements in.
    :param use_extra2: A boolean indicating whether to swap in `extra2`.
    :return: None. The array is modified in place to place the k-th smallest
        element at index `k`.
    """
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
    Swap elements in the main array and optionally in two additional arrays.

    This function swaps elements at indices `i` and `j` in the main array,
    and if `extra1` is provided, it also swaps elements in `extra1` at the same indices.
    Similarly, if `extra2` is provided, it swaps elements in `extra2` at the same indices.

    :param arr: The main array in which to swap elements.
    :param i: The index of the first element to swap.
    :param j: The index of the second element to swap.
    :param extra1: An optional additional array to swap elements in.
    :param extra2: Another optional additional array to swap elements in.
    :return: None. The arrays are modified in place.
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
    Partition the array into two parts based on a pivot value.

    Elements less than the pivot are moved to the left, and elements greater
    than or equal to the pivot are moved to the right. The pivot is chosen as
    the element at index `start`.

    :param data: The array to partition.
    :param start: The starting index for the partitioning.
    :param end: The ending index for the partitioning.
    :param p_ord: The pivot value used for partitioning.
    :param extra1: An optional additional array to swap elements in.
    :param extra2: Another optional additional array to swap elements in.
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
    Perform the quickselect algorithm to find the k-th smallest element in the array.

    This function modifies the array in place to bring the k-th smallest element
    to the k-th position. It uses a partitioning approach similar to quicksort.

    :param data: The array to perform quickselect on.
    :param start: The starting index for the quickselect operation.
    :param end: The ending index for the quickselect operation.
    :param k: The index of the element to select (0-based).
    :param extra1: An optional additional array to swap elements in.
    :param extra2: Another optional additional array to swap elements in.
    :return: None. The array is modified in place to place the k-th smallest
        element at index `k`.
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
