from typing import Any, Optional, Union

import numpy as np
from numba import njit

from tdamapper._common import Array

_ARR = np.zeros(1)


@njit  # type: ignore
def swap(arr: Array[Any], i: int, j: int) -> None:  # pragma: no cover
    arr[i], arr[j] = arr[j], arr[i]


@njit  # type: ignore
def _swap_all(
    arr: Array[Any],
    i: int,
    j: int,
    extra1: Array[Any],
    use_extra1: bool,
    extra2: Array[Any],
    use_extra2: bool,
) -> None:  # pragma: no cover
    swap(arr, i, j)
    if use_extra1:
        swap(extra1, i, j)
    if use_extra2:
        swap(extra2, i, j)


@njit  # type: ignore
def _partition(
    data: Array[Any],
    start: int,
    end: int,
    p_ord: Any,
    extra1: Array[Any],
    use_extra1: bool,
    extra2: Array[Any],
    use_extra2: bool,
) -> int:  # pragma: no cover
    higher = start
    for j in range(start, end):
        j_ord = data[j]
        if j_ord < p_ord:
            _swap_all(data, higher, j, extra1, use_extra1, extra2, use_extra2)
            higher += 1
    return higher


@njit  # type: ignore
def _quickselect(
    data: Array[Any],
    start: int,
    end: int,
    k: int,
    extra1: Array[Any],
    use_extra1: bool,
    extra2: Array[Any],
    use_extra2: bool,
) -> None:  # pragma: no cover
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


def _to_array(
    extra1: Optional[Array[Any]] = None, extra2: Optional[Array[Any]] = None
) -> tuple[Array[Any], Array[Any]]:
    extra1_arr = _ARR if extra1 is None else extra1
    extra2_arr = _ARR if extra2 is None else extra2
    return extra1_arr, extra2_arr


def _use_array(
    extra1: Optional[Array[Any]] = None, extra2: Optional[Array[Any]] = None
) -> tuple[bool, bool]:
    use_extra1 = extra1 is not None
    use_extra2 = extra2 is not None
    return use_extra1, use_extra2


def swap_all(
    arr: Array[Any],
    i: int,
    j: int,
    extra1: Optional[Array[Any]] = None,
    extra2: Optional[Array[Any]] = None,
) -> None:
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


def partition(
    data: Array[Any],
    start: int,
    end: int,
    p_ord: Any,
    extra1: Optional[Array[Any]] = None,
    extra2: Optional[Array[Any]] = None,
) -> int:
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


def quickselect(
    data: Array[Any],
    start: int,
    end: int,
    k: int,
    extra1: Optional[Array[Any]] = None,
    extra2: Optional[Array[Any]] = None,
) -> None:
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
