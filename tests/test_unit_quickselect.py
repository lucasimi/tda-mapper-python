import numpy as np
import pytest

from tdamapper.utils.quickselect import partition, quickselect
from tests.test_utils import list_int_random


@pytest.mark.parametrize(
    "arr, arr_extra",
    [
        (np.array([0, 1, -1]), np.array([4, 5, 6])),
        (np.array([3, 2, 1]), np.array([7, 8, 9])),
        (np.array([10, 20, 30]), np.array([11, 12, 13])),
        (np.array([-5, -10, -15]), np.array([-1, -2, -3])),
        (np.array([0, 0, 1, 6, 9, 1, 1]), np.array([-1, -2, -3, 0, 0, 0, 0])),
        (np.array(list_int_random(1000)), np.array(list_int_random(1000))),
    ],
)
def test_partition(arr, arr_extra):
    n = len(arr)
    for choice in range(n):
        h = partition(arr, 0, n, choice, arr_extra)
        for i in range(0, h):
            assert arr[i] < choice
        for i in range(h, n):
            assert arr[i] >= choice


@pytest.mark.parametrize(
    "arr, arr_extra",
    [
        (np.array([0, 1, -1]), np.array([4, 5, 6])),
        (np.array([3, 2, 1]), np.array([7, 8, 9])),
        (np.array([10, 20, 30]), np.array([11, 12, 13])),
        (np.array([-5, -10, -15]), np.array([-1, -2, -3])),
        (np.array([0, 0, 1, 6, 9, 1, 1]), np.array([-1, -2, -3, 0, 0, 0, 0])),
        (np.array(list_int_random(1000)), np.array(list_int_random(1000))),
    ],
)
def test_quickselect(arr, arr_extra):
    n = len(arr)
    for choice in range(n):
        quickselect(arr, 0, n, choice, arr_extra)
        val = arr[choice]
        for i in range(0, choice):
            assert arr[i] <= val
        for i in range(choice, n):
            assert arr[i] >= val
