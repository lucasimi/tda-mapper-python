import random

import numpy as np

from tdamapper.utils.quickselect import partition, quickselect


def test_partition():
    n = 1000
    arr = np.array([i for i in range(n)])
    arr_extra = np.array([random.randint(0, n - 1) for i in range(n)])
    for choice in range(n):
        h = partition(arr, 0, n, choice, arr_extra)
        for i in range(0, h):
            assert arr[i] < choice
        for i in range(h, n):
            assert arr[i] >= choice


def test_quickselect_bounds():
    arr = np.array([0, 1, -1])
    arr_extra = np.array([4, 5, 6])
    quickselect(arr, 1, 2, 0, arr_extra)
    assert 0 == arr[0]
    assert 1 == arr[1]
    assert -1 == arr[2]
    assert 4 == arr_extra[0]
    assert 5 == arr_extra[1]
    assert 6 == arr_extra[2]


def test_quickselect():
    n = 1000
    arr = np.array([i for i in range(n)])
    arr_extra = np.array([random.randint(0, n - 1) for i in range(n)])
    for choice in range(n):
        quickselect(arr, 0, n, choice, arr_extra)
        val = arr[choice]
        for i in range(0, choice):
            assert arr[i] <= val
        for i in range(choice, n):
            assert arr[i] >= val


def test_partition_tuple():
    n = 1000
    arr_data = np.array([random.randint(0, n - 1) for i in range(n)])
    arr_ord = np.array(list(range(n)))
    for choice in range(n):
        h = partition(arr_ord, 0, n, choice, arr_data)
        for i in range(0, h):
            assert arr_ord[i] < choice
        for i in range(h, n):
            assert arr_ord[i] >= choice


def test_quickselect_tuple():
    n = 1000
    arr_data = np.array([random.randint(0, n - 1) for i in range(n)])
    arr_ord = np.array(list(range(n)))
    for choice in range(n):
        quickselect(arr_ord, 0, n, choice, arr_data)
        val = arr_ord[choice]
        for i in range(0, choice):
            assert arr_ord[i] <= val
        for i in range(choice, n):
            assert arr_ord[i] >= val
