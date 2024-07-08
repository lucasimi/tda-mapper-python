import unittest
import random

from tdamapper.utils.quickselect import (
    partition,
    quickselect,
    partition_tuple,
    quickselect_tuple)


class TestQuickSelect(unittest.TestCase):

    def test_partition(self):
        n = 1000
        arr = [(i, random.randint(0, n - 1)) for i in range(n)]
        for choice in range(n):
            h = partition(arr, 0, n, choice)
            for i in range(0, h):
                self.assertTrue(arr[i][0] < choice)
            for i in range(h, n):
                self.assertTrue(arr[i][0] >= choice)

    def test_quickselect_bounds(self):
        arr = [(0, 4), (1, 5), (-1, 6)]
        quickselect(arr, 1, 2, 0)
        self.assertEqual((0, 4), arr[0])
        self.assertEqual((1, 5), arr[1])
        self.assertEqual((-1, 6), arr[2])

    def test_quickselect(self):
        n = 1000
        arr = [(i, random.randint(0, n - 1)) for i in range(n)]
        for choice in range(n):
            quickselect(arr, 0, n, choice)
            val = arr[choice][0]
            for i in range(0, choice):
                self.assertTrue(arr[i][0] <= val)
            for i in range(choice, n):
                self.assertTrue(arr[i][0] >= val)
    
    def test_partition_tuple(self):
        n = 1000
        arr_data = [random.randint(0, n - 1) for i in range(n)]
        arr_ord = list(range(n))
        for choice in range(n):
            h = partition_tuple(arr_ord, arr_data, 0, n, choice)
            for i in range(0, h):
                self.assertTrue(arr_ord[i] < choice)
            for i in range(h, n):
                self.assertTrue(arr_ord[i] >= choice)

    def test_quickselect_tuple(self):
        n = 1000
        arr_data = [random.randint(0, n - 1) for i in range(n)]
        arr_ord = list(range(n))
        for choice in range(n):
            quickselect_tuple(arr_ord, arr_data, 0, n, choice)
            val = arr_ord[choice]
            for i in range(0, choice):
                self.assertTrue(arr_ord[i] <= val)
            for i in range(choice, n):
                self.assertTrue(arr_ord[i] >= val)
    