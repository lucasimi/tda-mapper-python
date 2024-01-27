import unittest
import random
from tdamapper.utils.quickselect import partition, quickselect


class TestQuickSelect(unittest.TestCase):

    def testPartition(self):
        n = 1000
        arr = [random.randint(0, n - 1) for _ in range(n)]
        for choice in range(n):
            h = partition(arr, 0, n, choice)
            for i in range(0, h):
                self.assertTrue(arr[i] < choice)
            for i in range(h, n):
                self.assertTrue(arr[i] >= choice)

    def testQuickSelect(self):
        n = 1000
        arr = [random.randint(0, n - 1) for _ in range(n)]
        for choice in range(n):
            quickselect(arr, 0, n, choice)
            val = arr[choice]
            for i in range(0, choice):
                self.assertTrue(arr[i] <= val)
            for i in range(choice, n):
                self.assertTrue(arr[i] >= val)
    