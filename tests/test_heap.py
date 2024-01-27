import unittest
import random
from tdamapper.utils.heap import MaxHeap


def maxheap(data):
    m = MaxHeap()
    for x in data:
        m.add(x, x)
    return m


class TestMaxHeap(unittest.TestCase):

    def testEmpty(self):
        m = MaxHeap()
        self.assertEqual(0, len(m))

    def testMax(self):
        data = list(range(10))
        random.shuffle(data)
        m = maxheap(data)
        self.assertEqual((9, 9), m.top())
        self.assertEqual(10, len(m))

    def testMaxRandom(self):
        data = random.sample(list(range(1000)), 100)
        m = maxheap(data)
        self.assertEqual(100, len(m))
        max_data = max(data)
        self.assertEqual((max_data, max_data), m.top())
        self.assertNotEqual(0, len(m))
        collected = []
        for _ in range(10):
            collected.append(m.pop())
        data.sort()
        collected.sort()
        self.assertEqual(collected, [(x, x) for x in data[-10:]])
        self.assertEqual(90, len(m))
