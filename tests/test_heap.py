import unittest
import random

from mapper.utils.heap import MaxHeap

def maxheap(data):
    m = MaxHeap()
    for x in data:
        m.insert(x)
    return m

class TestMaxHeap(unittest.TestCase):

    def testEmpty(self):
        m = MaxHeap()
        self.assertTrue(m.empty())

    def testMax(self):
        data = list(range(10))
        random.shuffle(data)
        m = maxheap(data)
        self.assertEqual(9, m.max())
        self.assertEqual(10, len(m))

    def testMaxRandom(self):
        data = random.sample(list(range(1000)), 100)
        m = maxheap(data)
        self.assertEqual(100, len(m))
        self.assertEqual(max(data), m.max())
        self.assertFalse(m.empty())
        collected = []
        for _ in range(10):
            collected.append(m.extract_max())
        data.sort()
        collected.sort()
        self.assertEqual(collected, data[-10:])
        self.assertEqual(90, len(m))
        self.assertFalse(m.empty())
