import time
import unittest
import numpy as np

from mapper.search import BallSearch, KnnSearch


def dist(x, y):
    return np.linalg.norm(x - y)


def dataset(dim=1, num=10000):
    return [np.random.rand(dim) for _ in range(num)]


class TestMapper(unittest.TestCase):

    def testTrivial(self):
        return
        data = dataset()
        bs = BallSearch(10, dist)
        bs.fit(data)
        for x in data:
            result = bs.neighbors(x)
            expected = [y for y in data if dist(x, y) <= 10]
            self.assertEqual(len(expected), len(result))

    def testBench(self):
        times = 10
        data = dataset()
        bs = BallSearch(10, dist)
        t0 = time.time()
        for _ in range(times):
            bs.fit(data)
        t1 = time.time()
        print(f'Build time {t1 - t0}s')
        for _ in range(times):
            for x in data:
                result = bs.neighbors(x)
                break
        t2 = time.time()
        print(f'Search time {t2 - t1}s')


if __name__ == '__main__':
    unittest.main()
