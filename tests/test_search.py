import time
import logging
import random
import sys
import unittest
import numpy as np

from mapper.search import BallSearch, KnnSearch


logger = logging.getLogger()
logger.level = logging.INFO
logger.addHandler(logging.StreamHandler(sys.stdout))


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
        bs = BallSearch(0.0005, dist)
        ks = KnnSearch(10, dist)
        t0 = time.time()
        for _ in range(times):
            bs.fit(data)
        t1 = time.time()
        for _ in range(times):
            ks.fit(data)
        t2 = time.time()
        for _ in range(times):
            x = random.choice(data)
            bsResults = bs.neighbors(x)
        t3 = time.time()
        for _ in range(times):
            x = random.choice(data)
            knnResults = ks.neighbors(x)
        t4 = time.time()
        logger.info(f'\nBall Search: {len(bsResults)} results, fit in {t1 - t0}s, search in {t3 - t2}s')
        logger.info(f'KNN Search: {len(knnResults)} results, fit in {t2 - t1}s, search in {t4 - t3}s')



if __name__ == '__main__':
    unittest.main()
