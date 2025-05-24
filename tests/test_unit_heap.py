import random

from tdamapper.utils.heap import MaxHeap


def maxheap(data):
    m = MaxHeap()
    for x in data:
        m.add(x, x)
    return m


def test_empty():
    m = MaxHeap()
    assert 0 == len(m)


def test_max():
    data = list(range(10))
    random.shuffle(data)
    m = maxheap(data)
    assert (9, 9) == m.top()
    assert 10 == len(m)


def test_max_random():
    data = random.sample(list(range(1000)), 100)
    m = maxheap(data)
    assert 100 == len(m)
    max_data = max(data)
    assert (max_data, max_data) == m.top()
    assert 0 != len(m)
    collected = []
    for _ in range(10):
        collected.append(m.pop())
    data.sort()
    collected.sort()
    assert collected == [(x, x) for x in data[-10:]]
    assert 90 == len(m)
