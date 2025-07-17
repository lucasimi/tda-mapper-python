import pytest

from tdamapper.utils.heap import MaxHeap
from tests.test_utils import list_int_random


def _check_heap_property(data, i=0):
    if i >= len(data):
        return
    i_left = 2 * i + 1
    if i_left < len(data):
        assert data[i] >= data[i_left]
    i_right = 2 * i + 2
    if i_right < len(data):
        assert data[i] >= data[i_right]
    _check_heap_property(data, i_left)
    _check_heap_property(data, i_right)


@pytest.mark.parametrize(
    "data",
    [
        [],
        [1, 2, 3, 4, 5],
        [5, 4, 3, 2, 1],
        [1, 1, 1, 1, 2],
        list_int_random(10),
        list_int_random(100),
        list_int_random(1000),
    ],
)
def test_max_heap(data):
    m = MaxHeap()
    for x in data:
        m.add(x, x)
    _check_heap_property(list(m))
