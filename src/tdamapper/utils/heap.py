from __future__ import annotations

from typing import Any, Generic, Iterator, Optional, Protocol, TypeVar


def _left(i: int) -> int:
    return 2 * i + 1


def _right(i: int) -> int:
    return 2 * i + 2


def _parent(i: int) -> int:
    return max(0, (i - 1) // 2)


class Comparable(Protocol):

    def __lt__(self: K, other: Comparable) -> bool: ...

    def __le__(self: K, other: Comparable) -> bool: ...

    def __gt__(self: K, other: Comparable) -> bool: ...

    def __ge__(self: K, other: Comparable) -> bool: ...


K = TypeVar("K", bound=Comparable)

V = TypeVar("V")


class _HeapNode(Generic[K, V]):

    _key: K
    _value: V

    def __init__(self, key: K, value: V) -> None:
        self._key = key
        self._value = value

    def get(self) -> tuple[K, V]:
        return self._key, self._value

    def __lt__(self, other: _HeapNode[K, V]) -> bool:
        return self._key < other._key

    def __le__(self, other: _HeapNode[K, V]) -> bool:
        return self._key <= other._key

    def __gt__(self, other: _HeapNode[K, V]) -> bool:
        return self._key > other._key

    def __ge__(self, other: _HeapNode[K, V]) -> bool:
        return self._key >= other._key


class MaxHeap(Generic[K, V]):

    _heap: list[_HeapNode[K, V]]
    _iter: Iterator[_HeapNode[K, V]]

    def __init__(self) -> None:
        self._heap = []

    def __iter__(self) -> MaxHeap[K, V]:
        self._iter = iter(self._heap)
        return self

    def __next__(self) -> tuple[K, V]:
        node = next(self._iter)
        return node.get()

    def __len__(self) -> int:
        return len(self._heap)

    def top(self) -> Optional[tuple[K, V]]:
        if not self._heap:
            return None
        return self._heap[0].get()

    def pop(self) -> Optional[tuple[K, V]]:
        if not self._heap:
            return None
        max_val = self._heap[0]
        self._heap[0] = self._heap[-1]
        self._heap.pop()
        self._bubble_down()
        return max_val.get()

    def add(self, key: K, val: V) -> None:
        self._heap.append(_HeapNode(key, val))
        self._bubble_up()

    def _get_local_max(self, i: int) -> int:
        heap_len = len(self._heap)
        left = _left(i)
        right = _right(i)
        if left >= heap_len:
            return i
        if right >= heap_len:
            if self._heap[i] < self._heap[left]:
                return left
            return i
        max_child = left
        if self._heap[left] < self._heap[right]:
            max_child = right
        if self._heap[i] < self._heap[max_child]:
            return max_child
        return i

    def _fix_down(self, i: int) -> int:
        local_max = self._get_local_max(i)
        if i < local_max:
            self._heap[i], self._heap[local_max] = (
                self._heap[local_max],
                self._heap[i],
            )
            return local_max
        return i

    def _fix_up(self, i: int) -> int:
        parent = _parent(i)
        if self._heap[parent] < self._heap[i]:
            self._heap[i], self._heap[parent] = self._heap[parent], self._heap[i]
            return parent
        return i

    def _bubble_down(self) -> None:
        current = 0
        done = False
        while not done:
            local_max = self._fix_down(current)
            done = current == local_max
            current = local_max

    def _bubble_up(self) -> None:
        current = len(self._heap) - 1
        done = False
        while not done:
            local_max = self._fix_up(current)
            done = local_max == current
            current = local_max
