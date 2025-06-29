"""
This module implements a max-heap data structure.
It provides methods to add elements, pop the maximum element,
and iterate over the elements in the heap.
"""

from __future__ import annotations

from typing import Generic, Optional, Tuple, TypeVar


def _left(i: int) -> int:
    return 2 * i + 1


def _right(i: int) -> int:
    return 2 * i + 2


def _parent(i: int) -> int:
    return max(0, (i - 1) // 2)


K = TypeVar("K")
T = TypeVar("T")


class _HeapNode(Generic[K, T]):
    """
    A node in the max-heap, storing a key and a value.

    It provides methods to get the key and value, and to compare nodes.
    The comparison is based on the key, allowing the heap to maintain
    the max-heap property. It is important to note that the key must be
    comparable with other keys in the heap for the heap to function correctly.

    :param key: The key used for ordering in the heap.
    :param value: The value associated with the key.
    """

    def __init__(self, key: K, value: T):
        self._key = key
        self._value = value

    def get(self) -> Tuple[K, T]:
        """
        Returns the key and value of the node.
        :return: A tuple containing the key and value.
        :rtype: tuple
        """
        return self._key, self._value

    def __lt__(self, other: _HeapNode[K, T]) -> bool:
        return self._key < other._key

    def __le__(self, other: _HeapNode[K, T]) -> bool:
        return self._key <= other._key

    def __gt__(self, other: _HeapNode[K, T]) -> bool:
        return self._key > other._key

    def __ge__(self, other: _HeapNode[K, T]) -> bool:
        return self._key >= other._key


class MaxHeap(Generic[K, T]):
    """
    A max-heap data structure that allows for efficient retrieval of the maximum element.

    It supports adding elements, popping the maximum element, and iterating over the elements.
    It is important to note that the keys used in the heap must be comparable with each other.
    """

    def __init__(self):
        self._heap = []
        self._iter = None

    def __iter__(self) -> MaxHeap[K, T]:
        self._iter = iter(self._heap)
        return self

    def __next__(self) -> Tuple[K, T]:
        node = next(self._iter)
        return node.get()

    def __len__(self) -> int:
        return len(self._heap)

    def top(self) -> Tuple[Optional[K], Optional[T]]:
        """
        Returns the maximum element in the heap without removing it.

        :return: A tuple containing the key and value of the maximum element, or (None, None) if the heap is empty.
        """
        if not self._heap:
            return (None, None)
        return self._heap[0].get()

    def pop(self) -> Optional[Tuple[K, T]]:
        """
        Removes and returns the maximum element from the heap.

        :return: A tuple containing the key and value of the maximum element, or None if the heap is empty.
        :rtype: tuple
        """
        if not self._heap:
            return None
        max_val = self._heap[0]
        self._heap[0] = self._heap[-1]
        self._heap.pop()
        self._bubble_down()
        return max_val.get()

    def add(self, key: K, val: T) -> None:
        """
        Adds a new element to the heap with the specified key and value.

        :param key: The key used for ordering in the heap.
        :param val: The value associated with the key.
        :return: None
        """
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
            self._heap[i], self._heap[parent] = (
                self._heap[parent],
                self._heap[i],
            )
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
