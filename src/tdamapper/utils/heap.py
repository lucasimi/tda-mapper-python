"""
This module implements a max-heap data structure. It provides methods to add
elements, pop the maximum element, and retrieve the top element without
removing it. It is designed to be used in scenarios where you need to maintain
a collection of elements and frequently access the maximum element
efficiently.
"""

from __future__ import annotations

from typing import Any, Generic, Iterator, Optional, Protocol, TypeVar


class Comparable(Protocol):
    """
    A protocol that defines the methods required for an object to be
    orderable. This is used to ensure that the keys in the heap can be
    compared with each other.
    """

    def __lt__(self: K, other: K) -> bool: ...

    def __le__(self: K, other: K) -> bool: ...

    def __gt__(self: K, other: K) -> bool: ...

    def __ge__(self: K, other: K) -> bool: ...


K = TypeVar("K", bound=Comparable)

V = TypeVar("V")


def _left(i: int) -> int:
    """
    Returns the index of the left child of the node at index i in a binary heap.

    :param i: The index of the parent node.
    :return: The index of the left child.
    """
    return 2 * i + 1


def _right(i: int) -> int:
    """
    Returns the index of the right child of the node at index i in a binary heap.

    :param i: The index of the parent node.
    :return: The index of the right child.
    """
    return 2 * i + 2


def _parent(i: int) -> int:
    """
    Returns the index of the parent node of the node at index i in a binary heap.

    :param i: The index of the child node.
    :return: The index of the parent node.
    """
    return max(0, (i - 1) // 2)


class _HeapNode(Generic[K, V]):
    """
    A private class representing a node in the max-heap. Each node contains a
    key and a value. The key is used to determine the order of the nodes in
    the heap, with larger keys being prioritized over smaller keys. The value
    can be any associated data that you want to store with the key.

    :param key: The key of the node, used for ordering in the heap.
    :param value: The value associated with the key.
    """

    def __init__(self, key: K, value: V) -> None:
        self._key = key
        self._value = value

    def get(self) -> tuple[K, V]:
        """
        Returns the key and value of the node as a tuple.

        :return: A tuple containing the key and value of the node.
        """
        return self._key, self._value

    def __lt__(self, other: _HeapNode[K, Any]) -> bool:
        return self._key < other._key

    def __le__(self, other: _HeapNode[K, Any]) -> bool:
        return self._key <= other._key

    def __gt__(self, other: _HeapNode[K, Any]) -> bool:
        return self._key > other._key

    def __ge__(self, other: _HeapNode[K, Any]) -> bool:
        return self._key >= other._key


class MaxHeap(Generic[K, V]):
    """
    A max-heap implementation that allows for efficient retrieval and removal
    of the maximum element. The heap is implemented as a list of _HeapNode
    objects, where each node contains a key and a value. The key is used to
    determine the order of the elements in the heap, with larger keys being
    prioritized over smaller keys.
    """

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

    def top(self) -> tuple[Optional[K], Optional[V]]:
        """
        Returns the maximum element of the heap without removing it.

        :return: A tuple containing the key and value of the maximum element,
            or (None, None) if the heap is empty.
        """
        if not self._heap:
            return (None, None)
        return self._heap[0].get()

    def pop(self) -> tuple[Optional[K], Optional[V]]:
        """
        Removes and returns the maximum element from the heap.

        :return: A tuple containing the key and value of the maximum element,
            or (None, None) if the heap is empty.
        """
        if not self._heap:
            return (None, None)
        max_val = self._heap[0]
        self._heap[0] = self._heap[-1]
        self._heap.pop()
        self._bubble_down()
        return max_val.get()

    def add(self, key: K, val: V) -> None:
        """
        Adds a new element to the heap.

        :param key: The key of the element to be added, which determines its
            position in the heap.
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
