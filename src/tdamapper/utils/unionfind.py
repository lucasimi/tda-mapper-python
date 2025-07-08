"""
This module implements a Union-Find (Disjoint Set Union) data structure.
It provides methods to find the root of an element and to union two elements,
effectively merging their sets. The Union-Find structure is useful for
tracking connected components in a graph or for managing equivalence relations.
It supports path compression in the `find` method and union by size in the `union`
method to optimize performance.
"""

from typing import Generic, Hashable, Iterable, TypeVar

T = TypeVar("T", bound=Hashable)


class UnionFind(Generic[T]):

    _parent: dict[T, T]
    _size: dict[T, int]

    """
    A Union-Find (Disjoint Set Union) data structure.
    It supports efficient union and find operations with path compression
    and union by size.

    :param items: An iterable of elements to initialize the Union-Find structure.
    """

    def __init__(self, items: Iterable[T]) -> None:
        self._parent = {x: x for x in items}
        self._size = {x: 1 for x in items}

    def find(self, x: T) -> T:
        """
        Find the root of the element x with path compression.

        :param x: The element whose root is to be found.
        :return: The root of the element x.
        """
        root = x
        while root != self._parent[root]:
            root = self._parent[root]
        tmp = x
        while tmp != root:
            parent = self._parent[tmp]
            self._parent[tmp] = root
            tmp = parent
        return root

    def union(self, x: T, y: T) -> T:
        """
        Union the sets containing elements x and y.
        If x and y are in different sets, merge them and return the root of the
        resulting set. If they are already in the same set, return the root.

        :param x: The first element to union.
        :param y: The second element to union.
        :return: The root of the unioned set containing x and y.
        """
        x, y = self.find(x), self.find(y)
        if x != y:
            x_size, y_size = self._size[x], self._size[y]
            if x_size < y_size:
                to_keep, to_move = y, x
            else:
                to_keep, to_move = x, y
            self._parent[to_move] = to_keep
            self._size[to_keep] = x_size + y_size
            return to_keep
        else:
            return x
