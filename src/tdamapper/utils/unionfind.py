"""
This module implements a Union-Find data structure that supports union and
find operations.
"""

from typing import Any, Iterable


class UnionFind:
    """
    A Union-Find data structure that supports union and find operations.

    This implementation uses path compression for efficient find operations
    and union by size to keep the tree flat. It allows for efficient
    determination of connected components in a set of elements.

    :param X: An iterable of elements to initialize the Union-Find structure.
    """

    _parent: dict[Any, Any]
    _size: dict[Any, int]

    def __init__(self, items: Iterable[Any]):
        self._parent = {x: x for x in items}
        self._size = {x: 1 for x in items}

    def find(self, x: Any) -> Any:
        """
        Finds the class of an element, applying path compression.

        :param x: The element to find the class of.
        :return: The representative of the class containing x.
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

    def union(self, x: Any, y: Any) -> Any:
        """
        Unites the classes of two elements.

        :param x: The first element.
        :param y: The second element.
        :return: The representative of the class after the union operation.
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
