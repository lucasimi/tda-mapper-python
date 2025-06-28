"""
This module implements a Union-Find (Disjoint Set Union) data structure.
It provides methods to find the root of an element and to union two elements.
It is useful for efficiently managing and merging disjoint sets.
"""

from typing import Dict, Generic, Iterable, TypeVar

T = TypeVar("T")


class UnionFind(Generic[T]):
    """
    A Union-Find data structure for managing disjoint sets.

    It supports efficient union and find operations, allowing for the merging
    of sets and finding the root representative of an element. It can be used
    for network connectivity problems, clustering, and other applications where
    it is necessary to manage and query disjoint sets of elements.

    :param items: An iterable containing the initial elements to be managed by
        the Union-Find structure.
    """

    def __init__(self, items: Iterable[T]):
        self._parent: Dict[T, T] = {x: x for x in items}
        self._size = {x: 1 for x in items}

    def find(self, x: T) -> T:
        """
        Finds the root representative of the set containing element x.

        This method implements path compression to flatten the structure,
        making future queries faster by ensuring that all elements point
        directly to the root.

        :param x: The element whose root representative is to be found.
        :return: The root representative of the set containing x.
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
        Unites the sets containing elements x and y.

        If the elements are already in the same set, no action is taken. If
        they are in different sets, the smaller set is merged into the larger
        one, ensuring that the root of the larger set becomes the new root
        representative.

        :param x: The first element to be united.
        :param y: The second element to be united.
        :return: The root representative of the united set.
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
        return x
