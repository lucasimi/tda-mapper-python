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
    It supports efficient union and find operations, allowing for the
    merging of sets and finding the root representative of an element.
    It is particularly useful in algorithms that require dynamic connectivity,
    such as Kruskal's algorithm for minimum spanning trees or in network connectivity problems.

    :param X: An iterable containing the initial elements to be managed by the Union-Find structure.
    :type X: iterable
    """

    def __init__(self, X: Iterable[T]):
        self._parent: Dict[T, T] = {x: x for x in X}
        self._size = {x: 1 for x in X}

    def find(self, x: T) -> T:
        """
        Finds the root representative of the set containing element x.
        This method implements path compression to flatten the structure,
        making future queries faster by ensuring that all elements point directly to the root.
        :param x: The element whose root representative is to be found.
        :type x: Any
        :return: The root representative of the set containing x.
        :rtype: Any
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
        If the elements are already in the same set, no action is taken.
        If they are in different sets, the smaller set is merged into the larger set,
        ensuring that the root of the larger set becomes the new root representative.
        :param x: The first element to be united.
        :param y: The second element to be united.
        :type x: Any
        :type y: Any
        :return: The root representative of the united set.
        :rtype: Any
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
