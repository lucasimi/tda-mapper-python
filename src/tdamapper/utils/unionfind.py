from typing import Any, Iterable


class UnionFind:

    _parent: dict[Any, Any]
    _size: dict[Any, int]

    def __init__(self, X: Iterable[Any]):
        self._parent = {x: x for x in X}
        self._size = {x: 1 for x in X}

    def find(self, x: Any) -> Any:
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
