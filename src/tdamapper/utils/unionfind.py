class UnionFind:

    def __init__(self, X):
        self.__parent = {x: x for x in X}
        self.__size = {x: 1 for x in X}

    def find(self, x):
        root = x
        while root != self.__parent[root]:
            root = self.__parent[root]
        tmp = x
        while tmp != root:
            parent = self.__parent[tmp]
            self.__parent[tmp] = root
            tmp = parent
        return root

    def union(self, x, y):
        x, y = self.find(x), self.find(y)
        if x != y:
            x_size, y_size = self.__size[x], self.__size[y]
            if x_size < y_size:
                to_keep, to_move = y, x
            else:
                to_keep, to_move = x, y
            self.__parent[to_move] = to_keep
            self.__size[to_keep] = x_size + y_size
            return to_keep
        else:
            return x
