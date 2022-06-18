"""A collection of functions to build open covers"""
from .search import TrivialSearch


class CoverGraph:
    
    def __init__(self, search_algo=None, clustering_algo=None):
        if not search_algo:
            self.__search_algo = TrivialSearch()
        else:
            self.__search_algo = search_algo
        if not clustering_algo:
            self.__clustering_algo = TrivialClustering()
        else:
            self.__clustering_algo = clustering_algo

    def fit_predict(self, X, y=None):
        cluster_count = 0
        self.__search_algo.fit(X)
        multilabels = [[] for _ in X]
        for i, cover_i in enumerate(multilabels):
            cover_i = multilabels[i]
            if not cover_i:
                neighs_ids = self.__search_algo.neighbors(X[i])
                neighs = [X[j] for j in neighs_ids]
                labels = self.__clustering_algo.fit_predict(neighs)
                max_label = 0
                for (n, label) in zip(neighs_ids, labels):
                    if label != -1:
                        if label > max_label:
                            max_label = label
                        multilabels[n].append(cluster_count + label)
                cluster_count += max_label + 1
        return multilabels


class InducedClustering:

    def __init__(self, search_algo=None):
        if not search_algo:
            self.__search_algo = TrivialSearch()
        else:
            self.__search_algo = search_algo

    def fit_predict(self, X, y=None):
        cover_graph = CoverGraph(search_algo=self.__search_algo)
        multilabels = cover_graph.fit_predict(X) 
        label_values = set()
        for labels in multilabels:
            label_values.update(labels)
        uf = UnionFind(label_values)
        cc = [None for _ in X]
        for i, labels in enumerate(multilabels):
            if len(labels) > 1:
                for first, second in zip(labels, labels[1:]):
                    root = uf.union(first, second)
            else:
                root = uf.find(labels[0])
            cc[i] = root
        return cc


class UnionFind:

    def __init__(self, X):
        self.__parent = {x:x for x in X}
        self.__size = {x:1 for x in X}

    def find(self, x):
        y = x
        while y != self.__parent[y]:
            y = self.__parent[y]
        return y

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


            





class TrivialClustering:

    def fit_predict(self, X, y=None):
        return [0 for _ in X]

