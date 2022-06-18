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


class TrivialClustering:

    def fit_predict(self, X, y=None):
        return [0 for _ in X]

