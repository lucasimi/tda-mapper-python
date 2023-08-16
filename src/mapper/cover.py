from mapper.core import build_charts
from mapper.search import BallSearch, KnnSearch, TrivialSearch, CubicSearch


class BallCover:

    def __init__(self, radius, metric):
        self.__radius = radius
        self.__metric = metric

    def charts(self, X):
        search = BallSearch(self.__radius, self.__metric)
        return build_charts(X, search)

    def get_params(self, deep=True):
        parameters = {}
        parameters['radius'] = self.__radius
        parameters['metric'] = self.__metric
        return parameters


class KnnCover:

    def __init__(self, neighbors, metric):
        self.__neighbors = neighbors
        self.__metric = metric

    def charts(self, X):
        search = KnnSearch(self.__neighbors, self.__metric)
        return build_charts(X, search)

    def get_params(self, deep=True):
        parameters = {}
        parameters['neighbors'] = self.__neighbors
        parameters['metric'] = self.__metric
        return parameters


class CubicCover:

    def __init__(self, n, perc):
        self.__n = n
        self.__perc = perc

    def charts(self, X):
        search = CubicSearch(self.__n, self.__perc)
        return build_charts(X, search)

    def get_params(self, deep=True):
        parameters = {}
        parameters['n'] = self.__n
        parameters['perc'] = self.__perc
        return parameters


class TrivialCover:

    def __init__(self):
        pass

    def charts(self, X): 
        search = TrivialSearch()
        return build_charts(X, search)

    def get_params(self, deep=True):
        parameters = {}
        return parameters
