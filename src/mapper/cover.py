from mapper.core import *


class BallCover:

    def __init__(self, radius, metric): 
        self.__radius = radius 
        self.__metric = metric 

    def charts(self, X): 
        search = BallSearch(self.__radius, self.__metric)
        return generate_charts(X, search)

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
        return generate_charts(X, search)

    def get_params(self, deep=True):
        parameters = {}
        parameters['neighbors'] = self.__neighbors
        parameters['metric'] = self.__metric
        return parameters


class TrivialCover:

    def __init__(self): 
        pass

    def charts(self, X): 
        search = TrivialSearch()
        return generate_charts(X, search)

    def get_params(self, deep=True):
        parameters = {}
        return parameters