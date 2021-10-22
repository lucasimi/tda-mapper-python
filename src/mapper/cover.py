"""A collection of functions to build open covers"""
from .utils.balltree import BallTree


class SearchCover:
    
    def __init__(self, lens, metric, search_algo):
        self.__lens = lens
        self.__metric = metric
        self.__search_algo = search_algo

    def cover(self, data):
        data_ids = list(range(len(data)))
        metric_ids = lambda i, j: self.__metric(self.__lens(data[i]), self.__lens(data[j]))
        self.__search_algo.setup(data_ids, metric_ids)
        non_covered_ids = {i for i in data_ids}
        groups = []
        while non_covered_ids:
            point_id = non_covered_ids.pop()
            neighbors_ids = self.__search_algo.find_neighbors(point_id)
            non_covered_ids.difference_update(neighbors_ids)
            if neighbors_ids:
                groups.append(neighbors_ids)
        return groups


class TrivialCover:

    def cover(self, data):
        """Return a trivial grouping of data with a single ball"""
        data_ids = list(range(len(data)))
        return [data_ids]

