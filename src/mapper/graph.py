"""Classes for storing the mapper graph"""
import math
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse

from .utils.balltree import BallTree

EDGE_COLOR = 'rgba(1, 1, 1, 0.5)'
VERTEX_BORDER_COLOR = '#111'


class MeanStats:

    def __init__(self, point, value, color):
        self.__point = point
        self.__value = value
        self.__color = color

    def get_color(self):
        return self.__color

    def get_point(self):
        return self.__point

    def get_value(self):
        return self.__value


class Vertex:
    """A class representing a cluster as a vertex of the mapper graph"""

    def __init__(self, data=None, size=0):
        self.__data = data
        self.__size = size

    def get_data(self):
        return self.__data

    def get_size(self):
        return self.__size

    def set_data(self, data):
        self.__data = data

    def set_size(self, size):
        self.__size = size


class Edge:
    """A class representing a directed edge between two clusters (as vertices)"""

    def __init__(self, weight, union, intersection):
        self.__weight = weight
        self.__union = union
        self.__intersection = intersection

    def get_weight(self):
        """Return the weight of the edge"""
        return self.__weight

    def get_similarity(self):
        """Return the similarity between the cluster represented by source and target vertices"""
        return 1.0 - self.__intersection / self.__union

    def set_union(self, union):
        self.__union = union

    def set_intersection(self, intersection):
        self.__intersection = intersection


class Graph:
    "A class representing a mapper graph"

    def __init__(self):
        self.__adjaciency = {}
        self.__vertices = {}
        self.__vertex_points = {}
        self.__edges = {}
        self.__labels = None
        self.__tree = None

    def _build_tree(self, metric, lens):
        data = [np.array(v.get_data().get_point()) for v in self.__vertices.values()]
        pullback = lambda x, y : metric(lens(x), lens(y))
        self.__tree = BallTree(pullback, data)

    def add_vertex(self, vertex_id, vertex, points):
        """Add a new vertex to the graph"""
        self.__adjaciency[vertex_id] = []
        self.__vertices[vertex_id] = vertex
        self.__vertex_points[vertex_id] = points

    def add_edge(self, source_id, target_id, edge):
        """Add a new edge to the graph"""
        self.__adjaciency[source_id].append(target_id)
        self.__edges[(source_id, target_id)] = edge

    def get_vertices(self):
        """Return the ids of the vertices of the graph"""
        return self.__adjaciency.keys()

    def get_vertex(self, vertex_id):
        """Return the vertex for a given id"""
        return self.__vertices[vertex_id]

    def get_points(self, vertex_id):
        return self.__vertex_points[vertex_id]

    def get_adjaciency(self, vertex_id):
        """Return the adjaciency list of a given vertex"""
        return self.__adjaciency[vertex_id]

    def get_edge(self, source_id, target_id):
        """Return the edge for two specified vertices"""
        return self.__edges[(source_id, target_id)]

    def _predict(self, x_value):
        nn = self.__tree.nn_search(np.array(x_value))
        return nn

    def compute_labels(self):
        self.__labels = {u_id: None for u_id in self.__vertices}
        label_count = 0
        for u_id in self.__vertices:
            if not self.__labels[u_id]:
                label_count += 1
                self._set_vertex_label(u_id, label_count)

    def _set_vertex_label(self, u_id, label_count):
        if not self.__labels[u_id]:
            self.__labels[u_id] = label_count
            for v_id in self.__adjaciency[u_id]:
                self._set_vertex_label(v_id, label_count)

    def get_point_labels(self):
        point_label = {}
        for u_id in self.__vertices:
            u_label = self.__labels[u_id]
            for point in self.__vertex_points[u_id]:
                point_label[point] = u_label
        return point_label

    def get_vertex_label(self, vertex_id):
        return self.__labels[vertex_id]

    def test_kpi(self, metric, lens, test_set, kpi):
        errs = []
        if not self.__tree:
            self._build_tree(metric, lens)
        for x in test_set:
            x_pred = self._predict(x)
            err = kpi(x, x_pred)
            errs.append(err)
        return errs

    def test_mape(self, metric, lens, test_set):
        return self.test_kpi(metric, lens, test_set, mape)

    def test_mae(self, metric, lens, test_set):
        return self.test_kpi(metric, lens, test_set, mae)

    def test_rmse(self, metric, lens, test_set):
        return self.test_kpi(metric, lens, test_set, lambda x, x_pred: math.sqrt(mse(x, x_pred)))
