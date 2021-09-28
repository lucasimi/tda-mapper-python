"""Classes for storing the mapper graph"""
import math
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse

from .utils.balltree import BallTree

EDGE_COLOR = 'rgba(1, 1, 1, 0.5)'
VERTEX_BORDER_COLOR = '#111'


class Vertex:
    """A class representing a cluster as a vertex of the mapper graph"""

    def __init__(self, ids):
        self.__ids = ids

    def get_ids(self):
        return self.__ids

    def get_size(self):
        return len(self.__ids)


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


class GraphColormap:

    def __init__(self, colormap=np.nanmean, aggfunc=np.nanmean):
        self.__colormap = colormap
        self.__aggfunc = aggfunc

    def compute(self, graph, data):
        colors = {}
        for u in graph.get_vertices():
            u_points = [data[i] for i in graph.get_points(u)]
            u_color = self.__aggfunc([self.__colormap(x) for x in u_points])
            colors[u] = u_color
        return colors


class GraphStats:

    def __init__(self, lens, metric, aggfunc=lambda x: np.nanmean(x, axis=0)):
        self.__aggfunc = aggfunc
        self.__lens = lens
        self.__metric = metric
        self.__tree = None

    def compute(self, graph, data):
        stats = {}
        for u in graph.get_vertices():
            u_points = [data[i] for i in graph.get_points(u)]
            u_stats = self.__aggfunc(u_points)
            stats[u] = u_stats
        metric = lambda x, y: self.__metric(self.__lens(x), self.__lens(y))
        self.__tree = BallTree(metric, stats.values())
        return stats

    def predict(self, point):
        return self.__tree.nn_search(point)

    def test_kpi(self, test_set, kpi):
        errs = []
        for x in test_set:
            x_pred = self.predict(x)
            err = kpi(x, x_pred)
            errs.append(err)
        return errs

    def test_mape(self, test_set):
        return self.test_kpi(test_set, mape)

    def test_mae(self, test_set):
        return self.test_kpi(test_set, mae)

    def test_rmse(self, test_set):
        return self.test_kpi(test_set, lambda x, x_pred: math.sqrt(mse(x, x_pred)))


class Graph:
    "A class representing a mapper graph"

    def __init__(self):
        self.__adjaciency = {}
        self.__vertices = {}
        self.__edges = {}
        self.__labels = None

    def add_vertex(self, vertex_id, vertex):
        """Add a new vertex to the graph"""
        self.__adjaciency[vertex_id] = []
        self.__vertices[vertex_id] = vertex

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
        return self.__vertices[vertex_id].get_ids()

    def get_adjaciency(self, vertex_id):
        """Return the adjaciency list of a given vertex"""
        return self.__adjaciency[vertex_id]

    def get_edge(self, source_id, target_id):
        """Return the edge for two specified vertices"""
        return self.__edges[(source_id, target_id)]

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
            for point in self.__vertices[u_id].get_ids():
                point_label[point] = u_label
        return point_label

    def get_vertex_label(self, vertex_id):
        return self.__labels[vertex_id]
