"""A module for the exact mapper algorithm"""
import numpy as np

from .graph import MeanStats, Vertex, Edge, Graph


def _point_labels(labels):
    point_labels_dict = {}
    for ball_id, labels in enumerate(labels):
        for point_id, label in labels:
            if point_id not in point_labels_dict:
                point_labels_dict[point_id] = set()
            point_labels_dict[point_id].add((ball_id, label))
    return point_labels_dict


def _build_vertices(data, labels, mapper_graph, lens, colormap):
    vertex_ids = {}
    vertex_count = 0
    for ball_id, ls in enumerate(labels):
        clusters_dict = {}
        for point_id, label in ls:
            if label not in clusters_dict:
                clusters_dict[label] = []
            clusters_dict[label].append(point_id)
        for label, cluster in clusters_dict.items():
            points = [data[i] for i in cluster]
            point = np.nanmean(points)
            values = [lens(x) for x in points]
            value = np.nanmean([x for x in values])
            color = np.nanmean([colormap(x) for x in values])
            vertex = Vertex(MeanStats(point, value, color), len(cluster))
            vertex_ids[(ball_id, label)] = vertex_count
            mapper_graph.add_vertex(vertex_count, vertex)
            vertex_count += 1
    return vertex_ids


def _build_edges(point_labels, vertex_ids, mapper_graph):
    for clusters in point_labels.values():
        for cluster_s in clusters:
            vert_s = vertex_ids[cluster_s]
            for cluster_t in clusters:
                vert_t = vertex_ids[cluster_t]
                if cluster_s != cluster_t:
                    edge = Edge(1, 1, 0) #compute this correctly
                    mapper_graph.add_edge(vert_s, vert_t, edge)


def _compute_mapper(data, labels, lens, colormap):
    """Build a mapper graph from data"""
    mapper_graph = Graph()
    vert_ids = _build_vertices(data, labels, mapper_graph, lens, colormap)
    point_labels = _point_labels(labels)
    _build_edges(point_labels, vert_ids, mapper_graph)
    return mapper_graph


class Mapper:

    def __init__(self, cover_algo, clustering_algo):
        self.__cover_algo = cover_algo
        self.__clustering_algo = clustering_algo

    def run(self, data, lens, metric, colormap):
        pb_metric = lambda x, y: metric(lens(x), lens(y))
        atlas_ids = self.__cover_algo.cover(data, pb_metric)
        labels = self.__clustering_algo.fit(data, atlas_ids)
        return _compute_mapper(data, labels, lens, colormap)

