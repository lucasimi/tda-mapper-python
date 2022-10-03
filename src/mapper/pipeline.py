"""A module for the exact mapper algorithm"""
import logging
import time
import networkx as nx
from .cover import CoverGraph


_logger = logging.getLogger()
_logger.setLevel(logging.INFO)
_fhandler = logging.FileHandler(filename='mapper-tda.log', mode='a')
_logger.addHandler(_fhandler)
_formatter = logging.Formatter('%(asctime)s - %(name)s - %(lineno)s - %(levelname)s - %(message)s')
_fhandler.setFormatter(_formatter)


ATTR_IDS = 'ids'
ATTR_SIZE = 'size'


def _log(msg):
    logging.info(msg)


def compute_connected_components(graph):
    cc_id = 1
    vert_cc = {}
    for cc in nx.connected_components(graph):
        for node in cc:
            vert_cc[node] = cc_id
        cc_id += 1
    return vert_cc


def _build_graph(multilabels):
    graph = nx.Graph()
    clusters = set()
    sizes = {}
    point_ids = {}
    for point_id, point_labels in enumerate(multilabels):
        for label in point_labels:
            if label not in clusters:
                clusters.add(label)
                graph.add_node(label)
                sizes[label] = 0
                point_ids[label] = []
            sizes[label] += 1
            point_ids[label].append(point_id)
    nx.set_node_attributes(graph, sizes, ATTR_SIZE)
    nx.set_node_attributes(graph, point_ids, ATTR_IDS)
    edges = set()
    for labels in multilabels:
        for s in labels:
            for t in labels:
                if s != t and (s, t) not in edges:
                    graph.add_edge(s, t, weight=1) # TODO: compute weight correctly
                    edges.add((s, t))
                    graph.add_edge(t, s, weight=1) # TODO: compute weight correctly
                    edges.add((t, s))
    return graph


class MapperPipeline:

    def __init__(self, search_algo=None, clustering_algo=None):
        self.__cover_graph = CoverGraph(
            search_algo=search_algo,
            clustering_algo=clustering_algo
        )


    def fit(self, X):
        t0 = time.time()
        labels = self.__cover_graph.fit_predict(X)
        t1 = time.time()
        _log(f'Mapper labels computed in {t1 - t0}s')
        graph = _build_graph(labels)
        t2 = time.time()
        _log(f'Mapper graph computed in {t2 - t1}s')
        return graph
