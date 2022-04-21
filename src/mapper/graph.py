"""Classes for storing the mapper graph"""

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


class Graph:
    "A class representing a mapper graph"

    def __init__(self):
        self.__adjaciency = {}
        self.__vertices = {}
        self.__edges = {}
        self.__vert_cc = None  # connected components indexed by vertices
        self.__cc_vert = None  # vertices indexed by connected components

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

    def get_vertex_ids(self, vertex_id):
        return self.__vertices[vertex_id].get_ids()

    def get_adjaciency(self, vertex_id):
        """Return the adjaciency list of a given vertex"""
        return self.__adjaciency[vertex_id]

    def get_edge(self, source_id, target_id):
        """Return the edge for two specified vertices"""
        return self.__edges[(source_id, target_id)]

    def _compute_ccs(self):
        vert_ccs = {u_id: None for u_id in self.__vertices}
        cc_count = 0
        for u_id in self.__vertices:
            if not vert_ccs[u_id]:
                cc_count += 1
                self._set_cc(vert_ccs, u_id, cc_count)
        self.__vert_cc = vert_ccs
        ccs = {}
        for u, u_cc in vert_ccs.items():
            if u_cc not in ccs:
                ccs[u_cc] = []
            ccs[u_cc].append(u)
        self.__cc_vert = ccs

    def _set_cc(self, vert_ccs, u_id, cc_label):
        if not vert_ccs[u_id]:
            vert_ccs[u_id] = cc_label
            for v_id in self.__adjaciency[u_id]:
                self._set_cc(vert_ccs, v_id, cc_label)

    def get_cc_vertices(self, cc):
        return self.__cc_vert[cc]

    def get_ids_ccs(self):
        ids_cc = {}
        for u_id, u_vert in self.__vertices.items():
            cc_id = self.get_vertex_cc(u_id)
            for point_id in u_vert.get_ids():
                ids_cc[point_id] = cc_id
        return ids_cc

    def get_cc_ids(self, cc):
        ids = set()
        for u in self.get_cc_vertices(cc):
            ids.update(self.get_vertex_ids(u))
        return ids

    def get_vertex_cc(self, vertex_id):
        return self.__vert_cc[vertex_id]

    def get_ccs(self):
        return self.__cc_vert.keys()
