"""Classes for storing the mapper graph"""
import math
import networkx as nx
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error as mape

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
        self.__edges = {}
        self.__tree = None

    def _build_tree(self, metric, lens):
        data = [np.array(v.get_data().get_point()) for v in self.__vertices.values()]
        pullback = lambda x, y : metric(lens(x), lens(y))
        self.__tree = BallTree(pullback, data)

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

    def get_adjaciency(self, vertex_id):
        """Return the adjaciency list of a given vertex"""
        return self.__adjaciency[vertex_id]

    def get_edge(self, source_id, target_id):
        """Return the edge for two specified vertices"""
        return self.__edges[(source_id, target_id)]

    def _predict(self, x_value):
        nn = self.__tree.nn_search(np.array(x_value))
        return nn

    def test(self, metric, lens, test_set):
        errs = []
        if not self.__tree:
            self._build_tree(metric, lens)
        for x in test_set:
            x_pred = self._predict(x)
            err = mape(x, x_pred)
            errs.append(err)
        return errs

class GraphPlot:

    def __init__(self, graph):
        """build inner nx graph"""
        self.__nx = nx.Graph()
        for u in graph.get_vertices():
            u_vert = graph.get_vertex(u)
            u_color = u_vert.get_data().get_color()
            u_size = u_vert.get_size()
            self.__nx.add_node(u, size=u_size, color=u_color)
            for v in graph.get_adjaciency(u):
                uv_weight = graph.get_edge(u, v).get_weight()
                self.__nx.add_edge(u, v, weight=uv_weight)
        self.__size_dict = nx.get_node_attributes(self.__nx, 'size')
        self.__color_dict = nx.get_node_attributes(self.__nx, 'color')
        self.__pos2d_dict = nx.spring_layout(self.__nx, dim=2)
        self.__pos3d_dict = nx.spring_layout(self.__nx, dim=3)

    def plot(self, title='Node value', frontend='plotly', width=512, height=512):
        if frontend == 'plotly':
            return self._plot2d(title, width, height)
        elif frontend == 'pyplot':
            return self._plt_display(title, width, height)
        elif frontend == '3d':
            return self._plot3d(title, width, height)

    def _plt_display(self, title, width, height):
        max_size = max(self.__size_dict.values()) if self.__size_dict else 1.0
        node_sizes = []
        for node in self.__nx.nodes():
            size = float(self.__size_dict[node]) / max_size
            node_sizes.append(600.0 * size)
        colors = list(self.__color_dict.values())
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_facecolor('#fff')
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(0)
        plt.set_cmap('viridis_r')
        nx.draw_networkx_edges(self.__nx,
            self.__pos2d_dict,
            edge_color='#111',
            alpha=0.25)
        nodes = nx.draw_networkx_nodes(self.__nx,
            self.__pos2d_dict,
            node_color=colors,
            node_size=node_sizes,
            alpha=1.0,
            edgecolors='#111')
        colorbar = fig.colorbar(nodes, orientation='horizontal', aspect=60, pad=0.0)
        colorbar.set_label(title)
        colorbar.outline.set_linewidth(0)
        return fig

    def _plot2d_nodes(self, title):
        node_x, node_y = [], []
        node_colors, node_sizes = [], []
        node_texts = []
        max_size = max(self.__size_dict.values()) if self.__size_dict else 1.0
        for node in self.__nx.nodes():
            x, y = self.__pos2d_dict[node]
            color = self.__color_dict[node]
            size = float(self.__size_dict[node]) / max_size
            node_x.append(x)
            node_y.append(y)
            node_colors.append(color)
            node_sizes.append(30.0 * math.sqrt(size))
            size_text = str(self.__size_dict[node])
            color_text = str(self.__color_dict[node])
            txt = "size: " + size_text + ", color: " + color_text
            node_texts.append(txt)
        node_trace = go.Scattergl(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='viridis',
                reversescale=True,
                color=node_colors,
                opacity=1.0,
                size=node_sizes,
                colorbar=dict(
                    thickness=10,
                    title=title,
                    xanchor='left',
                    titleside='right',
                    xpad=0
                ),
                line_width=1.0,
                line_color='#111'))
        node_trace.text = node_texts
        return node_trace

    def _plot2d_edges(self):
        edge_x, edge_y = [], []
        for edge in self.__nx.edges():
            x0, y0 = self.__pos2d_dict[edge[0]]
            x1, y1 = self.__pos2d_dict[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)
        edge_trace = go.Scattergl(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=1.0, color='rgba(1, 1, 1, 0.25)'),
            hoverinfo='none'
        )
        return edge_trace

    def _plot2d(self, title, width, height):
        edge_trace = self._plot2d_edges()
        node_trace = self._plot2d_nodes(title)
        axis = dict(showbackground=False,
                showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                title='')
        fig = go.Figure(data=[edge_trace, node_trace],
            layout=go.Layout(
                width=width, height=height,
                plot_bgcolor='#fff',
                autosize=True,
                showlegend=False,
                scene=dict(xaxis=dict(axis),
                            yaxis=dict(axis)),
                hovermode='closest',
                margin=dict(b=0,l=0,r=0,t=0),
                annotations=[dict(
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.000, 
                    y=0.000,
                    text='') ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
            )
        return fig
        
    def _plot3d_edges(self):
        edge_x, edge_y, edge_z = [], [], []
        for edge in self.__nx.edges():
            x0, y0, z0 = self.__pos3d_dict[edge[0]]
            x1, y1, z1 = self.__pos3d_dict[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None) #stop from drawing line
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None) #stop from drawing line
            edge_z.append(z0)
            edge_z.append(z1)
            edge_z.append(None) #stop from drawing line
        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            line=dict(width=1.0, color='rgba(1, 1, 1, 0.5)'),
            hoverinfo='none',
            mode='lines')
        return edge_trace

    def _plot3d_nodes(self, title):
        node_x, node_y, node_z = [], [], []
        node_colors, node_sizes = [], []
        node_texts = []
        max_size = max(self.__size_dict.values()) if self.__size_dict else 1.0
        for node in self.__nx.nodes():
            x, y, z = self.__pos3d_dict[node]
            color = self.__color_dict[node]
            size = float(self.__size_dict[node]) / max_size
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)
            node_colors.append(color)
            node_sizes.append(20.0 * math.sqrt(size))
            size_text = str(self.__size_dict[node])
            color_text = str(self.__color_dict[node])
            txt = "size: " + size_text + ", color: " + color_text
            node_texts.append(txt)
        node_trace = go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='viridis',
                reversescale=True,
                color=node_colors,
                opacity=1.0,
                size=node_sizes,
                colorbar=dict(
                    thickness=10,
                    title=title,
                    xanchor='left',
                    titleside='right',
                    xpad=0
                ),
                line_width=1.0,
                line_color='#111'))
        node_trace.text = node_texts
        return node_trace

    def _plot3d(self, title, width, height):
        edge_trace = self._plot3d_edges()
        node_trace = self._plot3d_nodes(title)
        axis = dict(showbackground=False,
                showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                title='')
        fig = go.Figure(data=[edge_trace, node_trace],
            layout=go.Layout(
                width=width, height=height,
                plot_bgcolor='#f0f2f6',
                autosize=True,
                showlegend=False,
                scene=dict(xaxis=dict(axis),
                            yaxis=dict(axis),
                            zaxis=dict(axis),
                            ),
                hovermode='closest',
                margin=dict(b=0,l=0,r=0,t=0),
                annotations=[dict(
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.000, 
                    y=0.000,
                    text='') ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
            )
        return fig
