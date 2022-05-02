"""module for storing and drawing a cover graph"""
import math

import plotly.graph_objects as go
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

BORDER_COLOR = '#111'

ATTR_SIZE = 'size'
ATTR_COLOR = 'color'
ATTR_CC = 'cc'
ATTR_IDS = 'ids'
ATTR_MIN_COLOR = 'min_color'
ATTR_MAX_COLOR = 'max_color'

FE_MATPLOTLIB = 'matplotlib'
FE_PLOTLY = 'plotly'

class CoverGraph:

    "A class representing a cover graph"
    def __init__(self, cover_arr):
        self.__graph = self._build_graph(cover_arr)
        self.__pos2d = nx.spring_layout(self.__graph)
        self._compute_connected_components(self.__graph)
        self._compute_colors(self.__graph, None, None)

    
    def colorize(self, data, colormap):
        self._compute_colors(self.__graph, data, colormap)


    def _build_graph(self, cluster_arr):
        graph = nx.Graph()
        vertices = {}
        for i, clusters in enumerate(cluster_arr):
            for c in clusters:
                if c not in vertices:
                    vertices[c] = []
                vertices[c].append(i)
        for cluster, cluster_points in vertices.items():
            graph.add_node(
                cluster, 
                ids=cluster_points,
                size=len(cluster_points))
        for p in cluster_arr:
            for s in p:
                for t in p:
                    if s != t:
                        graph.add_edge(s, t, weight=1) # TODO: compute weight correctly
        return graph


    def _compute_connected_components(self, graph):
        cc_id = 1
        vert_cc = {}
        for cc in nx.connected_components(graph):
            for node in cc:
                vert_cc[node] = cc_id
            cc_id += 1
        nx.set_node_attributes(graph, vert_cc, ATTR_CC)
    

    def _compute_colors(self, graph, data, colormap):
        nodes = graph.nodes()
        colors = {}
        if data is None or colormap is None:
            colors = {node: 0.5 for node in self.__graph.nodes()}
            graph_min_color = 0.0
            graph_max_color = 1.0
        else:
            graph_min_color = float('inf')
            graph_max_color = -float('inf')
            for node in nodes:
                node_colors = [colormap(data[i]) for i in nodes[node][ATTR_IDS]]
                min_color = min(node_colors)
                if min_color < graph_min_color:
                    graph_min_color = min_color
                max_color = max(node_colors)
                if max_color > graph_max_color:
                    graph_max_color = max_color
                colors[node] = np.nanmean(node_colors)
        graph.graph[ATTR_MIN_COLOR] = graph_min_color
        graph.graph[ATTR_MAX_COLOR] = graph_max_color
        nx.set_node_attributes(graph, colors, ATTR_COLOR)


    def plot(self, width, height, label='', frontend=FE_MATPLOTLIB):
        if frontend == FE_MATPLOTLIB:
            return self._plot_matplotlib(width, height, label)
        elif frontend == FE_PLOTLY:
            return self._plot_plotly_2d(width, height, label)


    def _plot_matplotlib(self, width, height, label):
        nodes = self.__graph.nodes()
        sizes = nx.get_node_attributes(self.__graph, ATTR_SIZE)
        max_size = max(sizes.values()) if sizes else 1.0
        colors = nx.get_node_attributes(self.__graph, ATTR_COLOR)
        px = 1/plt.rcParams['figure.dpi']
        fig, ax = plt.subplots(figsize=(width * px, height * px))
        ax.set_facecolor('#fff')
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(0)
        _ = nx.draw_networkx_edges(
            self.__graph,
            self.__pos2d,
            #edge_color='#111',
            edge_color=BORDER_COLOR,
            alpha=0.25,
            ax=ax)
        verts = nx.draw_networkx_nodes(
            self.__graph,
            self.__pos2d,
            node_color=[colors[v] for v in nodes],
            node_size=[600.0 * sizes[v] / max_size for v in nodes],
            alpha=1.0,
            edgecolors=BORDER_COLOR,
            cmap='viridis_r',
            vmin=self.__graph.graph[ATTR_MIN_COLOR],
            vmax=self.__graph.graph[ATTR_MAX_COLOR],
            ax=ax)
        colorbar = fig.colorbar(verts, orientation='vertical', aspect=60, pad=0.0)
        colorbar.set_label(label)
        colorbar.outline.set_linewidth(0)
        return fig


    def _plot_plotly_2d(self, width, height, label):
        edge_trace = self._plot_plotly_2d_edges()
        node_trace = self._plot_plotly_2d_nodes(label)
        axis = dict(
            showbackground=False,
            showline=False,
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            title=''
        )
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                width=width, height=height,
                plot_bgcolor='rgba(0, 0, 0, 0)',
                autosize=False,
                showlegend=False,
                scene=dict(
                    xaxis=dict(axis),
                    yaxis=dict(axis)
                ),
                hovermode='closest',
                margin=dict(
                    b=0,
                    l=0,
                    r=0,
                    t=0
                ),
                annotations=[
                    dict(
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        x=0.000, 
                        y=0.000,
                        text=''
                    ) 
                ],
                xaxis=dict(
                    showgrid=False,
                    zeroline=False, 
                    showticklabels=False
                ),
                yaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False
                )
            )
        )
        return fig


    def _plot_plotly_2d_edges(self):
        edge_x, edge_y = [], []
        for edge in self.__graph.edges():
            x0, y0 = self.__pos2d[edge[0]]
            x1, y1 = self.__pos2d[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)
        edge_trace = go.Scatter(
            x=edge_x, 
            y=edge_y,
            mode='lines',
            opacity=0.75,
            line=dict(
                width=1.0, 
                color='rgba(0.5, 0.5, 0.5, 0.5)'
            ),
            hoverinfo='none'
        )
        return edge_trace


    def _plot_plotly_2d_nodes(self, label):
        nodes = self.__graph.nodes()
        sizes = nx.get_node_attributes(self.__graph, ATTR_SIZE)
        max_size = max(sizes.values()) if sizes else 1.0
        colors = nx.get_node_attributes(self.__graph, ATTR_COLOR)
        ccs = nx.get_node_attributes(self.__graph, ATTR_CC)

        node_x, node_y = [], []
        node_colors, node_sizes = [], []
        node_captions = []

        for node in nodes:
            x, y = self.__pos2d[node]
            node_x.append(x)
            node_y.append(y)
            color = colors[node]
            node_colors.append(color)
            size = float(sizes[node]) / max_size
            node_sizes.append(30.0 * math.sqrt(size))
            node_label = f'size: {sizes[node]}, color: {colors[node]:.2f}, cc: {ccs[node]}'
            node_captions.append(node_label)
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers',
            hoverinfo='text',
            opacity=1.0,
            marker=dict(
                showscale=True,
                colorscale='viridis',
                reversescale=True,
                color=node_colors,
                cmax=self.__graph.graph[ATTR_MAX_COLOR],
                cmin=self.__graph.graph[ATTR_MIN_COLOR],
                opacity=0.85,
                size=node_sizes,
                colorbar=dict(
                    thickness=12,
                    title=label,
                    xanchor='left',
                    titleside='right',
                    xpad=0
                ),
                line_width=1.0,
                line_color='rgba(0.25, 0.25, 0.25, 1.0)'))
        node_trace.text = node_captions
        return node_trace
