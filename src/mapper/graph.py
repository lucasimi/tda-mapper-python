"""module for storing and drawing a cover graph"""
import math

import plotly.graph_objects as go
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

NODE_ALPHA = 0.85
EDGE_ALPHA = 0.85
EDGE_WIDTH = 0.5
EDGE_COLOR = '#777'

FONT_SIZE = 8
COLOR_FORMAT = '.2e'
TICKS_NUM = 5

ATTR_SIZE = 'size'
ATTR_COLOR = 'color'
ATTR_CC = 'cc'
ATTR_IDS = 'ids'
ATTR_MIN_COLOR = 'min_color'
ATTR_MAX_COLOR = 'max_color'

DPIS = 96

FE_MATPLOTLIB = 'matplotlib'
FE_PLOTLY = 'plotly'

class CoverGraph:

    "A class representing a cover graph"
    def __init__(self, cover_arr):
        self.__graph = self._build_graph(cover_arr)
        self.__pos2d = nx.spring_layout(self.__graph)
        self._compute_connected_components(self.__graph)
        self._compute_colors(self.__graph, None, None)


    def get_nx(self):
        return self.__graph


    def __len__(self):
        return len(self.__graph.nodes())

    
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


    def plot(self, width, height, frontend, label=''):
        if frontend == FE_MATPLOTLIB:
            return self._plot_matplotlib(width, height, label)
        elif frontend == FE_PLOTLY:
            return self._plot_plotly_2d(width, height, label)


    def _plot_matplotlib_old(self, width, height, label):
        nodes = self.__graph.nodes()
        sizes = nx.get_node_attributes(self.__graph, ATTR_SIZE)
        max_size = max(sizes.values()) if sizes else 1.0
        colors = nx.get_node_attributes(self.__graph, ATTR_COLOR)
        min_color = self.__graph.graph[ATTR_MIN_COLOR]
        max_color = self.__graph.graph[ATTR_MAX_COLOR]
        fig, ax = plt.subplots(figsize=(width / DPIS, height / DPIS), dpi=DPIS)
        ax.set_facecolor('#fff')
        #for axis in ['top','bottom','left','right']:
            #ax.spines[axis].set_linewidth(0)
        edges = nx.draw_networkx_edges(
            self.__graph,
            self.__pos2d,
            edge_color=EDGE_COLOR,
            alpha=EDGE_ALPHA,
            width=EDGE_WIDTH,
            ax=ax)
        verts = nx.draw_networkx_nodes(
            self.__graph,
            self.__pos2d,
            node_color=[colors[v] for v in nodes],
            node_size=[300 * sizes[v] / max_size for v in nodes],
            alpha=NODE_ALPHA,
            edgecolors=EDGE_COLOR,
            cmap='viridis_r',
            vmin=min_color,
            vmax=max_color,
            linewidths=EDGE_WIDTH,
            ax=ax)
        colorbar = plt.colorbar(
            verts,
            orientation='vertical',
            aspect=height/(0.025 * width),
            pad=-0.025,
            ax=ax,
            fraction=0.025
        )
        colorbar.set_label(label)
        colorbar.outline.set_linewidth(0)
        #colorbar.ax.tick_params(labelsize=FONT_SIZE)
        colorbar.ax.tick_params(size=0)
        colorbar.ax.yaxis.set_tick_params(color=EDGE_COLOR, labelcolor=EDGE_COLOR)
        #fig.tight_layout(pad=0, rect=(0.0, 0.0, 1.0, 1.0))
        fig.patch.set_alpha(0.0)
        fig.subplots_adjust(bottom=0.0, right=1.0, top=1.0, left=0.0)
        ax.patch.set_alpha(0.0)
        return fig


    def _plot_matplotlib_nodes(self, ax, width, height, label):
        nodes = self.__graph.nodes()
        sizes = nx.get_node_attributes(self.__graph, ATTR_SIZE)
        max_size = max(sizes.values()) if sizes else 1.0
        colors = nx.get_node_attributes(self.__graph, ATTR_COLOR)
        min_color = self.__graph.graph[ATTR_MIN_COLOR]
        max_color = self.__graph.graph[ATTR_MAX_COLOR]
        nodes_x = [self.__pos2d[node][0] for node in nodes]
        nodes_y = [self.__pos2d[node][1] for node in nodes]
        nodes_c = [colors[node] for node in nodes]
        nodes_s = [sizes[node] for node in nodes]
        verts = ax.scatter(
            x=nodes_x,
            y=nodes_y,
            c=nodes_c,
            s=nodes_s,
            alpha=NODE_ALPHA,
            vmin=min_color,
            vmax=max_color,
            edgecolors=EDGE_COLOR,
            linewidths=EDGE_WIDTH
        )
        colorbar = plt.colorbar(
            verts,
            orientation='vertical',
            aspect=height/(0.025 * width),
            pad=-0.025,
            ax=ax,
            fraction=0.025
        )
        colorbar.set_label(label)
        colorbar.set_alpha(NODE_ALPHA)
        colorbar.outline.set_linewidth(0)
        colorbar.outline.set_color(EDGE_COLOR)
        colorbar.ax.yaxis.set_tick_params(color=EDGE_COLOR, labelcolor=EDGE_COLOR)


    def _plot_matplotlib_edges(self, ax):
        min_color = self.__graph.graph[ATTR_MIN_COLOR]
        max_color = self.__graph.graph[ATTR_MAX_COLOR]
        colors = nx.get_node_attributes(self.__graph, ATTR_COLOR)
        edges = self.__graph.edges()
        segments = [[self.__pos2d[edge[i]] for i in [0, 1]] for edge in edges]
        cols = [0.5 * (colors[edge[0]] + colors[edge[1]]) for edge in edges]
        norm = plt.Normalize(min_color, max_color)
        lines = LineCollection(
            segments,
            cmap='viridis',
            norm=norm,
            linewidth=EDGE_WIDTH,
            alpha=EDGE_ALPHA
        )
        lines.set_array(cols)
        ax.add_collection(lines)


    def _plot_matplotlib(self, width, height, label):
        fig, ax = plt.subplots(figsize=(width / DPIS, height / DPIS), dpi=DPIS)
        ax.set_facecolor('#fff')
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(0)
        fig.patch.set_alpha(0.0)
        fig.subplots_adjust(bottom=0.0, right=1.0, top=1.0, left=0.0)
        ax.patch.set_alpha(0.0)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        self._plot_matplotlib_edges(ax)
        self._plot_matplotlib_nodes(ax, width, height, label)
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
                width=width,
                height=height,
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
            opacity=EDGE_ALPHA,
            line=dict(
                width=1.5 * EDGE_WIDTH, 
                color=EDGE_COLOR
            ),
            hoverinfo='none'
        )
        return edge_trace


    def _plot_plotly_2d_nodes(self, label):
        nodes = self.__graph.nodes()
        sizes = nx.get_node_attributes(self.__graph, ATTR_SIZE)
        max_size = max(sizes.values()) if sizes else 1.0
        colors = nx.get_node_attributes(self.__graph, ATTR_COLOR)
        min_color = self.__graph.graph[ATTR_MIN_COLOR]
        max_color = self.__graph.graph[ATTR_MAX_COLOR]
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
            node_sizes.append(25.0 * math.sqrt(sizes[node] / max_size))
            node_label = f'size: {sizes[node]:.2e}, color: {colors[node]:.2e}<br>connected component: {ccs[node]}'
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
                cmax=max_color,
                cmin=min_color,
                opacity=NODE_ALPHA,
                size=node_sizes,
                colorbar=dict(
                    #tickformat=COLOR_FORMAT,
                    outlinewidth=0,
                    borderwidth=0,
                    orientation='v',
                    thickness=0.025,
                    thicknessmode='fraction',
                    title=label,
                    xanchor='left',
                    titleside='right',
                    ypad=0,
                    #tickfont=dict(size=1.4 * FONT_SIZE),
                    #tickvals=[min_color + i * (max_color - min_color) / TICKS_NUM for i in range(TICKS_NUM + 1)],
                ),
                line_width=1.4 * EDGE_WIDTH,
                line_color=EDGE_COLOR))
        node_trace.text = node_captions
        return node_trace
