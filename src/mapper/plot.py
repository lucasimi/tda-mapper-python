"""module for storing and drawing a cover graph"""
import math
import numpy as np

import plotly.graph_objects as go
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import mapper.pipeline

NODE_ALPHA = 0.85
EDGE_ALPHA = 0.85
EDGE_WIDTH = 0.5
EDGE_COLOR = '#777'

FONT_SIZE = 8
COLOR_FORMAT = '.2e'
TICKS_NUM = 5

DPIS = 96

FE_MATPLOTLIB = 'matplotlib'
FE_PLOTLY = 'plotly'


class MapperPlot:

    "A class representing a cover graph"
    def __init__(self, graph):
        self.__graph = graph
        self.__pos2d = nx.spring_layout(self.__graph)
        self.__colors = {x:0.5 for x in self.__graph.nodes()}

    def colorize(self, data, colormap=lambda x: x, agg=np.nanmean):
        colors = {}
        nodes = self.__graph.nodes()
        for node_id in nodes:
            node_data = [data[i] for i in nodes[node_id][mapper.pipeline.ATTR_IDS]]
            node_colors = [colormap(x) for x in node_data]
            agg_color = agg(node_colors)
            colors[node_id] = agg_color
        self.__colors = colors

    def plot(self, frontend, width, height, title=''):
        if frontend == FE_MATPLOTLIB:
            return self._plot_matplotlib(width, height, title)
        elif frontend == FE_PLOTLY:
            return self._plot_plotly_2d(width, height, title)
        else:
            raise Exception(f'unexpected argument {frontend} for frontend')

    def _plot_matplotlib_nodes(self, ax, width, height, title):
        nodes = self.__graph.nodes()
        sizes = nx.get_node_attributes(self.__graph, mapper.pipeline.ATTR_SIZE)
        max_size = max(sizes.values())
        min_color = min(self.__colors.values())
        max_color = max(self.__colors.values())
        nodes_x = [self.__pos2d[node][0] for node in nodes]
        nodes_y = [self.__pos2d[node][1] for node in nodes]
        nodes_c = [self.__colors[node] for node in nodes]
        nodes_s = [250.0 * math.sqrt(sizes[node]/max_size) for node in nodes]
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
            pad=0.0,
            ax=ax,
            fraction=0.025
        )
        colorbar.set_label(title, color=EDGE_COLOR)
        colorbar.set_alpha(NODE_ALPHA)
        colorbar.outline.set_linewidth(0)
        colorbar.outline.set_color(EDGE_COLOR)
        colorbar.ax.yaxis.set_tick_params(color=EDGE_COLOR, labelcolor=EDGE_COLOR)

    def _plot_matplotlib_edges(self, ax):
        min_color = min(self.__colors.values())
        max_color = max(self.__colors.values())
        edges = self.__graph.edges()
        segments = [(self.__pos2d[edge[0]], self.__pos2d[edge[1]]) for edge in edges]
        cols = [0.5 * (self.__colors[edge[0]] + self.__colors[edge[1]]) for edge in edges]
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

    def _plot_matplotlib(self, width, height, title):
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
        self._plot_matplotlib_nodes(ax, width, height, title)
        return fig

    def _plot_plotly_2d(self, width, height, title):
        edge_trace = self._plot_plotly_2d_edges()
        node_trace = self._plot_plotly_2d_nodes(title)
        axis = dict(
            showbackground=False,
            showline=False,
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            title=title
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

    def _plot_plotly_2d_nodes(self, title):
        nodes = self.__graph.nodes()
        sizes = nx.get_node_attributes(self.__graph, mapper.pipeline.ATTR_SIZE)
        max_size = max(sizes.values()) if sizes else 1.0
        min_color = min(self.__colors.values())
        max_color = max(self.__colors.values())

        node_x, node_y = [], []
        node_colors, node_sizes = [], []
        node_captions = []

        for node in nodes:
            x, y = self.__pos2d[node]
            node_x.append(x)
            node_y.append(y)
            color = self.__colors[node]
            node_colors.append(color)
            node_sizes.append(25.0 * math.sqrt(sizes[node] / max_size))
            node_label = f'size: {sizes[node]:.2e}, color: {self.__colors[node]:.2e}'
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
                reversescale=False,
                color=node_colors,
                cmax=max_color,
                cmin=min_color,
                opacity=NODE_ALPHA,
                size=node_sizes,
                colorbar=dict(
                    outlinewidth=0,
                    borderwidth=0,
                    orientation='v',
                    thickness=0.025,
                    thicknessmode='fraction',
                    title=title,
                    xanchor='left',
                    titleside='right',
                    ypad=0,
                ),
                line_width=1.4 * EDGE_WIDTH,
                line_color=EDGE_COLOR))
        node_trace.text = node_captions
        return node_trace
