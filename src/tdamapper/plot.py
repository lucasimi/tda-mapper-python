"""A module for plotting the Mapper graph."""
import math

import numpy as np
import networkx as nx
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from tdamapper.core import (
    ATTR_SIZE,
    aggregate_graph,
)

_NODE_OUTER_WIDTH = 0.75
_NODE_OUTER_COLOR = '#777'

_EDGE_WIDTH = 0.75
_EDGE_COLOR = '#777'

_TICKS_NUM = 10


def _fmt(x, max_len=3):
    fmt = f'.{max_len}g'
    return f'{x:{fmt}}'


def _plotly_label(node_id, size, color):
    node_label_size = _fmt(size, 5)
    node_label_color = _fmt(color, 3)
    return f'color: {node_label_color}<br>node: {node_id}<br>size: {node_label_size}'


def _init_positions(g, **kwargs):
    pos = kwargs.get('pos')
    if pos is None:
        return nx.spring_layout(g, **kwargs)
    return pos


class _Plot:

    def __init__(self, dim, graph, pos, colors, cmap):
        self.dim = dim
        self.graph = graph
        self.pos = pos
        self.colors = colors
        self.cmap = cmap


class _PlotPlotly3D(_Plot):

    def plot(self, title, width, height):
        edge_trace = self._plot_edges()
        node_trace = self._plot_nodes(title)
        axis = dict(
            showline=True,
            linecolor='black',
            linewidth=1,
            mirror=True,
            visible=True,
            showticklabels=False,
            title=''
        )
        layout = go.Layout(
            width=width,
            height=height,
            plot_bgcolor='rgba(0, 0, 0, 0)',
            autosize=False,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=10, l=10, r=10, t=10),
            xaxis=axis,
            yaxis=axis,
            scene=dict(
                xaxis=dict(
                    showgrid=True,
                    visible=True,
                    backgroundcolor='rgba(0, 0, 0, 0)',
                    showaxeslabels=False,
                    showline=True,
                    linecolor='black',
                    gridcolor='rgba(230, 230, 230, 1.0)',
                    linewidth=1,
                    mirror=True,
                    showticklabels=False,
                    title='',
                ),
                yaxis=dict(
                    showgrid=True,
                    visible=True,
                    backgroundcolor='rgba(0, 0, 0, 0)',
                    showaxeslabels=False,
                    showline=True,
                    linecolor='black',
                    gridcolor='rgba(230, 230, 230, 1.0)',
                    linewidth=1,
                    mirror=True,
                    showticklabels=False,
                    title='',
                ),
                zaxis=dict(
                    showgrid=True,
                    visible=True,
                    backgroundcolor='rgba(0, 0, 0, 0)',
                    showaxeslabels=False,
                    showline=True,
                    linecolor='black',
                    gridcolor='rgba(230, 230, 230, 1.0)',
                    linewidth=1,
                    mirror=True,
                    showticklabels=False,
                    title='',
                )
            )
        )
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=layout,
        )
        return fig

    def _plot_edges(self):
        edge_x, edge_y, edge_z = [], [], []
        for edge in self.graph.edges():
            x0, y0, z0 = self.pos[edge[0]]
            x1, y1, z1 = self.pos[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)
            edge_z.append(z0)
            edge_z.append(z1)
            edge_z.append(None)
        edge_trace = go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode='lines',
            opacity=1.0,
            line_width=_EDGE_WIDTH,
            line_color=_EDGE_COLOR,
            hoverinfo='skip'
        )
        return edge_trace

    def _plot_nodes(self, title):
        nodes = self.graph.nodes()
        node_x, node_y, node_z = [], [], []
        node_captions = []
        for node in nodes:
            x, y, z = self.pos[node]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)
            size = nodes[node][ATTR_SIZE]
            node_label = _plotly_label(node, size, self.colors[node])
            node_captions.append(node_label)
        node_trace = go.Scatter3d(
            x=node_x,
            y=node_y,
            z=node_z,
            mode='markers',
            hoverinfo='text',
            opacity=1.0,
            marker=self._node_marker(title),
            text=node_captions,
        )
        return node_trace

    def _node_marker(self, title):
        nodes = self.graph.nodes()
        colors = [self.colors[node] for node in nodes]
        min_color = min(self.colors.values())
        max_color = max(self.colors.values())
        sizes = nx.get_node_attributes(self.graph, ATTR_SIZE)
        max_size = max(sizes.values()) if sizes else 1.0
        node_sizes = [25.0 * math.sqrt(sizes[node] / max_size) for node in nodes]
        return go.scatter3d.Marker(
            showscale=True,
            colorscale=self.cmap,
            reversescale=False,
            color=colors,
            cmax=max_color,
            cmin=min_color,
            opacity=1.0,
            size=node_sizes,
            colorbar=self._colorbar(title),
            line_width=_NODE_OUTER_WIDTH,
            line_color=_NODE_OUTER_COLOR,
            line_colorscale=self.cmap,
        )

    def _colorbar(self, title):
        return go.scatter3d.marker.ColorBar(
            showticklabels=True,
            outlinewidth=1,
            borderwidth=0,
            orientation='v',
            thickness=0.025,
            thicknessmode='fraction',
            title=title,
            xanchor='left',
            titleside='right',
            ypad=0,
            xpad=0,
            tickwidth=1,
            tickformat='.2g',
            nticks=_TICKS_NUM,
            tickmode='auto',
        )


class _PlotPlotly2D(_Plot):

    def plot(self, title, width, height):
        edge_trace = self._plot_edges()
        node_trace = self._plot_nodes(title)
        axis = dict(
            showline=True,
            linecolor='black',
            linewidth=1,
            mirror=True,
            visible=True,
            showticklabels=False,
        )
        layout = go.Layout(
            width=width,
            height=height,
            plot_bgcolor='rgba(0, 0, 0, 0)',
            autosize=False,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=10, l=10, r=10, t=10),
            xaxis=axis,
            yaxis=axis,
        )
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=layout
        )
        return fig

    def _plot_edges(self):
        edge_x, edge_y = [], []
        for edge in self.graph.edges():
            x0, y0 = self.pos[edge[0]]
            x1, y1 = self.pos[edge[1]]
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
            opacity=1.0,
            line=dict(
                width=_EDGE_WIDTH,
                color=_EDGE_COLOR
            ),
            hoverinfo='skip'
        )
        return edge_trace

    def _plot_nodes(self, title):
        nodes = self.graph.nodes()
        node_x, node_y = [], []
        node_captions = []
        for node in nodes:
            x, y = self.pos[node]
            node_x.append(x)
            node_y.append(y)
            size = nodes[node][ATTR_SIZE]
            node_label = _plotly_label(node, size, self.colors[node])
            node_captions.append(node_label)
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers',
            hoverinfo='text',
            opacity=1.0,
            marker=self._node_marker(title),
            text=node_captions,
        )
        return node_trace

    def _node_marker(self, title):
        nodes = self.graph.nodes()
        colors = [self.colors[node] for node in nodes]
        min_color = min(self.colors.values())
        max_color = max(self.colors.values())
        sizes = nx.get_node_attributes(self.graph, ATTR_SIZE)
        max_size = max(sizes.values()) if sizes else 1.0
        node_sizes = [25.0 * math.sqrt(sizes[node] / max_size) for node in nodes]
        return go.scatter.Marker(
            showscale=True,
            colorscale=self.cmap,
            reversescale=False,
            color=colors,
            cmax=max_color,
            cmin=min_color,
            opacity=1.0,
            size=node_sizes,
            colorbar=self._colorbar(title),
            line_width=_NODE_OUTER_WIDTH,
            line_color=_NODE_OUTER_COLOR,
        )

    def _colorbar(self, title):
        return go.scatter.marker.ColorBar(
            showticklabels=True,
            outlinewidth=1,
            borderwidth=0,
            orientation='v',
            thickness=0.025,
            thicknessmode='fraction',
            title=title,
            xanchor='left',
            titleside='right',
            ypad=0,
            xpad=0,
            tickwidth=1,
            tickformat='.2g',
            nticks=_TICKS_NUM,
            tickmode='auto',
        )


class _PlotPlotlyGL2D(_Plot):

    def plot(self, title, width, height):
        edge_trace = self._plot_edges()
        node_trace = self._plot_nodes(title)
        axis = dict(
            showline=True,
            linecolor='black',
            linewidth=1,
            mirror=True,
            visible=True,
            showticklabels=False,
        )
        layout = go.Layout(
            width=width,
            height=height,
            plot_bgcolor='rgba(0, 0, 0, 0)',
            autosize=False,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=10, l=10, r=10, t=10),
            xaxis=axis,
            yaxis=axis,
        )
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=layout
        )
        return fig

    def _plot_edges(self):
        edge_x, edge_y = [], []
        for edge in self.graph.edges():
            x0, y0 = self.pos[edge[0]]
            x1, y1 = self.pos[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)
        edge_trace = go.Scattergl(
            x=edge_x,
            y=edge_y,
            mode='lines',
            opacity=1.0,
            line=dict(
                width=_EDGE_WIDTH,
                color=_EDGE_COLOR
            ),
            hoverinfo='skip'
        )
        return edge_trace

    def _plot_nodes(self, title):
        nodes = self.graph.nodes()
        node_x, node_y = [], []
        node_captions = []
        for node in nodes:
            x, y = self.pos[node]
            node_x.append(x)
            node_y.append(y)
            size = nodes[node][ATTR_SIZE]
            node_label = _plotly_label(node, size, self.colors[node])
            node_captions.append(node_label)
        node_trace = go.Scattergl(
            x=node_x,
            y=node_y,
            mode='markers',
            hoverinfo='text',
            opacity=1.0,
            marker=self._node_marker(title),
            text=node_captions,
        )
        return node_trace

    def _node_marker(self, title):
        nodes = self.graph.nodes()
        colors = [self.colors[node] for node in nodes]
        min_color = min(self.colors.values())
        max_color = max(self.colors.values())
        sizes = nx.get_node_attributes(self.graph, ATTR_SIZE)
        max_size = max(sizes.values()) if sizes else 1.0
        node_sizes = [25.0 * math.sqrt(sizes[node] / max_size) for node in nodes]
        return go.scattergl.Marker(
            showscale=True,
            colorscale=self.cmap,
            reversescale=False,
            color=colors,
            cmax=max_color,
            cmin=min_color,
            opacity=1.0,
            size=node_sizes,
            colorbar=self._colorbar(title),
            line_width=_NODE_OUTER_WIDTH,
            line_color=_NODE_OUTER_COLOR,
        )

    def _colorbar(self, title):
        return go.scattergl.marker.ColorBar(
            showticklabels=True,
            outlinewidth=1,
            borderwidth=0,
            orientation='v',
            thickness=0.025,
            thicknessmode='fraction',
            title=title,
            xanchor='left',
            titleside='right',
            ypad=0,
            xpad=0,
            tickwidth=1,
            tickformat='.2g',
            nticks=_TICKS_NUM,
            tickmode='auto',
        )


class _PlotMatplotlib(_Plot):
    def plot(self, title, width, height):
        px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
        fig, ax = plt.subplots(figsize=(width * px, height * px))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        self._plot_edges(ax)
        self._plot_nodes(title, ax)
        return fig, ax

    def _plot_nodes(self, title, ax):
        nodes = self.graph.nodes()
        sizes = nx.get_node_attributes(self.graph, ATTR_SIZE)
        max_size = max(sizes.values())
        min_color = min(self.colors.values())
        max_color = max(self.colors.values())
        nodes_x = [self.pos[node][0] for node in nodes]
        nodes_y = [self.pos[node][1] for node in nodes]
        nodes_c = [self.colors[node] for node in nodes]
        nodes_s = [250.0 * math.sqrt(sizes[node] / max_size) for node in nodes]
        verts = ax.scatter(
            x=nodes_x,
            y=nodes_y,
            c=nodes_c,
            s=nodes_s,
            cmap=self.cmap,
            alpha=1.0,
            vmin=min_color,
            vmax=max_color,
            edgecolors=_NODE_OUTER_COLOR,
            linewidths=_NODE_OUTER_WIDTH
        )
        colorbar = plt.colorbar(
            verts,
            orientation='vertical',
            location='right',
            aspect=40,
            pad=0.02,
            ax=ax,
            format="%.2g"
        )
        colorbar.set_label(title, color=_NODE_OUTER_COLOR)
        colorbar.set_alpha(1.0)
        colorbar.outline.set_color(_NODE_OUTER_COLOR)
        colorbar.ax.yaxis.set_tick_params(
            color=_NODE_OUTER_COLOR,
            labelcolor=_NODE_OUTER_COLOR
        )
        colorbar.ax.tick_params(labelsize=8)
        colorbar.ax.locator_params(nbins=10)

    def _plot_edges(self, ax):
        edges = self.graph.edges()
        segments = [(self.pos[edge[0]], self.pos[edge[1]]) for edge in edges]
        lines = LineCollection(
            segments,
            color=_EDGE_COLOR,
            linewidth=_EDGE_WIDTH,
            alpha=1.0,
            zorder=-1,
            antialiased=True
        )
        ax.add_collection(lines)


class MapperPlot:
    """
    Creates a plot for the Mapper graph, and turn it into a displayable figure.

    :param X: A dataset.
    :type X: `numpy.ndarray` or list-like.
    :param graph: The Mapper graph.
    :type graph: `networkx.Graph`
    :param colors: A dataset of values to plot as nodes color.
    :type colors: `numpy.ndarray` or list-like.
    :param agg: Aggregation function that computes nodes color.
    :type agg: Callable on the values of colors.
    :param cmap: A colormap, to convert values into colors.
    :type cmap: `str`
    :param kwargs: Additional arguments to `networkx.spring_layout`.
    :type: `dict`
    """

    def __init__(
            self, X, graph,
            colors=None,
            agg=np.nanmean,
            cmap='jet',
            **kwargs
    ):
        self.__X = X
        self.__graph = graph
        self.__cmap = cmap
        item_colors = [np.nanmean(x) for x in self.__X] if colors is None else colors
        self.__colors = aggregate_graph(item_colors, self.__graph, agg)
        self.__dim = kwargs.get('dim', 2)
        self.__kwargs = {}
        self.__kwargs.update(kwargs)
        self.__kwargs['dim'] = self.__dim
        self.__pos = _init_positions(self.__graph, **kwargs)
        self.__kwargs['pos'] = self.__pos

    def with_colors(self, colors, agg=np.nanmean, cmap='jet'):
        return MapperPlot(self.__X, self.__graph,
                          colors=colors, agg=agg, cmap=cmap, **self.__kwargs)

    def plot(self, title=None, width=512, height=512, backend='plotly', **kwargs):
        """
        Turns the plot object into a displayable figure.

        :param title: Title displayed on the figure.
        :type title: `str`
        :param width: The width of the figure expressed in pixels.
        :type width: `int`
        :param height: The height of the figure expressed in pixels.
        :type height: `int`
        :param backend: The type of backend used for drawing.
        :type backend: 'matplotlib' or 'plotly' or 'plotly_gl'.
        :param kwargs: Additional arguments to supply.
        :type kwargs: `dict`
        """
        if self.__dim == 2:
            if backend == 'matplotlib':
                return _PlotMatplotlib(
                    dim=self.__dim,
                    graph=self.__graph,
                    pos=self.__pos,
                    colors=self.__colors,
                    cmap=self.__cmap).plot(
                        title=title,
                        width=width,
                        height=height)
            elif backend == 'plotly':
                return _PlotPlotly2D(
                    dim=self.__dim,
                    graph=self.__graph,
                    pos=self.__pos,
                    colors=self.__colors,
                    cmap=self.__cmap).plot(
                        title=title,
                        width=width,
                        height=height)
            elif backend == 'plotly_gl':
                return _PlotPlotlyGL2D(
                    dim=self.__dim,
                    graph=self.__graph,
                    pos=self.__pos,
                    colors=self.__colors,
                    cmap=self.__cmap).plot(
                    title=title,
                    width=width,
                    height=height)
        elif backend == 'plotly':
            return _PlotPlotly3D(
                dim=self.__dim,
                graph=self.__graph,
                pos=self.__pos,
                colors=self.__colors,
                cmap=self.__cmap).plot(
                    title=title,
                    width=width,
                    height=height)
