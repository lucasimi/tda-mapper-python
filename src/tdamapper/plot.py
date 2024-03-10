"""
This module provides functionalities to visualize the Mapper graph.
"""

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

_NODE_OPACITY = 1.0

_EDGE_WIDTH = 0.75

_EDGE_OPACITY = 1.0

_EDGE_COLOR = '#777'

_TICKS_NUM = 10


def _nodes_pos(graph, dim, seed, iterations):
    return nx.spring_layout(
        graph,
        dim=dim,
        seed=seed,
        iterations=iterations)


def _nodes_array(graph, dim, pos):
    return tuple([pos[n][i] for n in graph.nodes()] for i in range(dim))


def _edges_array(graph, dim, pos):
    edges_arr = tuple([] for i in range(dim))
    for edge in graph.edges():
        pos0, pos1 = pos[edge[0]], pos[edge[1]]
        for i in range(dim):
            edges_arr[i].append(pos0[i])
            edges_arr[i].append(pos1[i])
            edges_arr[i].append(None)
    return edges_arr


def _fmt(x, max_len=3):
    fmt = f'.{max_len}g'
    return f'{x:{fmt}}'


def _plotly_colorbar(dim, title=None):
    cbar = dict(
        showticklabels=True,
        outlinewidth=1,
        borderwidth=0,
        orientation='v',
        thickness=0.025,
        thicknessmode='fraction',
        xanchor='left',
        titleside='right',
        ypad=0,
        xpad=0,
        tickwidth=1,
        tickformat='.2g',
        nticks=_TICKS_NUM,
        tickmode='auto')
    if title is not None:
        cbar['title'] = title
    if dim == 3:
        return go.scatter3d.marker.ColorBar(cbar)
    elif dim == 2:
        return go.scatter.marker.ColorBar(cbar)


def _plotly_nodes_text(graph, colors=None):
    attr_size = nx.get_node_attributes(graph, ATTR_SIZE)
    if colors is None:
        def _lbl(n):
            size = _fmt(attr_size[n], 5)
            return f'node: {n}<br>size: {size}'
    else:
        def _lbl(n):
            col = _fmt(colors[n], 3)
            size = _fmt(attr_size[n], 5)
            return f'color: {col}<br>node: {n}<br>size: {size}'
    return [_lbl(n) for n in graph.nodes()]


def _plotly_nodes_trace(graph, node_arr, dim):
    attr_size = nx.get_node_attributes(graph, ATTR_SIZE)
    max_size = max(attr_size.values()) if attr_size else 1.0
    scatter_text = _plotly_nodes_text(graph)
    marker_size = [25.0 * math.sqrt(attr_size[n] / max_size) for n in graph.nodes()]
    scatter = dict(
        name='nodes_trace',
        x=node_arr[0],
        y=node_arr[1],
        mode='markers',
        hoverinfo='text',
        opacity=_NODE_OPACITY,
        text=scatter_text,
        marker=dict(
            showscale=True,
            reversescale=False,
            size=marker_size,
            opacity=_NODE_OPACITY,
            line_width=_NODE_OUTER_WIDTH,
            line_color=_NODE_OUTER_COLOR))
    if dim == 3:
        scatter.update(dict(
            z=node_arr[2]))
        return go.Scatter3d(scatter)
    elif dim == 2:
        return go.Scatter(scatter)


def _plotly_edges_trace(edge_arr, dim):
    scatter = dict(
        name='edges_trace',
        x=edge_arr[0],
        y=edge_arr[1],
        mode='lines',
        opacity=_EDGE_OPACITY,
        line_width=_EDGE_WIDTH,
        line_color=_EDGE_COLOR,
        hoverinfo='skip')
    if dim == 3:
        scatter.update(dict(
            z=edge_arr[2]))
        return go.Scatter3d(scatter)
    elif dim == 2:
        return go.Scatter(scatter)


def _plotly_layout():
    axis = dict(
        showline=True,
        linecolor='black',
        linewidth=1,
        mirror=True,
        visible=True,
        showticklabels=False,
        title='')
    scene_axis = dict(
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
        title='')
    return go.Layout(
        uirevision='constant',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        autosize=False,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=10, l=10, r=10, t=10),
        xaxis=axis,
        yaxis=axis,
        scene=dict(
            xaxis=scene_axis,
            yaxis=scene_axis,
            zaxis=scene_axis))


def _plotly_mapper_fig(graph, dim, seed, iterations):
    pos = _nodes_pos(graph, dim, seed, iterations)
    node_arr = _nodes_array(graph, dim, pos)
    edge_arr = _edges_array(graph, dim, pos)
    _edges_tr = _plotly_edges_trace(edge_arr, dim)
    _nodes_tr = _plotly_nodes_trace(graph, node_arr, dim)
    _layout = _plotly_layout()
    return go.Figure(
        data=[_edges_tr, _nodes_tr],
        layout=_layout)


class MapperLayoutInteractive:
    """
    Class for generating and visualizing the Mapper graph.

    This class creates a metric embedding of the Mapper graph in 2D or 3D and
    converts it into a Plotly figure suitable for interactive display.

    :param graph: The precomputed Mapper graph to be embedded. This can be
        obtained by calling :func:`tdamapper.core.mapper_graph` or
        :func:`tdamapper.core.MapperAlgorithm.fit_transform`.
    :type graph: :class:`networkx.Graph`, required
    :param dim: The dimension of the graph embedding (2 or 3).
    :type dim: int
    :param seed: The random seed used to construct the graph embedding.
    :type seed: int, optional (default: 42)
    :param iterations: The number of iterations used to construct the graph embedding.
    :type iterations: int, optional (default: 50)
    :param colors: An array of values that determine the color of each node in
        the graph, useful for highlighting different features of the data.
    :type colors: array-like of shape (n,) or list-like of size n
    :param agg: A function used to aggregate the `colors` data when multiple
        points are mapped to a single node. The final color of each node is
        obtained by mapping the aggregated value with the colormap `cmap`.
    :type agg: Callable, optional
    :param title: The title to be displayed alongside the figure.
    :type title: str, optional
    :param width: The desired width of the figure in pixels.
    :type width: int, optional (default: 512)
    :param height: The desired height of the figure in pixels.
    :type height: int, optional (default: 512)
    :param cmap: The name of a colormap used to map `color` data values,
        aggregated by `agg`, to actual RGBA colors.
    :type cmap: str, optional
    """

    def __init__(self, graph, dim,
                 seed=42,
                 iterations=50,
                 colors=None,
                 agg=np.nanmean,
                 title=None,
                 width=512,
                 height=512,
                 cmap='jet'):
        self.__graph = graph
        self.__dim = dim
        self.seed = seed
        self.iterations = iterations
        self.colors = colors
        self.agg = agg
        self.title = title
        self.width = width
        self.height = height
        self.cmap = cmap
        self.fig = _plotly_mapper_fig(
            self.__graph,
            self.__dim,
            self.seed,
            self.iterations)
        self._update_traces_col()
        self._update_layout()
        self._update_traces_cmap()
        self._update_traces_title()

    def _update_traces_pos(self):
        pos = _nodes_pos(self.__graph, self.__dim, self.seed, self.iterations)
        node_arr = _nodes_array(self.__graph, self.__dim, pos)
        edge_arr = _edges_array(self.__graph, self.__dim, pos)
        if self.__dim == 3:
            self.fig.update_traces(
                patch=dict(
                    x=node_arr[0],
                    y=node_arr[1],
                    z=node_arr[3]),
                selector=dict(
                    name='nodes_trace'))
            self.fig.update_traces(
                patch=dict(
                    x=edge_arr[0],
                    y=edge_arr[1],
                    z=edge_arr[3]),
                selector=dict(
                    name='edges_trace'))
        elif self.__dim == 2:
            self.fig.update_traces(
                patch=dict(
                    x=node_arr[0],
                    y=node_arr[1]),
                selector=dict(
                    name='nodes_trace'))
            self.fig.update_traces(
                patch=dict(
                    x=edge_arr[0],
                    y=edge_arr[1]),
                selector=dict(
                    name='edges_trace'))

    def _update_traces_col(self):
        if (self.colors is not None) and (self.agg is not None):
            colors_agg = aggregate_graph(self.colors, self.__graph, self.agg)
            colors_list = [colors_agg[n] for n in self.__graph.nodes()]
            self.fig.update_traces(
                patch=dict(
                    marker_color=colors_list,
                    marker_cmax=max(colors_list),
                    marker_cmin=min(colors_list),
                    text=_plotly_nodes_text(self.__graph, colors_agg)),
                selector=dict(
                    name='nodes_trace'))

    def _update_traces_cmap(self):
        self.fig.update_traces(
            patch=dict(
                marker_colorscale=self.cmap,
                marker_line_colorscale=self.cmap),
            selector=dict(
                name='nodes_trace'))

    def _update_traces_title(self):
        self.fig.update_traces(
            patch=dict(
                marker_colorbar=_plotly_colorbar(self.__dim, self.title)),
            selector=dict(
                name='nodes_trace'))

    def _update_layout(self):
        self.fig.update_layout(
            width=self.width,
            height=self.height)

    def update(self,
               seed=None,
               iterations=None,
               colors=None,
               agg=None,
               title=None,
               width=None,
               height=None,
               cmap=None):
        """
        Update the figure.

        This method modifies the figure returned by the `plot` function. After
        calling this method, the figure will be updated according to the supplied
        parameters.

        :param seed: The random seed used to construct the graph embedding.
        :type seed: int, optional
        :param iterations: The number of iterations used to construct the graph embedding.
        :type iterations: int, optional
        :param colors: An array of values that determine the color of each node in
            the graph, useful for highlighting different features of the data.
        :type colors: array-like of shape (n,) or list-like of size n
        :param agg: A function used to aggregate the `colors` data when multiple
            points are mapped to a single node. The final color of each node is
            obtained by mapping the aggregated value with the colormap `cmap`.
        :type agg: Callable, optional
        :param title: The title to be displayed alongside the figure.
        :type title: str, optional
        :param width: The desired width of the figure in pixels.
        :type width: int, optional
        :param height: The desired height of the figure in pixels.
        :type height: int, optional
        :param cmap: The name of a colormap used to map `color` data values,
            aggregated by `agg`, to actual RGBA colors.
        :type cmap: str, optional
        """
        _update_pos = False
        if seed is not None:
            self.seed = seed
            _update_pos = True
        if iterations is not None:
            self.iterations = iterations
            _update_pos = True
        if _update_pos:
            self._update_traces_pos()
        _update_col = False
        if agg is not None:
            self.agg = agg
        if (colors is not None) and (agg is not None):
            self.colors = colors
            self.agg = agg
            _update_col = True
        if (colors is not None) and (self.agg is not None):
            self.colors = colors
            _update_col = True
        if (self.colors is not None) and (agg is not None):
            self.agg = agg
            _update_col = True
        if _update_col:
            self._update_traces_col()
        if cmap is not None:
            self.cmap = cmap
            self._update_traces_cmap()
        if title is not None:
            self.title = title
            self._update_traces_title()
        _update_layout = False
        if (width is not None) and (height is not None):
            self.width = width
            self.height = height
            _update_layout = True
        if (width is not None) and (self.height is not None):
            self.width = width
            _update_layout = True
        if height is not None:
            self.height = height
            _update_layout = True
        if _update_layout:
            self._update_layout()

    def plot(self):
        """
        Plot the Mapper graph.
        
        :return: An interactive Plotly figure that can be displayed on screen and notebooks.
            For 3D embeddings, the figure requires a WebGL context to be shown.
        :rtype: :class:`plotly.graph_objects.Figure`
        """
        return self.fig


class MapperLayoutStatic:
    """
    Class for generating and visualizing the Mapper graph.

    This class creates a metric embedding of the Mapper graph in 2D and
    converts it into a matplotlib figure suitable for static display.

    :param graph: The precomputed Mapper graph to be embedded. This can be
        obtained by calling :func:`tdamapper.core.mapper_graph` or
        :func:`tdamapper.core.MapperAlgorithm.fit_transform`.
    :type graph: :class:`networkx.Graph`, required
    :param dim: The dimension of the graph embedding (only 2 is supported, for compatibility).
    :type dim: int
    :param seed: The random seed used to construct the graph embedding.
    :type seed: int, optional (default: 42)
    :param iterations: The number of iterations used to construct the graph embedding.
    :type iterations: int, optional (default: 50)
    :param colors: An array of values that determine the color of each node in
        the graph, useful for highlighting different features of the data.
    :type colors: array-like of shape (n,) or list-like of size n
    :param agg: A function used to aggregate the `colors` data when multiple
        points are mapped to a single node. The final color of each node is
        obtained by mapping the aggregated value with the colormap `cmap`.
    :type agg: Callable, optional
    :param title: The title to be displayed alongside the figure.
    :type title: str, optional
    :param width: The desired width of the figure in pixels.
    :type width: int, optional (default: 512)
    :param height: The desired height of the figure in pixels.
    :type height: int, optional (default: 512)
    :param cmap: The name of a colormap used to map `color` data values,
        aggregated by `agg`, to actual RGBA colors.
    :type cmap: str, optional
    """

    def __init__(self, graph, dim,
                 seed=42,
                 iterations=50,
                 colors=None,
                 agg=np.nanmean,
                 title=None,
                 width=512,
                 height=512,
                 cmap='jet'):
        self.__graph = graph
        self.__dim = dim
        self.seed = seed
        self.iterations = iterations
        self.colors = colors
        self.agg = agg
        self.title = title
        self.width = width
        self.height = height
        self.cmap = cmap

    def plot(self):
        """
        Plot the Mapper graph.
        
        :return: A static matplotlib figure that can be displayed on screen and notebooks.
        :rtype: :class:`matplotlib.figure.Figure`, :class:`matplotlib.axes.Axes`
        """
        px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
        fig, ax = plt.subplots(figsize=(self.width * px, self.height * px))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        pos = _nodes_pos(self.__graph, self.__dim, self.seed, self.iterations)
        self._plot_edges(ax, pos)
        self._plot_nodes(ax, pos)
        return fig, ax

    def _plot_nodes(self, ax, nodes_pos):
        nodes_arr = _nodes_array(self.__graph, self.__dim, nodes_pos)
        attr_size = nx.get_node_attributes(self.__graph, ATTR_SIZE)
        max_size = max(attr_size.values()) if attr_size else 1.0
        colors_agg = aggregate_graph(self.colors, self.__graph, self.agg)
        marker_color = [colors_agg[n] for n in self.__graph.nodes()]
        marker_size = [200.0 * math.sqrt(attr_size[n] / max_size) for n in self.__graph.nodes()]
        verts = ax.scatter(
            x=nodes_arr[0],
            y=nodes_arr[1],
            c=marker_color,
            s=marker_size,
            cmap=self.cmap,
            alpha=1.0,
            vmin=min(marker_color),
            vmax=max(marker_color),
            edgecolors=_NODE_OUTER_COLOR,
            linewidths=_NODE_OUTER_WIDTH)
        colorbar = plt.colorbar(
            verts,
            orientation='vertical',
            location='right',
            aspect=40,
            pad=0.02,
            ax=ax,
            format="%.2g")
        colorbar.set_label(
            self.title,
            color=_NODE_OUTER_COLOR)
        colorbar.set_alpha(1.0)
        colorbar.outline.set_color(_NODE_OUTER_COLOR)
        colorbar.ax.yaxis.set_tick_params(
            color=_NODE_OUTER_COLOR,
            labelcolor=_NODE_OUTER_COLOR)
        colorbar.ax.tick_params(labelsize=8)
        colorbar.ax.locator_params(nbins=10)

    def _plot_edges(self, ax, nodes_pos):
        segments = [(nodes_pos[e[0]], nodes_pos[e[1]]) for e in self.__graph.edges()]
        lines = LineCollection(
            segments,
            color=_EDGE_COLOR,
            linewidth=_EDGE_WIDTH,
            alpha=1.0,
            zorder=-1,
            antialiased=True)
        ax.add_collection(lines)
