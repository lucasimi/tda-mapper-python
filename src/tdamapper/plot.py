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


def _node_pos(graph, dim, seed, iterations):
    return nx.spring_layout(
        graph,
        dim=dim,
        seed=seed,
        iterations=iterations)


def _node_col(graph, colors, agg, default=0.5):
    if colors is not None:
        return aggregate_graph(colors, graph, agg)
    else:
        return [default for _ in graph.nodes()]


def _node_pos_array(graph, dim, node_pos):
    return tuple([node_pos[n][i] for n in graph.nodes()] for i in range(dim))


def _edge_pos_array(graph, dim, node_pos):
    edges_arr = tuple([] for i in range(dim))
    for edge in graph.edges():
        pos0, pos1 = node_pos[edge[0]], node_pos[edge[1]]
        for i in range(dim):
            edges_arr[i].append(pos0[i])
            edges_arr[i].append(pos1[i])
            edges_arr[i].append(None)
    return edges_arr


def _fmt(x, max_len=3):
    fmt = f'.{max_len}g'
    return f'{x:{fmt}}'


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
        self.__seed = seed
        self.__iterations = iterations
        self.__colors = colors
        self.__agg = agg
        self.__title = title
        self.__width = width
        self.__height = height
        self.__cmap = cmap
        node_col = _node_col(self.__graph, self.__colors, self.__agg)
        self.__fig = self._figure(node_col)

    def _nodes_trace(self, node_pos_arr, node_col):
        attr_size = nx.get_node_attributes(self.__graph, ATTR_SIZE)
        max_size = max(attr_size.values()) if attr_size else 1.0
        scatter_text = self._text(node_col)
        marker_size = [25.0 * math.sqrt(attr_size[n] / max_size) for n in self.__graph.nodes()]
        colors = list(node_col.values())
        scatter = dict(
            name='nodes_trace',
            x=node_pos_arr[0],
            y=node_pos_arr[1],
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
                line_color=_NODE_OUTER_COLOR,
                line_colorscale=self.__cmap,
                color=colors,
                colorscale=self.__cmap,
                cmin=min(colors),
                cmax=max(colors),
                colorbar=self._colorbar()))
        if self.__dim == 3:
            scatter.update(dict(
                z=node_pos_arr[2]))
            return go.Scatter3d(scatter)
        elif self.__dim == 2:
            return go.Scatter(scatter)

    def _edges_trace(self, edge_pos_arr, node_col):
        scatter = dict(
            name='edges_trace',
            x=edge_pos_arr[0],
            y=edge_pos_arr[1],
            mode='lines',
            opacity=_EDGE_OPACITY,
            line_width=_EDGE_WIDTH,
            line_color=_EDGE_COLOR,
            hoverinfo='skip')
        if self.__dim == 3:
            colors_avg = []
            for e in self.__graph.edges():
                c0, c1 = node_col[e[0]], node_col[e[1]]
                colors_avg.append(c0)
                colors_avg.append(c1)
                colors_avg.append(c1)
            colors = list(node_col.values())
            scatter.update(dict(
                z=edge_pos_arr[2],
                line_color=colors_avg,
                line_cmin=min(colors),
                line_cmax=max(colors),
                line_colorscale=self.__cmap))
            return go.Scatter3d(scatter)
        elif self.__dim == 2:
            scatter.update(dict(
                marker_colorscale=self.__cmap,
                marker_line_colorscale=self.__cmap))
            return go.Scatter(scatter)

    def _layout(self):
        line_col = 'rgba(230, 230, 230, 1.0)'
        axis = dict(
            showline=True,
            linewidth=1,
            mirror=True,
            visible=True,
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            title='')
        scene_axis = dict(
            showgrid=True,
            visible=True,
            backgroundcolor='rgba(0, 0, 0, 0)',
            showaxeslabels=False,
            showline=True,
            linecolor=line_col,
            zerolinecolor=line_col,
            gridcolor=line_col,
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
            width=self.__width,
            height=self.__height,
            scene=dict(
                xaxis=scene_axis,
                yaxis=scene_axis,
                zaxis=scene_axis))

    def _figure(self, node_col):
        node_pos = _node_pos(self.__graph, self.__dim, self.__seed, self.__iterations)
        node_pos_arr = _node_pos_array(self.__graph, self.__dim, node_pos)
        edge_pos_arr = _edge_pos_array(self.__graph, self.__dim, node_pos)
        _edges_tr = self._edges_trace(edge_pos_arr, node_col)
        _nodes_tr = self._nodes_trace(node_pos_arr, node_col)
        _layout = self._layout()
        return go.Figure(
            data=[_edges_tr, _nodes_tr],
            layout=_layout)

    def _colorbar(self):
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
        if self.__title is not None:
            cbar['title'] = self.__title
        if self.__dim == 3:
            return go.scatter3d.marker.ColorBar(cbar)
        elif self.__dim == 2:
            return go.scatter.marker.ColorBar(cbar)

    def _text(self, colors):
        attr_size = nx.get_node_attributes(self.__graph, ATTR_SIZE)

        def _lbl(n):
            col = _fmt(colors[n], 3)
            size = _fmt(attr_size[n], 5)
            return f'color: {col}<br>node: {n}<br>size: {size}'
        return [_lbl(n) for n in self.__graph.nodes()]

    def _update_traces_pos(self):
        pos = _node_pos(self.__graph, self.__dim, self.__seed, self.__iterations)
        node_arr = _node_pos_array(self.__graph, self.__dim, pos)
        edge_arr = _edge_pos_array(self.__graph, self.__dim, pos)
        if self.__dim == 3:
            self.__fig.update_traces(
                patch=dict(
                    x=node_arr[0],
                    y=node_arr[1],
                    z=node_arr[2]),
                selector=dict(
                    name='nodes_trace'))
            self.__fig.update_traces(
                patch=dict(
                    x=edge_arr[0],
                    y=edge_arr[1],
                    z=edge_arr[2]),
                selector=dict(
                    name='edges_trace'))
        elif self.__dim == 2:
            self.__fig.update_traces(
                patch=dict(
                    x=node_arr[0],
                    y=node_arr[1]),
                selector=dict(
                    name='nodes_trace'))
            self.__fig.update_traces(
                patch=dict(
                    x=edge_arr[0],
                    y=edge_arr[1]),
                selector=dict(
                    name='edges_trace'))

    def _update_traces_col(self):
        if (self.__colors is not None) and (self.__agg is not None):
            nodes_col = _node_col(self.__graph, self.__colors, self.__agg)
            colors_list = list(nodes_col.values())
            self._update_node_trace_col(nodes_col, colors_list)
            self._update_edge_trace_col(nodes_col, colors_list)

    def _update_edge_trace_col(self, colors_agg, colors_list):
        colors_avg = []
        for edge in self.__graph.edges():
            c0, c1 = colors_agg[edge[0]], colors_agg[edge[1]]
            colors_avg.append(c0)
            colors_avg.append(c1)
            colors_avg.append(c1)
        if not colors_avg:
            return
        if self.__dim == 3:
            self.__fig.update_traces(
                patch=dict(
                    line_color=colors_avg,
                    line_colorscale=self.__cmap,
                    line_cmax=max(colors_list),
                    line_cmin=min(colors_list)),
                selector=dict(
                    name='edges_trace'))

    def _update_node_trace_col(self, colors_agg, colors_list):
        self.__fig.update_traces(
            patch=dict(
                text=self._text(colors_agg),
                marker_color=colors_list,
                marker_cmax=max(colors_list),
                marker_cmin=min(colors_list)),
            selector=dict(
                name='nodes_trace'))

    def _update_traces_cmap(self):
        self.__fig.update_traces(
            patch=dict(
                marker_colorscale=self.__cmap,
                marker_line_colorscale=self.__cmap),
            selector=dict(
                name='nodes_trace'))
        if self.__dim == 3:
            self.__fig.update_traces(
                patch=dict(
                    line_colorscale=self.__cmap),
                selector=dict(
                    name='edges_trace'))

    def _update_traces_title(self):
        self.__fig.update_traces(
            patch=dict(
                marker_colorbar=self._colorbar()),
            selector=dict(
                name='nodes_trace'))

    def _update_layout(self):
        self.__fig.update_layout(
            width=self.__width,
            height=self.__height)

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
        _update_col = False
        _update_layout = False
        if seed is not None:
            self.__seed = seed
            _update_pos = True
        if iterations is not None:
            self.__iterations = iterations
            _update_pos = True
        if _update_pos:
            self._update_traces_pos()
        if agg is not None:
            self.__agg = agg
        if (colors is not None) and (agg is not None):
            self.__colors = colors
            self.__agg = agg
            _update_col = True
        if (colors is not None) and (self.__agg is not None):
            self.__colors = colors
            _update_col = True
        if (self.__colors is not None) and (agg is not None):
            self.__agg = agg
            _update_col = True
        if _update_col:
            self._update_traces_col()
        if cmap is not None:
            self.__cmap = cmap
            self._update_traces_cmap()
        if title is not None:
            self.__title = title
            self._update_traces_title()
        if (width is not None) and (height is not None):
            self.__width = width
            self.__height = height
            _update_layout = True
        if (width is not None) and (self.__height is not None):
            self.__width = width
            _update_layout = True
        if height is not None:
            self.__height = height
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
        return self.__fig


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
        self.__seed = seed
        self.__iterations = iterations
        self.__colors = colors
        self.__agg = agg
        self.__title = title
        self.__width = width
        self.__height = height
        self.__cmap = cmap

    def plot(self):
        """
        Plot the Mapper graph.
        
        :return: A static matplotlib figure that can be displayed on screen and notebooks.
        :rtype: :class:`matplotlib.figure.Figure`, :class:`matplotlib.axes.Axes`
        """
        px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
        fig, ax = plt.subplots(figsize=(self.__width * px, self.__height * px))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        pos = _node_pos(self.__graph, self.__dim, self.__seed, self.__iterations)
        self._plot_edges(ax, pos)
        self._plot_nodes(ax, pos)
        return fig, ax

    def _plot_nodes(self, ax, nodes_pos):
        nodes_arr = _node_pos_array(self.__graph, self.__dim, nodes_pos)
        attr_size = nx.get_node_attributes(self.__graph, ATTR_SIZE)
        max_size = max(attr_size.values()) if attr_size else 1.0
        colors_agg = _node_col(self.__graph, self.__colors, self.__agg)
        marker_color = [colors_agg[n] for n in self.__graph.nodes()]
        marker_size = [200.0 * math.sqrt(attr_size[n] / max_size) for n in self.__graph.nodes()]
        verts = ax.scatter(
            x=nodes_arr[0],
            y=nodes_arr[1],
            c=marker_color,
            s=marker_size,
            cmap=self.__cmap,
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
            self.__title,
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
