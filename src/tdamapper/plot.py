import math
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from tdamapper.core import ATTR_SIZE, compute_local_interpolation


_NODE_OUTER_WIDTH = 0.75
_NODE_OUTER_COLOR = '#777'

_EDGE_WIDTH = 0.75
_EDGE_COLOR = '#777'

_TICKS_NUM = 10


class MapperPlot:

    def __init__(self, X, graph,
        colors=None,
        agg=np.nanmean,
        cmap='jet',
        **kwargs):
        self.__X = X
        self.__graph = graph
        self.__cmap = cmap
        item_colors = [np.nanmean(x) for x in self.__X] if colors is None else colors
        self.__colors = compute_local_interpolation(item_colors, self.__graph, agg)
        self.__dim = kwargs.get('dim', 2)
        self.__kwargs = {}
        self.__kwargs.update(kwargs)
        self.__kwargs['dim'] = self.__dim
        self.__pos = self._init_positions(self.__graph, **kwargs)
        self.__kwargs['pos'] = self.__pos

    def _init_positions(self, g, **kwargs):
        pos = kwargs.get('pos')
        if pos is None:
            return nx.spring_layout(g, **kwargs)
        return pos

    def with_colors(self, colors, agg=np.nanmean, cmap='jet'):
        return MapperPlot(self.__X, self.__graph,
            colors=colors, agg=agg, cmap=cmap, **self.__kwargs)

    def _plot_static(self, title='', ax=None):
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(6, 6))
        return self._plot_matplotlib(title, ax)

    def plot(self, *args, style='interactive', **kwargs):
        if not self.__pos:
            return
        if self.__dim == 2:
            if style == 'interactive':
                return self._plot_interactive_2d(*args, **kwargs)
            return self._plot_static(*args, **kwargs)
        if self.__dim == 3:
            return self._plot_interactive_3d(*args, **kwargs)

    def _plot_interactive_2d(self, title='', width=512, height=512):
        return self._plot_plotly_2d(title, width, height)

    def _plot_interactive_3d(self, title='', width=512, height=512):
        return self._plot_plotly_3d(title, width, height)

    def _plot_matplotlib_nodes(self, title, ax):
        nodes = self.__graph.nodes()
        sizes = nx.get_node_attributes(self.__graph, ATTR_SIZE)
        max_size = max(sizes.values())
        min_color = min(self.__colors.values())
        max_color = max(self.__colors.values())
        nodes_x = [self.__pos[node][0] for node in nodes]
        nodes_y = [self.__pos[node][1] for node in nodes]
        nodes_c = [self.__colors[node] for node in nodes]
        nodes_s = [250.0 * math.sqrt(sizes[node]/max_size) for node in nodes]
        verts = ax.scatter(
            x=nodes_x,
            y=nodes_y,
            c=nodes_c,
            s=nodes_s,
            cmap=self.__cmap,
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

    def _plot_matplotlib_edges(self, ax):
        edges = self.__graph.edges()
        segments = [(self.__pos[edge[0]], self.__pos[edge[1]]) for edge in edges]
        lines = LineCollection(
            segments,
            color=_EDGE_COLOR,
            linewidth=_EDGE_WIDTH,
            alpha=1.0,
            zorder=-1,
            antialiased=True
        )
        ax.add_collection(lines)

    def _plot_matplotlib(self, title, ax):
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        self._plot_matplotlib_edges(ax)
        self._plot_matplotlib_nodes(title, ax)

    def _plot_plotly_2d(self, title, width, height):
        edge_trace = self._plot_plotly_2d_edges()
        node_trace = self._plot_plotly_2d_nodes(title)
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

    def _plot_plotly_2d_edges(self):
        edge_x, edge_y = [], []
        for edge in self.__graph.edges():
            x0, y0 = self.__pos[edge[0]]
            x1, y1 = self.__pos[edge[1]]
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
            hoverinfo='none'
        )
        return edge_trace

    def _fmt(self, x, max_len=3):
        fmt = f'.{max_len}g'
        return f'{x:{fmt}}'

    def _plotly_colorbar_2d(self, title):
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

    def _plot_plotly_2d_nodes(self, title):
        nodes = self.__graph.nodes()
        node_x, node_y = [], []
        node_captions = []
        for node in nodes:
            x, y = self.__pos[node]
            node_x.append(x)
            node_y.append(y)
            size = nodes[node][ATTR_SIZE]
            node_label = self._plotly_label(node, size, self.__colors[node])
            node_captions.append(node_label)
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers',
            hoverinfo='text',
            opacity=1.0,
            marker=self._plotly_node_marker_2d(title),
            text=node_captions,
        )
        return node_trace

    def _plotly_label(self, node_id, size, color):
        node_label_size = self._fmt(size, 5)
        node_label_color = self._fmt(color, 3)
        return f'color: {node_label_color}<br>node: {node_id}<br>size: {node_label_size}'

    def _plotly_node_marker_2d(self, title):
        nodes = self.__graph.nodes()
        colors = [self.__colors[node] for node in nodes]
        min_color = min(self.__colors.values())
        max_color = max(self.__colors.values())
        sizes = nx.get_node_attributes(self.__graph, ATTR_SIZE)
        max_size = max(sizes.values()) if sizes else 1.0
        node_sizes = [25.0 * math.sqrt(sizes[node] / max_size) for node in nodes]
        return go.scatter.Marker(
            showscale=True,
            colorscale=self.__cmap,
            reversescale=False,
            color=colors,
            cmax=max_color,
            cmin=min_color,
            opacity=1.0,
            size=node_sizes,
            colorbar=self._plotly_colorbar_2d(title),
            line_width=_NODE_OUTER_WIDTH,
            line_color=_NODE_OUTER_COLOR,
        )

    def _plot_plotly_3d(self, title, width, height):
        edge_trace = self._plot_plotly_3d_edges()
        node_trace = self._plot_plotly_3d_nodes(title)
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

    def _plot_plotly_3d_edges(self):
        edge_x, edge_y, edge_z = [], [], []
        for edge in self.__graph.edges():
            x0, y0, z0 = self.__pos[edge[0]]
            x1, y1, z1 = self.__pos[edge[1]]
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
            hoverinfo='none'
        )
        return edge_trace

    def _plotly_colorbar_3d(self, title):
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

    def _plotly_node_marker_3d(self, title):
        nodes = self.__graph.nodes()
        colors = [self.__colors[node] for node in nodes]
        min_color = min(self.__colors.values())
        max_color = max(self.__colors.values())
        sizes = nx.get_node_attributes(self.__graph, ATTR_SIZE)
        max_size = max(sizes.values()) if sizes else 1.0
        node_sizes = [25.0 * math.sqrt(sizes[node] / max_size) for node in nodes]
        return go.scatter3d.Marker(
            showscale=True,
            colorscale=self.__cmap,
            reversescale=False,
            color=colors,
            cmax=max_color,
            cmin=min_color,
            opacity=1.0,
            size=node_sizes,
            colorbar=self._plotly_colorbar_3d(title),
            line_width=_NODE_OUTER_WIDTH,
            line_color=_NODE_OUTER_COLOR,
            line_colorscale=self.__cmap,
        )

    def _plot_plotly_3d_nodes(self, title):
        nodes = self.__graph.nodes()
        node_x, node_y, node_z = [], [], []
        node_captions = []
        for node in nodes:
            x, y, z = self.__pos[node]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)
            size = nodes[node][ATTR_SIZE]
            node_label = self._plotly_label(node, size, self.__colors[node])
            node_captions.append(node_label)
        node_trace = go.Scatter3d(
            x=node_x,
            y=node_y,
            z=node_z,
            mode='markers',
            hoverinfo='text',
            opacity=1.0,
            marker=self._plotly_node_marker_3d(title),
            text=node_captions,
        )
        return node_trace
