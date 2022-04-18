import math

import networkx as nx
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt


class Network:

    def __init__(self, graph):
        self.__nx = nx.Graph()
        self.__graph = graph
        graph._compute_ccs()
        for u in graph.get_vertices():
            u_vert = graph.get_vertex(u)
            u_size = u_vert.get_size()
            u_cc = graph.get_vertex_cc(u)
            self.__nx.add_node(u, size=u_size, label=u_cc)
            for v in graph.get_adjaciency(u):
                uv_weight = graph.get_edge(u, v).get_weight()
                self.__nx.add_edge(u, v, weight=uv_weight)
        self.__size_dict = nx.get_node_attributes(self.__nx, 'size')
        self.__label_dict = nx.get_node_attributes(self.__nx, 'label')
        self.__pos2d_dict = nx.spring_layout(self.__nx, dim=2)
        self.__pos3d_dict = nx.spring_layout(self.__nx, dim=3)
        self.__vertex_colors = None
        self.__cc_colors = None

    def _compute_colors(self, colors, aggfunc):
        self.__vertex_colors = {}
        for u in self.__graph.get_vertices():
            u_colors = [colors[i] for i in self.__graph.get_vertex_ids(u)]
            self.__vertex_colors[u] = aggfunc(u_colors)
        self.__cc_colors = {}
        for cc in self.__graph.get_ccs():
            cc_colors = [colors[i] for i in self.__graph.get_cc_ids(cc)]
            cc_color = aggfunc(cc_colors)
            for u in self.__graph.get_cc_vertices(cc):
                self.__cc_colors[u] = cc_color

    def plot(self, colors, aggfunc=np.nanmean, title='Node value', frontend='plotly', width=512, height=512, use_cc_colors=False):
        self._compute_colors(colors, aggfunc)
        if frontend == 'plotly':
            return self._plot_plotly_2d(title, width, height, use_cc_colors)
        if frontend == 'pyplot':
            return self._plot_pyplot(title, width, height, use_cc_colors)
        if frontend == '3d':
            return self._plot_plotly_3d(title, width, height, use_cc_colors)

    def _plot_pyplot(self, title, width, height, use_cc_colors):
        px = 1/plt.rcParams['figure.dpi']
        max_size = max(self.__size_dict.values()) if self.__size_dict else 1.0
        node_sizes = []
        for node in self.__nx.nodes():
            size = float(self.__size_dict[node]) / max_size
            node_sizes.append(600.0 * size)
        colors = list(self.__cc_colors) if use_cc_colors else list(self.__vertex_colors)
        fig, ax = plt.subplots(figsize=(width * px, height * px))
        ax.set_facecolor('#fff')
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(0)
        plt.set_cmap('viridis_r')
        nx.draw_networkx_edges(self.__nx,
            self.__pos2d_dict,
            edge_color='#111',
            alpha=0.25,
            ax=ax)
        nodes = nx.draw_networkx_nodes(self.__nx,
            self.__pos2d_dict,
            node_color=colors,
            node_size=node_sizes,
            alpha=1.0,
            edgecolors='#111',
            ax=ax)
        colorbar = fig.colorbar(nodes, orientation='vertical', aspect=60, pad=0.0)
        colorbar.set_label(title)
        colorbar.outline.set_linewidth(0)
        return fig

    def _plot_plotly_2d_nodes(self, title, use_cc_colors):
        node_x, node_y = [], []
        node_colors, node_sizes = [], []
        node_texts = []
        colors = self.__cc_colors if use_cc_colors else self.__vertex_colors
        max_color = max(self.__vertex_colors.values())
        min_color = min(self.__vertex_colors.values())
        max_size = max(self.__size_dict.values()) if self.__size_dict else 1.0
        for node in self.__nx.nodes():
            x, y = self.__pos2d_dict[node]
            color = colors[node]
            size = float(self.__size_dict[node]) / max_size
            node_x.append(x)
            node_y.append(y)
            node_colors.append(color)
            node_sizes.append(30.0 * math.sqrt(size))
            size_text = self.__size_dict[node]
            color_text = colors[node]
            label = self.__label_dict[node]
            txt = f'size: {size_text}, color: {color_text:.2f}, cc: {label}'
            node_texts.append(txt)
        node_trace = go.Scatter(
            x=node_x, y=node_y,
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
                opacity=0.85,
                size=node_sizes,
                colorbar=dict(
                    thickness=12,
                    title=title,
                    xanchor='left',
                    titleside='right',
                    xpad=0
                ),
                line_width=1.0,
                line_color='rgba(0.25, 0.25, 0.25, 1.0)'))
        node_trace.text = node_texts
        return node_trace

    def _plot_plotly_2d_edges(self):
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
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            opacity=0.75,
            line=dict(width=1.0, color='rgba(0.5, 0.5, 0.5, 0.5)'),
            hoverinfo='none'
        )
        return edge_trace

    def _plot_plotly_2d(self, title, width, height, use_cc_colors):
        edge_trace = self._plot_plotly_2d_edges()
        node_trace = self._plot_plotly_2d_nodes(title, use_cc_colors)
        axis = dict(showbackground=False,
                showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                title='')
        fig = go.Figure(data=[edge_trace, node_trace],
            layout=go.Layout(
                width=width, height=height,
                plot_bgcolor='rgba(0, 0, 0, 0)',
                autosize=False,
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
        
    def _plot_plotly_3d_edges(self):
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

    def _plot_plotly_3d_nodes(self, title, use_cc_colors):
        node_x, node_y, node_z = [], [], []
        node_colors, node_sizes = [], []
        node_texts = []
        colors = self.__cc_colors if use_cc_colors else self.__vertex_colors
        max_color = max(self.__vertex_colors.values())
        min_color = min(self.__vertex_colors.values())
        max_size = max(self.__size_dict.values()) if self.__size_dict else 1.0
        for node in self.__nx.nodes():
            x, y, z = self.__pos3d_dict[node]
            color = colors[node]
            size = float(self.__size_dict[node]) / max_size
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)
            node_colors.append(color)
            node_sizes.append(20.0 * math.sqrt(size))
            size_text = self.__size_dict[node]
            color_text = colors[node]
            label = self.__label_dict[node]
            txt = f'size: {size_text}, color: {color_text:.2f}, cc: {label}'
            node_texts.append(txt)
        node_trace = go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers',
            hoverinfo='text',
            opacity=1.0,
            marker=dict(
                showscale=True,
                colorscale='viridis',
                reversescale=True,
                color=node_colors,
                opacity=0.85,
                size=node_sizes,
                cmax=max_color,
                cmin=min_color,
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

    def _plot_plotly_3d(self, title, width, height, use_cc_colors):
        edge_trace = self._plot_plotly_3d_edges()
        node_trace = self._plot_plotly_3d_nodes(title, use_cc_colors)
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
                autosize=False,
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
