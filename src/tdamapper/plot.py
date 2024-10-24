"""
This module provides functionalities to visualize the Mapper graph.
"""

import networkx as nx

import numpy as np

from tdamapper._plot_matplotlib import plot_matplotlib
from tdamapper._plot_plotly import plot_plotly, plot_plotly_update
from tdamapper._plot_pyvis import plot_pyvis


class MapperPlot:

    def __init__(self, graph, dim, iterations=50, seed=None):
        self.graph = graph
        self.dim = dim
        self.iterations = iterations
        self.seed = seed
        self.positions = nx.spring_layout(
            self.graph,
            dim=self.dim,
            seed=self.seed,
            iterations=self.iterations
        )

    def plot_matplotlib(
                self,
                colors,
                width=512,
                height=512,
                title=None,
                agg=np.nanmean,
                cmap='jet',
            ):
        return plot_matplotlib(
            self,
            colors=colors,
            width=width,
            height=height,
            title=title,
            agg=agg,
            cmap=cmap,
        )

    def plot_plotly(
                self,
                colors,
                width=512,
                height=512,
                title=None,
                agg=np.nanmean,
                cmap='jet'
            ):
        return plot_plotly(
            self,
            width=width,
            height=height,
            title=title,
            colors=colors,
            agg=agg,
            cmap=cmap
        )

    def plot_plotly_update(
                self,
                fig,
                width=None,
                height=None,
                title=None,
                colors=None,
                agg=None,
                cmap=None
            ):
        return plot_plotly_update(
            self,
            fig,
            width=width,
            height=height,
            title=title,
            colors=colors,
            agg=agg,
            cmap=cmap
        )

    def plot_pyvis(
                self,
                colors,
                notebook,
                output_file,
                width=512,
                height=512,
                agg=np.nanmean,
                cmap='jet'
            ):
        return plot_pyvis(
            self,
            width=width,
            height=height,
            colors=colors,
            agg=agg,
            cmap=cmap,
            notebook=notebook,
            output_file=output_file
        )
