"""
This module provides functionalities to visualize the Mapper graph.
"""

import networkx as nx

import numpy as np

from tdamapper._plot_matplotlib import plot_matplotlib
from tdamapper._plot_plotly import plot_plotly, plot_plotly_update
from tdamapper._plot_pyvis import plot_pyvis


class MapperPlot:
    """
    Class for generating and visualizing the Mapper graph.

    This class creates a metric embedding of the Mapper graph in 2D or 3D and
    converts it into a plot.

    :param graph: The precomputed Mapper graph to be embedded. This can be
        obtained by calling :func:`tdamapper.core.mapper_graph` or
        :func:`tdamapper.core.MapperAlgorithm.fit_transform`.
    :type graph: :class:`networkx.Graph`, required
    :param dim: The dimension of the graph embedding (2 or 3).
    :type dim: int
    :param iterations: The number of iterations used to construct the graph embedding.
    :type iterations: int, optional (default: 50)
    :param seed: The random seed used to construct the graph embedding.
    :type seed: int, optional
    """

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
                agg=np.nanmean,
                title=None,
                width=512,
                height=512,
                cmap='jet'
            ):
        """
        Draw a static plot using Matplotlib.

        :param colors: An array of values that determine the color of each
            node in the graph, useful for highlighting different features of
            the data.
        :type colors: array-like of shape (n,) or list-like of size n
        :param agg: A function used to aggregate the `colors` array over the
            points within a single node. The final color of each node is
            obtained by mapping the aggregated value with the colormap `cmap`.
        :type agg: Callable, optional (default: `numpy.nanmean`)
        :param title: The title to be displayed alongside the figure.
        :type title: str, optional
        :param width: The desired width of the figure in pixels.
        :type width: int, optional (default: 512)
        :param height: The desired height of the figure in pixels.
        :type height: int, optional (default: 512)
        :param cmap: The name of a colormap used to map `colors` data values,
            aggregated by `agg`, to actual RGBA colors.
        :type cmap: str, optional (default: 'jet')
        """

        return plot_matplotlib(
            self,
            colors=colors,
            agg=agg,
            title=title,
            width=width,
            height=height,
            cmap=cmap,
        )

    def plot_plotly(
                self,
                colors,
                agg=np.nanmean,
                title=None,
                width=512,
                height=512,
                cmap='jet'
            ):
        """
        Draw an interactive plot using Plotly.

        :param colors: An array of values that determine the color of each
            node in the graph, useful for highlighting different features of
            the data.
        :type colors: array-like of shape (n,) or list-like of size n
        :param agg: A function used to aggregate the `colors` array over the
            points within a single node. The final color of each node is
            obtained by mapping the aggregated value with the colormap `cmap`.
        :type agg: Callable, optional (default: `numpy.nanmean`)
        :param title: The title to be displayed alongside the figure.
        :type title: str, optional
        :param width: The desired width of the figure in pixels.
        :type width: int, optional (default: 512)
        :param height: The desired height of the figure in pixels.
        :type height: int, optional (default: 512)
        :param cmap: The name of a colormap used to map `colors` data values,
            aggregated by `agg`, to actual RGBA colors.
        :type cmap: str, optional (default: 'jet')
        """
        return plot_plotly(
            self,
            colors=colors,
            agg=agg,
            title=title,
            width=width,
            height=height,
            cmap=cmap
        )

    def plot_plotly_update(
                self,
                fig,
                colors=None,
                agg=None,
                title=None,
                width=None,
                height=None,
                cmap=None
            ):
        """
        Draw an interactive plot using Plotly on a previously rendered figure.

        This is typically faster than calling `MapperPlot.plot_plotly` on a
        new set of parameters.

        :param fig: A Plotly Figure object obtained by calling the method
            `MapperPlot.plot_plotly`.
        :type fig: A Plotly Figure object
        :param colors: An array of values that determine the color of each
            node in the graph, useful for highlighting different features of
            the data.
        :type colors: array-like of shape (n,) or list-like of size n, optional
        :param agg: A function used to aggregate the `colors` array over the
            points within a single node. The final color of each node is
            obtained by mapping the aggregated value with the colormap `cmap`.
        :type agg: Callable, optional
        :param title: The title to be displayed alongside the figure.
        :type title: str, optional
        :param width: The desired width of the figure in pixels.
        :type width: int, optional
        :param height: The desired height of the figure in pixels.
        :type height: int, optional
        :param cmap: The name of a colormap used to map `colors` data values,
            aggregated by `agg`, to actual RGBA colors.
        :type cmap: str, optional
        """
        return plot_plotly_update(
            self,
            fig,
            colors=colors,
            agg=agg,
            title=title,
            width=width,
            height=height,
            cmap=cmap
        )

    def plot_pyvis(
                self,
                notebook,
                output_file,
                colors,
                agg=np.nanmean,
                title=None,
                width=512,
                height=512,
                cmap='jet'
            ):
        """
        Draw an interactive HTML plot using PyVis.

        :param notebook: Set to true when running inside Jupyter notebooks.
        :type notebook: bool
        :param output_file: The path where the html file is written.
        :type output_file: str
        :type colors: array-like of shape (n,) or list-like of size n
        :param colors: An array of values that determine the color of each
            node in the graph, useful for highlighting different features of
            the data.
        :type colors: array-like of shape (n,) or list-like of size n
        :param agg: A function used to aggregate the `colors` array over the
            points within a single node. The final color of each node is
            obtained by mapping the aggregated value with the colormap `cmap`.
        :type agg: Callable, optional (default: `numpy.nanmean`)
        :param title: The title to be displayed alongside the figure.
        :type title: str, optional
        :param width: The desired width of the figure in pixels.
        :type width: int, optional (default: 512)
        :param height: The desired height of the figure in pixels.
        :type height: int, optional (default: 512)
        :param cmap: The name of a colormap used to map `colors` data values,
            aggregated by `agg`, to actual RGBA colors.
        :type cmap: str, optional (default: 'jet')
        """
        return plot_pyvis(
            self,
            notebook=notebook,
            output_file=output_file,
            colors=colors,
            agg=agg,
            title=title,
            width=width,
            height=height,
            cmap=cmap,
        )
