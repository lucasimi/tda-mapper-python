"""
This module provides functionalities to visualize the Mapper graph.
"""

import igraph as ig
import networkx as nx
import numpy as np

from tdamapper.plot_backends.plot_matplotlib import plot_matplotlib
from tdamapper.plot_backends.plot_plotly import plot_plotly, plot_plotly_update
from tdamapper.plot_backends.plot_pyvis import plot_pyvis


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
    :param iterations: The number of iterations used to construct the graph
        embedding. Defaults to 50.
    :type iterations: int, optional
    :param seed: The random seed used to construct the graph embedding.
        Defaults to None.
    :type seed: int, optional
    :param layout_engine: The engine used to compute the graph layout in the
        specified dimensions. Possible values are 'igraph' and 'networkx'.
        Defaults to 'igraph'.
    :type layout_engine: str, optional
    """

    def __init__(
        self,
        graph,
        dim,
        iterations=50,
        seed=None,
        layout_engine="igraph",
    ):
        self.graph = graph
        self.dim = dim
        self.iterations = iterations
        self.seed = seed
        self.layout_engine = layout_engine
        self.positions = self._compute_pos()

    def _compute_pos(self):
        if self.layout_engine == "igraph":
            return self._compute_pos_ig()
        elif self.layout_engine == "networkx":
            return self._compute_pos_nx()
        else:
            raise ValueError(
                f"Unknown engine {self.layout_engine}. "
                "Only possible values are 'igraph' and 'networkx'"
            )

    def _compute_pos_nx(self):
        return nx.spring_layout(
            self.graph,
            dim=self.dim,
            seed=self.seed,
            iterations=self.iterations,
        )

    def _compute_pos_ig(self):
        if self.graph.number_of_nodes() == 0:
            return {}
        rng = np.random.default_rng(self.seed)
        random_pos = rng.random((len(self.graph.nodes()), self.dim))
        graph_ig = ig.Graph.from_networkx(self.graph)
        if self.dim == 2:
            layout = graph_ig.layout_fruchterman_reingold(
                niter=self.iterations,
                seed=random_pos,
            )
            pos = {
                node: (layout[i][0], layout[i][1])
                for i, node in enumerate(self.graph.nodes())
            }
        elif self.dim == 3:
            layout = graph_ig.layout_fruchterman_reingold_3d(
                niter=self.iterations,
                seed=random_pos,
            )
            pos = {
                node: (layout[i][0], layout[i][1], layout[i][2])
                for i, node in enumerate(self.graph.nodes())
            }
        return pos

    def plot_matplotlib(
        self,
        colors,
        node_size=1,
        agg=np.nanmean,
        title=None,
        cmap="jet",
        width=512,
        height=512,
    ):
        """
        Draw a static plot using Matplotlib.

        :param colors: An array of values that determine the color of each
            node in the graph, useful for highlighting different features of
            the data.
        :type colors: array-like of shape (n,) or list-like of size n
        :param node_size: A scaling factor for node size. Defaults to 1.
        :type node_size: float, optional
        :param agg: A function used to aggregate the `colors` array over the
            points within a single node. The final color of each node is
            obtained by mapping the aggregated value with the colormap `cmap`.
            Defaults to `numpy.nanmean`.
        :type agg: Callable, optional
        :param title: The title to be displayed alongside the figure.
        :type title: str, optional
        :param cmap: The name of a colormap used to map `colors` data values,
            aggregated by `agg`, to actual RGBA colors. Defaults to 'jet'.
        :type cmap: str, optional
        :param width: The desired width of the figure in pixels. Defaults to
            512.
        :type width: int, optional
        :param height: The desired height of the figure in pixels. Defaults to
            512
        :type height: int, optional

        :return: A static matplotlib figure that can be displayed on screen
            and notebooks.
        :rtype: :class:`matplotlib.figure.Figure`,
            :class:`matplotlib.axes.Axes`
        """

        return plot_matplotlib(
            self,
            colors=colors,
            node_size=node_size,
            agg=agg,
            title=title,
            cmap=cmap,
            width=width,
            height=height,
        )

    def plot_plotly(
        self,
        colors,
        node_size=1,
        agg=np.nanmean,
        title=None,
        cmap="jet",
        width=None,
        height=None,
    ):
        """
        Draw an interactive plot using Plotly.

        :param colors: An array of values that determine the color of each
            node in the graph, useful for highlighting different features of
            the data.
        :type colors: array-like of shape (n,) or list-like of size n
        :param node_size: A scaling factor for node size. When node_size is a
            list, the figure will display a slider with the specified values.
            Defaults to 1.
        :type node_size: int, float or list, optional
        :param agg: A function used to aggregate the `colors` array over the
            points within a single node. The final color of each node is
            obtained by mapping the aggregated value with the colormap `cmap`.
            Defaults to `numpy.nanmean`.
        :type agg: Callable, optional
        :param title: The title for the colormap. When colors has shape (n, m)
            and title is a list of string, each item will be used as title for
            its corresponding colormap.
        :type title: str, list[str], optional
        :param cmap: The name of a colormap used to map `colors` data values,
            aggregated by `agg`, to actual RGBA colors. Defaults to 'jet'.
        :type cmap: str, optional
        :param width: The desired width of the figure in pixels.
        :type width: int, optional
        :param height: The desired height of the figure in pixels.
        :type height: int, optional

        :return: An interactive Plotly figure that can be displayed on screen
            and notebooks. For 3D embeddings, the figure requires a WebGL
            context to be shown.
        :rtype: :class:`plotly.graph_objects.Figure`
        """
        return plot_plotly(
            self,
            colors=colors,
            node_size=node_size,
            agg=agg,
            title=title,
            cmap=cmap,
            width=width,
            height=height,
        )

    def plot_plotly_update(
        self,
        fig,
        colors=None,
        node_size=None,
        agg=None,
        title=None,
        cmap=None,
        width=None,
        height=None,
    ):
        """
        Draw an interactive plot using Plotly on a previously rendered figure.

        This is typically faster than calling `MapperPlot.plot_plotly` on a
        new set of parameters.

        :param fig: A Plotly Figure object obtained by calling the method
            `MapperPlot.plot_plotly`.
        :type fig: :class:`plotly.graph_objects.Figure`
        :param colors: An array of values that determine the color of each
            node in the graph, useful for highlighting different features of
            the data. Defaults to None.
        :type colors: array-like of shape (n,) or list-like of size n, optional
        :param node_size: A scaling factor for node size. Defaults to None.
        :type node_size: float, optional
        :param agg: A function used to aggregate the `colors` array over the
            points within a single node. The final color of each node is
            obtained by mapping the aggregated value with the colormap `cmap`.
            Defaults to None.
        :type agg: Callable, optional
        :param title: The title to be displayed alongside the figure. Defaults
            to None.
        :type title: str, optional
        :param cmap: The name of a colormap used to map `colors` data values,
            aggregated by `agg`, to actual RGBA colors. Defaults to None.
        :type cmap: str, optional
        :param width: The desired width of the figure in pixels. Defaults to
            None.
        :type width: int, optional
        :param height: The desired height of the figure in pixels. Defaults to
            None.
        :type height: int, optional

        :return: An interactive Plotly figure that can be displayed on screen
            and notebooks. For 3D embeddings, the figure requires a WebGL
            context to be shown.
        :rtype: :class:`plotly.graph_objects.Figure`
        """
        return plot_plotly_update(
            self,
            fig,
            colors=colors,
            node_size=node_size,
            agg=agg,
            title=title,
            cmap=cmap,
            width=width,
            height=height,
        )

    def plot_pyvis(
        self,
        output_file,
        colors,
        node_size=1,
        agg=np.nanmean,
        title=None,
        cmap="jet",
        width=512,
        height=512,
    ):
        """
        Draw an interactive HTML plot using PyVis.

        :param output_file: The path where the html file is written.
        :type output_file: str
        :type colors: array-like of shape (n,) or list-like of size n
        :param colors: An array of values that determine the color of each
            node in the graph, useful for highlighting different features of
            the data.
        :type colors: array-like of shape (n,) or list-like of size n
        :param node_size: A scaling factor for node size. Defaults to 1.
        :type node_size: float, optional
        :param agg: A function used to aggregate the `colors` array over the
            points within a single node. The final color of each node is
            obtained by mapping the aggregated value with the colormap `cmap`.
            Defaults to `numpy.nanmean`.
        :type agg: Callable, optional
        :param title: The title to be displayed alongside the figure. Defaults
            to None.
        :type title: str, optional
        :param cmap: The name of a colormap used to map `colors` data values,
            aggregated by `agg`, to actual RGBA colors. Defaults to 'jet'.
        :type cmap: str, optional
        :param width: The desired width of the figure in pixels. Defaults to
            512.
        :type width: int, optional
        :param height: The desired height of the figure in pixels. Defaults to
            512.
        :type height: int, optional
        """
        return plot_pyvis(
            self,
            output_file=output_file,
            colors=colors,
            node_size=node_size,
            agg=agg,
            title=title,
            cmap=cmap,
            width=width,
            height=height,
        )
