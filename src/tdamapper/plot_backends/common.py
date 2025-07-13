"""
This module defines structures and protocols used in the plot backends.
"""

from typing import Literal, Protocol

import networkx as nx


class MapperPlotType(Protocol):
    """
    A protocol defining the structure of a Mapper plot.
    """

    @property
    def graph(self) -> nx.Graph:
        """
        Get the Mapper graph from the Mapper plot.
        """

    @property
    def dim(self) -> Literal[2, 3]:
        """
        Get the dimension of the Mapper plot.
        """

    @property
    def positions(self) -> dict[int, tuple[float, ...]]:
        """
        Get the positions of the nodes in the Mapper graph.
        """
