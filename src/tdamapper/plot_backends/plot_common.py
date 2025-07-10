"""
Common types and protocols for Mapper plot backends.
"""

from typing import Any, Protocol

import networkx as nx
import numpy as np
from numpy.typing import NDArray


class MapperPlotType(Protocol):
    """
    Protocol for the Mapper plot type.

    This protocol defines the expected attributes for a Mapper plot type, which
    includes the dimension of the plot, the graph structure, and the positions
    of the nodes in the graph. The `dim` attribute represents the dimension of
    the plot, `graph` is a NetworkX graph object representing the Mapper graph,
    and `positions` is a dictionary mapping nodes to their positions in the
    plot.
    """

    @property
    def dim(self) -> int:
        """
        Returns the dimension of the plot.
        """

    @property
    def graph(self) -> nx.Graph:
        """
        Returns the NetworkX graph object representing the Mapper graph.
        """

    @property
    def positions(self) -> dict[Any, NDArray[np.float64]]:
        """
        Returns a dictionary mapping nodes to their positions in the plot.
        The keys are node identifiers, and the values are NumPy arrays of
        coordinates in the plot's dimension.
        """
