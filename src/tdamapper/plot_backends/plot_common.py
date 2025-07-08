from typing import Any, Protocol

import networkx as nx
import numpy as np
from numpy.typing import NDArray


class MapperPlotType(Protocol):

    dim: int
    graph: nx.Graph
    positions: dict[Any, NDArray[np.float64]]
