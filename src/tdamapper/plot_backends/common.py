from typing import Any, Protocol

import networkx as nx


class MapperPlotType(Protocol):

    @property
    def graph(self) -> nx.Graph: ...

    @property
    def dim(self) -> int: ...

    @property
    def positions(self) -> Any: ...
