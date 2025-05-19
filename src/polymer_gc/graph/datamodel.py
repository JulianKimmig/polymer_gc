from __future__ import annotations
import numpy as np
from typing import List, Optional, Any, Union, Tuple
import networkx as nx
from ..core.monomer import Monomer
from .plot import plot_polygraph


def make_nx_polygraph(g: "PolyGraph"):
    G = nx.Graph()
    for i, idx in enumerate(g):
        G.add_node(i, idx=idx)

    for edge in g.iter_edges():
        G.add_edge(edge[0], edge[1])

    return G


class PolyGraph:
    def __init__(
        self,
        nodes: np.ndarray,
        edges: np.ndarray,
        monomers: List[Monomer],
    ):
        self._nodes = nodes
        # make self._nodes read only
        self._nodes.flags.writeable = False

        self._edges = np.array(edges).astype(np.uint32)
        
        # make self._edges read only
        self._edges.flags.writeable = False

        self._nx_graph = None
        self._mass = None

        # make sure monomers is at list the size of indices in the graph
        if max(np.unique(nodes)) >= len(monomers):
            raise ValueError(
                f"The number of monomers must be at least the size of the graph but is {len(monomers)} and the graph has {max(np.unique(nodes))} indices."
            )
        self.monomers = monomers

    @property
    def nodes(self) -> np.ndarray:
        """
        Returns the nodes of the graph.
        """
        return self._nodes.copy()

    @property
    def edges(self) -> np.ndarray:
        """
        Returns the edges of the graph.
        """
        return self._edges.copy()

    def __iter__(self):
        return iter(self._nodes)

    def __len__(self):
        return len(self._nodes)

    def iter_edges(self):
        return iter(self._edges)

    def iter_nodes(self):
        return iter(self._nodes)

    ### magic method to display in jupyter notebook
    def _ipython_display_(self):
        fig = plot_polygraph(self.nx_graph)
        fig.show()
        return fig

    @property
    def nx_graph(self):
        if self._nx_graph is None:
            self._nx_graph = make_nx_polygraph(self)
        return self._nx_graph

    @property
    def mass(self) -> float:
        if self._mass is None:
            monomer_masses = np.array([m.mass for m in self.monomers])
            self._mass = monomer_masses[self._nodes].sum()
        return self._mass


class PolyGraphEnsemble:
    def __init__(
        self,
        graphs: List[PolyGraph],
    ):
        self.graphs = graphs
        # check that all monomers are the same
        for g in graphs[1:]:
            if len(g.monomers) != len(self.monomers):
                raise ValueError("All graphs must have the same number of monomers.")
            for m1, m2 in zip(self.monomers, g.monomers):
                if m1 != m2:
                    raise ValueError("All graphs must have the same monomers.")

    @property
    def monomers(self):
        return self.graphs[0].monomers

    @classmethod
    def from_lists(
        cls,
        nodes: List[np.ndarray],
        edges: List[np.ndarray],
        monomers: List[Monomer],
    ):
        return cls(
            graphs=[
                PolyGraph(nodes=g, edges=e, monomers=monomers)
                for g, e in zip(nodes, edges)
            ],
        )

    def __iter__(self):
        return iter(self.graphs)

    def __len__(self):
        return len(self.graphs)

    @property
    def masses(self) -> List[float]:
        """
        Returns the mass of the graphs in the ensemble.
        """
        return np.array([g.mass for g in self.graphs])

    def mass_distribution(self, bins=100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the mass distribution of the graphs in the ensemble.
        """
        return np.histogram(
            self.masses,
            bins=bins,
        )
