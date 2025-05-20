from typing import List, Tuple
import numpy as np
from .datamodel import PolyGraphEnsemble


def make_ring_polymer_from_linear(
    polymer_graph_ensemble: PolyGraphEnsemble,
):
    nodes = [g.nodes for g in polymer_graph_ensemble.graphs]
    edges: List[Tuple[int, int]] = [
        h.edges.tolist() for h in polymer_graph_ensemble.graphs
    ]
    for i, edgelist in enumerate(edges):
        edgelist.append([len(nodes[i]) - 1, 0])
    edges = [np.array(e) for e in edges]
    # Create a new list of nodes and edges for the ring polymer
    return PolyGraphEnsemble.from_lists(
        nodes=nodes,
        edges=edges,
        monomers=polymer_graph_ensemble.monomers,
    )
