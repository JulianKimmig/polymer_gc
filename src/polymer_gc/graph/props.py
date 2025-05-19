from .datamodel import PolyGraphEnsemble
import numpy as np
from tqdm import tqdm


def calculate_local_monomer_distribution(
    graphs: PolyGraphEnsemble, n=1000, window_size: int = 10
):
    n_monomers = len(graphs.monomers)
    normalized_graph_distribution = np.zeros((n, n_monomers))
    global_space = np.linspace(0, 1, n)

    if len(graphs) == 0:
        raise ValueError("No graphs provided for distribution calculation.")

    for i, g in tqdm(
        enumerate(graphs),
        desc="Calculating local monomer distribution",
        total=len(graphs),
        unit="graph",
    ):
        g = np.array(g.nodes)
        gl = len(g)
        graph_distribution = np.zeros((gl, n_monomers))
        graph_space = np.linspace(0, 1, gl)
        for j in range(gl):
            start = max(0, j - window_size // 2)
            end = min(gl, j + window_size // 2 + 1)
            f = 1 / (end - start)
            window = g[start:end]
            for monomer_idx in range(n_monomers):
                graph_distribution[j, monomer_idx] += (
                    np.sum(window == monomer_idx)
                ) * f

        for monomer_idx in range(n_monomers):
            normalized_graph_distribution[:, monomer_idx] += np.interp(
                global_space, graph_space, graph_distribution[:, monomer_idx]
            )

    normalized_graph_distribution /= len(graphs)  # Normalize by real counts
    return normalized_graph_distribution
