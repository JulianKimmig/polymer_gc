from typing import List, Tuple, Union
import numpy as np
from .datamodel import PolyGraphEnsemble, Monomer
from .utils import make_unique_monomers


def make_star_polymer_from_linear(
    polymer_graph_ensembles: Union[PolyGraphEnsemble, List[PolyGraphEnsemble]],
    center_monomer: Monomer,
    n: int = 2,
):
    if isinstance(polymer_graph_ensembles, list):
        n = len(polymer_graph_ensembles)
    else:
        polymer_graph_ensembles = [polymer_graph_ensembles] * n

    # 1. Create a comprehensive list of all unique monomer objects involved across all ensembles.
    # These will be the `ex_monomers` for the `make_unique_monomers` call at the end.
    # The indices used *during merging* will refer to this `preliminary_all_monomers_list`.
    preliminary_all_monomers_map = {center_monomer: 0}
    for ensemble in polymer_graph_ensembles:
        for monomer_obj in ensemble.monomers:
            if monomer_obj not in preliminary_all_monomers_map:
                preliminary_all_monomers_map[monomer_obj] = len(
                    preliminary_all_monomers_map
                )

    preliminary_all_monomers_list = list(preliminary_all_monomers_map.keys())

    # Check if all ensembles have the same number of graphs for zipping
    num_graphs_per_ensemble = [len(ens) for ens in polymer_graph_ensembles]
    if (
        not num_graphs_per_ensemble
    ):  # Should be caught by "if not polymer_graph_ensembles"
        return PolyGraphEnsemble.from_lists(nodes=[], edges=[], monomers=[])

    if len(set(num_graphs_per_ensemble)) > 1:
        raise ValueError(
            "All input PolyGraphEnsembles must have the same number of graphs to be merged."
        )

    num_graphs_to_create = np.min(num_graphs_per_ensemble)
    if num_graphs_to_create == 0:
        # Pass the unique monomers found, even if no graphs are created
        return PolyGraphEnsemble.from_lists(
            nodes=[], edges=[], monomers=preliminary_all_monomers_list
        )

    final_merged_nodes_list_temp_indices = []
    final_merged_edges_list = []

    for i_graph in range(num_graphs_to_create):
        # `graphs_to_merge_this_iteration` is a list of PolyGraph objects
        graphs_to_merge_this_iteration = [
            ens.graphs[i_graph] for ens in polymer_graph_ensembles
        ]

        accumulated_nodes_temp_indices = np.array([0], dtype=int)
        accumulated_edges = np.empty((0, 2), dtype=int)

        for current_segment_polygraph in graphs_to_merge_this_iteration:
            segment_original_nodes = current_segment_polygraph.nodes.copy()
            segment_original_edges = current_segment_polygraph.edges.copy()
            # Monomers specific to the ensemble this PolyGraph came from
            segment_local_monomers = current_segment_polygraph.monomers

            # Remap segment_original_nodes to use indices from preliminary_all_monomers_list
            # Create a mapping array: local_idx_in_segment_monomers -> idx_in_preliminary_all_monomers_list
            map_local_to_prelim_indices = np.array(
                [preliminary_all_monomers_map[m] for m in segment_local_monomers],
                dtype=int,
            )

            current_segment_nodes_temp_indices = np.array([], dtype=int)
            if segment_original_nodes.size > 0:
                current_segment_nodes_temp_indices = map_local_to_prelim_indices[
                    segment_original_nodes
                ]

            current_segment_edges = (
                segment_original_edges.copy()
            )  # Edges are 0-indexed for this segment

            # --- Concatenate nodes and edges ---
            node_index_offset = accumulated_nodes_temp_indices.shape[0]

            # Add connecting edge if this isn't the first segment and both parts have nodes
            new_connecting_edge = np.empty((0, 2), dtype=int)
            if (
                accumulated_nodes_temp_indices.size > 0
                and current_segment_nodes_temp_indices.size > 0
            ):
                new_connecting_edge = np.array([[0, node_index_offset]], dtype=int)

            accumulated_nodes_temp_indices = np.concatenate(
                (accumulated_nodes_temp_indices, current_segment_nodes_temp_indices)
            )

            # Concatenate edges: previous, then connecting, then new segment's edges (offset)
            accumulated_edges = np.concatenate((accumulated_edges, new_connecting_edge))
            if current_segment_edges.size > 0:
                accumulated_edges = np.concatenate(
                    (accumulated_edges, current_segment_edges + node_index_offset)
                )

        final_merged_nodes_list_temp_indices.append(accumulated_nodes_temp_indices)
        final_merged_edges_list.append(accumulated_edges)

    # Now, use make_unique_monomers to get the final unique monomer list and remapped nodes
    # `preliminary_all_monomers_list` is the `ex_monomers` list that
    # `final_merged_nodes_list_temp_indices` refer to.
    final_remapped_nodes_list, final_unique_monomer_list = make_unique_monomers(
        ex_monomers=preliminary_all_monomers_list,
        created_nodes_list=final_merged_nodes_list_temp_indices,
    )

    return PolyGraphEnsemble.from_lists(
        nodes=final_remapped_nodes_list,
        edges=final_merged_edges_list,
        monomers=final_unique_monomer_list,
    )
