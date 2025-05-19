from typing import List, Optional, Union
from collections.abc import Callable
import numpy as np
from .datamodel import PolyGraphEnsemble, Monomer


def make_unique_monomers(ex_monomers, created_nodes_list):
    # Consolidate monomer list and update node indices if monomers are used multiple times.
    # This ensures PolyGraphEnsemble uses a unique list of Monomer objects.

    final_monomer_objects_map = {}  # Monomer object -> new unique index
    for m_obj in (
        ex_monomers
    ):  # ex_monomers contains all start, main, end monomers as they were added
        if m_obj not in final_monomer_objects_map:
            final_monomer_objects_map[m_obj] = len(final_monomer_objects_map)

    unique_monomer_list_final = list(final_monomer_objects_map.keys())

    remapped_nodes_list = []
    if not ex_monomers:  # Should not happen if we reached here, but defensive.
        remapped_nodes_list = created_nodes_list  # all should be empty lists of nodes
    else:
        # Create a mapping array from old ex_monomers index to new unique_monomer_list_final index
        old_idx_to_new_idx_map = np.array(
            [final_monomer_objects_map[m_obj] for m_obj in ex_monomers], dtype=int
        )

        for g_nodes_old_indices in created_nodes_list:
            if len(g_nodes_old_indices) > 0:
                remapped_nodes_list.append(old_idx_to_new_idx_map[g_nodes_old_indices])
            else:
                remapped_nodes_list.append(
                    np.array([], dtype=int)
                )  # Keep empty node arrays empty
    # --- End of Monomer Unification Block ---

    return remapped_nodes_list, unique_monomer_list_final


def make_linear_polymer_graphs(
    M: Union[float, List[float], np.ndarray],
    monomers: List[Monomer],
    n: int = 1,
    rel_content: Optional[List[List[float]]] = None,
    seed: Optional[int] = None,
    startgroup: Optional[Monomer] = None,
    endgroup: Optional[Monomer] = None,
) -> PolyGraphEnsemble:
    """
    Generate random linear polymer graphs with approximate target mass(es).

    If M is a list or an np.array, one polymer graph is generated for each mass in the list.
    The `n` parameter is ignored in this case.
    If M is a float, `n` polymer graphs are generated, each targeting mass M.

    Polymer graphs are sequences of monomer indices, sampled according to
    optional local monomer distribution profiles across the chain length.
    The generated graphs are mass-adjusted to fit their respective target mass
    by removing monomers randomly until the total mass is within the allowed
    range.
    The mass of the starting and end groups can be specified. If not
    provided, no start or end group is added.

    Args:
        M (Union[float, List[float], np.ndarray]):
            Target total mass per polymer. If a list or np.array, generates one polymer
            for each mass, and `n` is ignored.
        monomers (List[Monomer]):
            List of Monomer objects representing the repeating units.
        n (int, optional):
            Number of polymer chains to generate if M is a float.
            Defaults to 1.
        rel_content (Optional[List[List[float]]], optional):
            Relative content profiles per monomer. Each inner list defines
            the relative probability of a monomer along the normalized chain length.
            If None, a uniform distribution is assumed for all monomers.
        seed (Optional[int], optional):
            Random seed for reproducibility.
        startgroup (Optional[Monomer], optional):
            Monomer object for the starting group. If None, no start group.
        endgroup (Optional[Monomer], optional):
            Monomer object for the end group. If None, no end group.

    Returns:
        PolyGraphEnsemble:
            An ensemble of generated polymer graphs.

    Raises:
        ValueError: If target mass M (or any in list M) is non-positive
                    after accounting for start/end groups, or if M list is empty.
        TypeError: If M is not a float or list of floats.
    """
    rng = np.random.default_rng(seed)

    monomer_actual_masses = [m.mass for m in monomers]
    startgroup_mass = startgroup.mass if startgroup is not None else 0.0
    endgroup_mass = endgroup.mass if endgroup is not None else 0.0

    # Determine num_graphs and target_masses_arr based on M
    if isinstance(M, list):
        if not M:
            raise ValueError("Target mass list M cannot be empty.")
        target_masses_input = np.array(M, dtype=float)
        num_graphs = len(target_masses_input)
    elif isinstance(M, np.ndarray):
        if M.size == 0:
            raise ValueError("Target mass array M cannot be empty.")
        target_masses_input = np.array(M, dtype=float).flatten()
        num_graphs = len(target_masses_input)
    elif isinstance(M, (float, int)):
        target_masses_input = np.array([float(M)] * n, dtype=float)
        num_graphs = n
    else:
        raise TypeError("M must be a float or a list of floats or numpy array.")

    # Adjust target masses for start/end groups
    adjusted_target_masses = target_masses_input - startgroup_mass - endgroup_mass
    if np.any(adjusted_target_masses <= 0):
        raise ValueError(
            "Target mass(es) must be greater than 0 after accounting for start/end groups."
        )

    # Prepare monomer list for graph construction (including potential start/end groups)
    ex_monomers = []
    if startgroup is not None:
        ex_monomers.append(startgroup)
    ex_monomers.extend(monomers)
    if endgroup is not None:
        ex_monomers.append(endgroup)

    # Mass array for calculations (with dummy 0 for 'empty' monomer)
    # This mass_ru_calc array corresponds to indices 1 to n_ru for actual monomers
    mass_ru_calc = np.array([0.0] + monomer_actual_masses)
    n_ru = len(monomer_actual_masses)  # Number of actual repeating units
    ru_indices = np.arange(1, n_ru + 1)  # Indices for sampling: 1 to n_ru

    if not monomer_actual_masses:  # Handle case of empty monomers list
        if np.any(adjusted_target_masses > 0):
            raise ValueError(
                "Cannot generate polymers with positive mass if no monomers are provided."
            )
        # If all target masses are zero (after start/end group deduction) and no monomers,
        # create empty main chains. Start/end groups might still form a "graph".
        nodes_list = [np.array([], dtype=int) for _ in range(num_graphs)]
        # Add start/end groups if present
        current_idx = 0
        if startgroup is not None:
            nodes_list = [np.insert(g, 0, current_idx) for g in nodes_list]
            current_idx += 1

        # For end group, its index depends on whether start group was added
        # This simplified logic assumes only start/end groups, no main chain
        if endgroup is not None:
            nodes_list = [np.append(g, current_idx) for g in nodes_list]

        edges_list = [
            np.stack([np.arange(len(g) - 1), np.arange(1, len(g))]).T
            if len(g) > 1
            else np.empty((0, 2), dtype=int)
            for g in nodes_list
        ]
        return PolyGraphEnsemble.from_lists(
            nodes=nodes_list,
            edges=edges_list,
            monomers=ex_monomers,
        )

    min_actual_monomer_mass = np.min(
        np.array(monomer_actual_masses)[np.array(monomer_actual_masses) > 0]
    )
    max_actual_monomer_mass = np.max(monomer_actual_masses)

    # Determine max_length for initial chain generation based on the largest adjusted target mass
    max_chain_mass = np.max(adjusted_target_masses)
    max_length = int(np.ceil(max_chain_mass / min_actual_monomer_mass))
    if (
        max_length == 0 and max_chain_mass > 0
    ):  # Ensure max_length is at least 1 if target mass > 0
        max_length = 1

    # Interpolate local relative content across chain length
    if rel_content is None:
        # Uniform distribution for all monomers if not specified
        _rel_content_processed = (np.ones(n_ru) / n_ru).reshape(n_ru, 1)
    else:
        _rel_content_processed = np.array(rel_content)
        if _rel_content_processed.shape[0] != n_ru:
            raise ValueError(
                f"rel_content must have {n_ru} profiles, one for each monomer."
            )

    local_content = np.zeros((max_length, n_ru))
    if max_length > 0:  # Proceed with interpolation only if max_length > 0
        # Ensure interp expects x to be sorted and unique, which np.arange and np.linspace provide
        xp_local_content = np.linspace(
            0, max_length - 1 if max_length > 1 else 0, _rel_content_processed.shape[1]
        )
        x_interp_points = np.arange(max_length)
        for i in range(n_ru):
            local_content[:, i] = np.interp(
                x_interp_points, xp_local_content, _rel_content_processed[i, :]
            )

    # Generate initial graph nodes_array
    # nodes_array stores indices from 1 to n_ru (referring to mass_ru_calc)
    nodes_array = np.zeros((num_graphs, max_length), dtype=int)
    if max_length > 0:  # Avoid operations if no length (e.g. target mass is zero)
        for i in range(max_length):
            p = local_content[i].copy()  # Ensure p is a 1D array
            # Handle cases where sum of p is zero to avoid division by zero
            # This can happen if rel_content profiles lead to zero probability at some point
            if p.sum() <= 0:
                # Fallback to uniform probability if sum is zero or negative
                # This should ideally not happen with valid rel_content.
                p = np.ones_like(p) / len(p) if len(p) > 0 else np.array([])
            else:
                p /= p.sum()

            if len(p) > 0:  # Check if there are probabilities to choose from
                nodes_array[:, i] = rng.choice(ru_indices, size=num_graphs, p=p)
            # if len(p) == 0 and n_ru > 0, this implies an issue with rel_content or monomer setup

    current_graph_masses = mass_ru_calc[nodes_array].sum(axis=1)
    # max_allowed_masses_arr is specific to each graph's target
    max_allowed_masses_arr = adjusted_target_masses + min_actual_monomer_mass / 2

    # Mass adjustment: remove random monomers until mass fits for each graph
    too_heavy = current_graph_masses > max_allowed_masses_arr

    # Iteratively remove units from graphs that are too heavy
    # This loop needs to handle cases where max_length or n_ru could be zero
    # max_actual_monomer_mass could be 0 if all monomer masses are 0.
    # Added check for max_actual_monomer_mass > 0 before division.
    while (
        np.any(too_heavy)
        and max_length > 0
        and n_ru > 0
        and max_actual_monomer_mass > 0
    ):
        # Calculate how many units to remove for each overweight graph
        overweight_amount = (
            current_graph_masses[too_heavy] - max_allowed_masses_arr[too_heavy]
        )
        # Ceiling division to find minimum number of max_mass_ru units to remove
        removable_units_count = np.ceil(
            overweight_amount / max_actual_monomer_mass
        ).astype(int)

        # Get indices of rows (graphs) that are too heavy
        heavy_graph_indices = np.where(too_heavy)[0]

        for graph_idx, num_to_remove in zip(heavy_graph_indices, removable_units_count):
            # Find positions of actual monomers (non-zero entries) in the current graph
            nonzero_positions = np.flatnonzero(nodes_array[graph_idx, :])

            if len(nonzero_positions) == 0:  # No monomers to remove
                continue

            # Determine how many monomers to actually remove (cannot remove more than available)
            actual_remove_count = min(num_to_remove, len(nonzero_positions))

            # Choose random positions among the non-zero entries to set to 0 (remove monomer)
            pos_to_zero_out = rng.choice(
                nonzero_positions, size=actual_remove_count, replace=False
            )
            nodes_array[graph_idx, pos_to_zero_out] = 0  # Set to 'empty' monomer index

        # Recalculate masses and update which graphs are too heavy
        current_graph_masses = mass_ru_calc[nodes_array].sum(axis=1)
        too_heavy = current_graph_masses > max_allowed_masses_arr

    # Cleanup: remove empty entries (0s) and reindex monomers to start from 0 (relative to `monomers` list)
    # `g[g != 0]` removes placeholders; `- 1` adjusts from 1-based to 0-based index for `monomers` list
    created_nodes_list = [(g[g != 0] - 1) for g in nodes_array]

    # Add start and end groups, adjusting indices accordingly
    # The indices in final_nodes_list now refer to the `monomers` list (0 to n_ru-1).
    # If startgroup exists, it's at ex_monomers[0]. Main monomers shift by +1.
    # If endgroup exists, its index depends on presence of startgroup.

    # Index for end group in ex_monomers list
    end_group_idx_in_ex_monomers = n_ru  # Base index if no startgroup
    if startgroup is not None:
        # Nodes from final_nodes_list need to be shifted by 1 to make space for startgroup at index 0
        # Startgroup is index 0 in ex_monomers
        created_nodes_list = [np.insert(g + 1, 0, 0) for g in created_nodes_list]
        end_group_idx_in_ex_monomers += (
            1
        )  # Shift endgroup index if startgroup is present

    if endgroup is not None:
        # Append endgroup index to each node list
        created_nodes_list = [
            np.append(g, end_group_idx_in_ex_monomers) for g in created_nodes_list
        ]

    # Create edges for linear polymers
    # An edge exists between node i and node i+1
    edges_list = [
        np.stack([np.arange(len(g_nodes) - 1), np.arange(1, len(g_nodes))]).T
        if len(g_nodes) > 1
        else np.empty((0, 2), dtype=int)
        for g_nodes in created_nodes_list
    ]

    remapped_nodes_list, unique_monomer_list_final = make_unique_monomers(
        ex_monomers=ex_monomers,
        created_nodes_list=created_nodes_list,
    )

    return PolyGraphEnsemble.from_lists(
        nodes=remapped_nodes_list,
        edges=edges_list,
        monomers=unique_monomer_list_final,
    )


def sigmoid(x, a=1, b=0.5):
    return 1 / (1 + np.exp(-a * (x - b)))


def norm_sigmoid(x, a=10):
    return sigmoid(x, a=a, b=0.5)


def default_gradient_sigmoid(x):
    return sigmoid(x, a=10, b=0.5)


def make_linear_gradient_polymer(
    M: Union[float, List[float], np.ndarray],
    monomers: List[Monomer],
    n: int = 1,
    gadient_functions: List[Callable[[np.ndarray], np.ndarray]] = None,
    seed: Optional[int] = None,
    startgroup: Optional[Monomer] = None,
    endgroup: Optional[Monomer] = None,
) -> PolyGraphEnsemble:
    x = np.linspace(0, 1, 1000)
    if gadient_functions is None:
        y = [default_gradient_sigmoid(x) for _ in range(len(monomers))]
        # reverse the first half of the list
        second_half = int(np.ceil(3 / 2))
        y[:second_half] = [y[i][::-1] for i in range(second_half)]
    else:
        if len(gadient_functions) != len(monomers):
            raise ValueError(
                "The number of gradient functions must match the number of mass_ru values."
            )
        y = [f(x) for f in gadient_functions]

    return make_linear_polymer_graphs(
        M=M,
        monomers=monomers,
        n=n,
        rel_content=y,
        seed=seed,
        startgroup=startgroup,
        endgroup=endgroup,
    )


def merge_linear_polymers_to_block(
    polymer_graph_ensembles: List[PolyGraphEnsemble],
    remove_overlapping_endgroups: bool = True,
) -> PolyGraphEnsemble:
    """
    Merges multiple linear polymer graph ensembles into a single ensemble of block copolymers.
    It takes the i-th graph from each input ensemble and merges them to form the i-th
    block copolymer graph in the output ensemble.

    Args:
        polymer_graph_ensembles (List[PolyGraphEnsemble]): A list of PolyGraphEnsemble objects.
            Each ensemble must contain the same number of graphs.
        remove_overlapping_endgroups (bool, optional): If True, attempts to remove
            potential endgroup monomers at the junction of blocks based on a heuristic
            (if the monomer value appears <= 2 times in its segment). Defaults to True.

    Returns:
        PolyGraphEnsemble: An ensemble containing the merged block copolymer graphs.

    Raises:
        ValueError: If input list is empty, ensembles have different numbers of graphs,
                    or other inconsistencies are found.
    """
    if not polymer_graph_ensembles:
        return PolyGraphEnsemble.from_lists(nodes=[], edges=[], monomers=[])

    if len(polymer_graph_ensembles) == 1:
        # If only one ensemble is provided, return it directly
        return polymer_graph_ensembles[0]

    # 1. Create a comprehensive list of all unique monomer objects involved across all ensembles.
    # These will be the `ex_monomers` for the `make_unique_monomers` call at the end.
    # The indices used *during merging* will refer to this `preliminary_all_monomers_list`.
    preliminary_all_monomers_map = {}
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

    num_graphs_to_create = num_graphs_per_ensemble[0]
    if num_graphs_to_create == 0:  # All ensembles are empty of graphs
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

        accumulated_nodes_temp_indices = np.array([], dtype=int)
        accumulated_edges = np.empty((0, 2), dtype=int)

        is_first_segment_in_merge = True

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

            # --- Overlap removal logic (if not the very first segment being added) ---
            if not is_first_segment_in_merge and remove_overlapping_endgroups:
                # A. Check end of ACCUMULATED chain (if it exists)
                if accumulated_nodes_temp_indices.size > 0:
                    val_last_node_accumulated = accumulated_nodes_temp_indices[-1]
                    if (
                        np.sum(
                            accumulated_nodes_temp_indices == val_last_node_accumulated
                        )
                        <= 2
                    ):
                        idx_of_last_node_accumulated = (
                            accumulated_nodes_temp_indices.shape[0] - 1
                        )
                        accumulated_nodes_temp_indices = accumulated_nodes_temp_indices[
                            :-1
                        ]
                        if (
                            accumulated_edges.size > 0
                        ):  # Ensure edges exist before filtering
                            accumulated_edges = accumulated_edges[
                                (
                                    accumulated_edges[:, 0]
                                    != idx_of_last_node_accumulated
                                )
                                & (
                                    accumulated_edges[:, 1]
                                    != idx_of_last_node_accumulated
                                )
                            ]

                # B. Check start of CURRENT segment (if it exists)
                if current_segment_nodes_temp_indices.size > 0:
                    val_first_node_current_segment = current_segment_nodes_temp_indices[
                        0
                    ]
                    if (
                        np.sum(
                            current_segment_nodes_temp_indices
                            == val_first_node_current_segment
                        )
                        <= 2
                    ):
                        current_segment_nodes_temp_indices = (
                            current_segment_nodes_temp_indices[1:]
                        )
                        if current_segment_edges.size > 0:  # Ensure edges exist
                            # Filter out edges connected to original node 0 and decrement others
                            current_segment_edges = current_segment_edges[
                                (current_segment_edges[:, 0] != 0)
                                & (current_segment_edges[:, 1] != 0)
                            ]
                            current_segment_edges = current_segment_edges - 1
                            # Ensure valid indices after decrementing
                            if (
                                current_segment_nodes_temp_indices.size > 0
                                and current_segment_edges.size > 0
                            ):  # Guard against empty nodes/edges
                                current_segment_edges = current_segment_edges[
                                    np.all(current_segment_edges >= 0, axis=1)
                                    & (
                                        current_segment_edges[:, 0]
                                        < current_segment_nodes_temp_indices.shape[0]
                                    )
                                    & (
                                        current_segment_edges[:, 1]
                                        < current_segment_nodes_temp_indices.shape[0]
                                    )
                                ]
                            elif (
                                current_segment_edges.size > 0
                            ):  # Edges exist but nodes became empty
                                current_segment_edges = np.empty((0, 2), dtype=int)

            # --- Concatenate nodes and edges ---
            node_index_offset = accumulated_nodes_temp_indices.shape[0]

            # Add connecting edge if this isn't the first segment and both parts have nodes
            new_connecting_edge = np.empty((0, 2), dtype=int)
            if (
                not is_first_segment_in_merge
                and accumulated_nodes_temp_indices.size > 0
                and current_segment_nodes_temp_indices.size > 0
            ):
                new_connecting_edge = np.array(
                    [[node_index_offset - 1, node_index_offset]], dtype=int
                )

            accumulated_nodes_temp_indices = np.concatenate(
                (accumulated_nodes_temp_indices, current_segment_nodes_temp_indices)
            )

            # Concatenate edges: previous, then connecting, then new segment's edges (offset)
            accumulated_edges = np.concatenate((accumulated_edges, new_connecting_edge))
            if current_segment_edges.size > 0:
                accumulated_edges = np.concatenate(
                    (accumulated_edges, current_segment_edges + node_index_offset)
                )

            is_first_segment_in_merge = False

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
