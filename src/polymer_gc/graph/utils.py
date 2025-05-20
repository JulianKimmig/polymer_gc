from typing import List, Tuple
import numpy as np
from ..core.monomer import Monomer


def make_unique_monomers(
    ex_monomers: List[Monomer],  # List of Monomer objects
    created_nodes_list: List[np.ndarray],  # List of np.ndarray of int (old indices)
) -> Tuple[List[np.ndarray], List[Monomer]]:
    """
    Consolidates a list of monomer objects and remaps node indices.

    Args:
        ex_monomers: The original list of monomer objects that node indices in
                              created_nodes_list refer to.
        created_nodes_list: A list of numpy arrays, where each array contains node indices
                            referring to positions in ex_monomers.

    Returns:
        A tuple containing:
        - remapped_nodes_list: List of numpy arrays with node indices remapped to refer
                               to unique_monomer_list_final.
        - unique_monomer_list_final: A new list containing only the unique monomer objects
                                     that are actually used in created_nodes_list.
    """

    # Handle cases where ex_monomers might be empty
    if not ex_monomers:
        # If there were no original monomers, all node lists must be empty.
        # And the final monomer list is empty.
        for g_nodes in created_nodes_list:
            if g_nodes.size > 0:  # Check if array has elements
                raise ValueError(
                    "created_nodes_list contains non-empty node arrays, "
                    "but ex_monomers is empty."
                )
        # All created_nodes_list entries must be empty arrays
        return [np.array([], dtype=int) for _ in created_nodes_list], []

    # Flatten node lists to find all used original indices
    all_nodes_flat_list = []
    if created_nodes_list:  # Ensure created_nodes_list is not None or empty
        for nodes_array in created_nodes_list:
            if nodes_array.size > 0:  # Check if the numpy array itself is not empty
                all_nodes_flat_list.extend(nodes_array.tolist())

    if not all_nodes_flat_list:
        # No monomers are actually used in any graph, or created_nodes_list was empty/contained only empty arrays.
        # Return empty node lists and an empty monomer list.
        return [np.array([], dtype=int) for _ in created_nodes_list], []

    all_nodes_flat_np = np.array(all_nodes_flat_list, dtype=int)

    # Ensure all indices in all_nodes_flat_np are valid for ex_monomers
    # This check is important as max() on empty array raises error.
    # all_nodes_flat_np is guaranteed non-empty here due to the check above.
    max_idx_used = np.max(all_nodes_flat_np)
    if max_idx_used >= len(ex_monomers):
        raise IndexError(
            f"An index ({max_idx_used}) in created_nodes_list is out of bounds "
            f"for ex_monomers (length {len(ex_monomers)})."
        )

    # Determine which original monomer *objects* are actually used and build the final unique list
    final_unique_monomer_objects_map = {}  # Monomer object -> new_final_index
    unique_monomer_list_final = []

    # Iterate through ex_monomers using the unique original indices found in the graphs.
    # This ensures that only used monomers are included and they are unique,
    # preserving the first encountered object instance for duplicates in ex_monomers.
    for original_idx in np.unique(
        all_nodes_flat_np
    ):  # Iterate over unique original indices that are used
        monomer_obj = ex_monomers[original_idx]
        if monomer_obj not in final_unique_monomer_objects_map:
            final_unique_monomer_objects_map[monomer_obj] = len(
                unique_monomer_list_final
            )
            unique_monomer_list_final.append(monomer_obj)

    # Create the mapping array from original_idx (in ex_monomers)
    # to new_final_idx (in unique_monomer_list_final)
    # This array must be of length len(ex_monomers)
    old_original_idx_to_new_final_idx_map = np.full(len(ex_monomers), -1, dtype=int)

    for original_idx in range(len(ex_monomers)):
        monomer_obj = ex_monomers[original_idx]
        if monomer_obj in final_unique_monomer_objects_map:
            # If this monomer object (from this original_idx) is part of the final unique set
            old_original_idx_to_new_final_idx_map[
                original_idx
            ] = final_unique_monomer_objects_map[monomer_obj]
        # else: its entry in old_original_idx_to_new_final_idx_map remains -1.
        # This means ex_monomers[original_idx] was either unused or a duplicate of an
        # object already processed from an earlier original_idx that was used.

    # Remap node lists
    remapped_nodes_list = []
    for g_nodes_old_indices in created_nodes_list:
        if g_nodes_old_indices.size > 0:
            new_nodes = old_original_idx_to_new_final_idx_map[g_nodes_old_indices]

            if np.any(new_nodes == -1):
                # This indicates an internal inconsistency: a node index in g_nodes_old_indices
                # referred to a monomer in ex_monomers that was used (it's in g_nodes_old_indices)
                # but somehow didn't make it into final_unique_monomer_objects_map mapped via its object,
                # or its mapping in old_original_idx_to_new_final_idx_map became -1.
                problematic_old_indices = g_nodes_old_indices[new_nodes == -1]
                # For more detailed debugging:
                # problematic_monomer_objects = [ex_monomers[i] for i in problematic_old_indices]
                raise ValueError(
                    f"Internal inconsistency during node remapping. "
                    f"Old node indices {problematic_old_indices} mapped to -1. "
                    f"This implies they referenced monomers in ex_monomers "
                    f"that were considered used but not found in the final unique monomer map "
                    f"or their mapping resulted in -1. "
                    f"ex_monomers length: {len(ex_monomers)}. "
                    # f"Problematic monomer objects: {[m.name for m in problematic_monomer_objects]}. " # If Monomer has .name
                    f"unique_monomer_list_final length: {len(unique_monomer_list_final)}."
                )
            remapped_nodes_list.append(new_nodes)
        else:
            # Keep empty node arrays empty
            remapped_nodes_list.append(np.array([], dtype=int))

    return remapped_nodes_list, unique_monomer_list_final
