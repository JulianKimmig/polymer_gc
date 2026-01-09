import torch
import numpy as np
from typing import List, Optional, Union
from tqdm import tqdm
from torch_geometric.data import Data
from polymer_gc.data.dataset import Dataset, DatasetItem


def make_graph_input(
    dataset: Dataset, items: Optional[List[Union[DatasetItem, int]]] = None
):
    data = dataset.load_entries_data(items)
    strucid_to_idx = {val: idx for idx, val in enumerate(data["structure_ids"])}
    vec_strucid_to_idx = np.vectorize(strucid_to_idx.get)

    # Target is now a single value: Tg
    target_names = dataset.config.targets  # Should be 'Tg'
    targets_arrays = [data["targets"][target_name] for target_name in target_names]
    print(f"Target variable(s): '{','.join(target_names)}'")

    all_graph_data = []
    # The 'graphs' key contains info to reconstruct graph objects
    for g_info in tqdm(data["graphs"], desc="Creating PyG Data objects"):
        # Node features are the pre-trained embeddings for the polymer structures
        # Note: In this dataset, there's only one "monomer" type per graph (the whole polymer)
        structure_idx = vec_strucid_to_idx(g_info["nodes"])
        embeddings = data["all_embeddings"][structure_idx]

        # Edges from the dummy graph used during data creation
        edges = torch.tensor(g_info["edges"], dtype=torch.long).T

        # Target 'y' is the Tg value for the corresponding entry
        target_values = np.concatenate(
            [
                np.array([targets_array[g_info["entry_pos"]]]).flatten()
                for targets_array in targets_arrays
            ]
        )

        graph_data_obj = Data(
            x=torch.tensor(embeddings, dtype=torch.float32),
            edge_index=edges,
            # Ensure y is a tensor with shape [1, 1] for consistency
            y=torch.tensor(target_values, dtype=torch.float32).unsqueeze(0),
            entry_pos=g_info["entry_pos"],  # Crucial for data splitting
            entry_id=g_info["entry_id"],
            mass_distribution=torch.tensor(
                data["sec"][g_info["sec_id"]], dtype=torch.float32
            ).unsqueeze(0)
            if g_info["sec_id"] is not None
            else None,
        )
        all_graph_data.append(graph_data_obj)

    return all_graph_data
