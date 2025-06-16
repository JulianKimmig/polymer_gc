from pathlib import Path
from polymer_gc.core.dataset import (
    AnyPolymerArchitecture,
    AnyPolymerSequence,
)
import numpy as np
from typing import get_args
import json
from model import RandomArchitectureDataset
from tqdm import tqdm


architectures = [seq for seq in get_args(AnyPolymerArchitecture)]
sequences = [seq for seq in get_args(AnyPolymerSequence)]

with open("RandomArchitecture.json", "r") as f:
    pg_dataset = RandomArchitectureDataset(**json.load(f))


try:
    for item in tqdm(pg_dataset.items, total=len(pg_dataset.items)):
        if (
            len(item.hot_encoded_architecture) == 0
            or np.sum(item.hot_encoded_architecture) == 0
            or len(item.hot_encoded_sequence) == 0
            or np.sum(item.hot_encoded_sequence) == 0
        ):
            item.hot_encoded_architecture = np.zeros(
                len(architectures), dtype=float
            ).tolist()
            item.hot_encoded_sequence = np.zeros(len(sequences), dtype=float).tolist()

            item.hot_encoded_architecture[
                architectures.index(item.architecture.__class__)
            ] = 1.0
            item.hot_encoded_sequence[sequences.index(item.sequence.__class__)] = 1.0
finally:
    json_path = Path("RandomArchitecture.json")
    json_path.write_text(pg_dataset.model_dump_json(indent=2))
