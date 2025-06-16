from pydantic import Field
from typing import Literal, TypeAlias
import polymer_gc


class RandomArchitectureEntry(polymer_gc.dataset.DataSetEntry):
    hot_encoded_architecture: list[float] = Field(
        description="Hot-encoded architecture vector of length 64.",
        default_factory=list,
    )
    hot_encoded_sequence: list[float] = Field(
        description="Hot-encoded sequence vector of length 64.", default_factory=list
    )


RandomArchitectureDataset: TypeAlias = polymer_gc.dataset.Dataset[
    RandomArchitectureEntry
]
