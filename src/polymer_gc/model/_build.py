
from polymer_gc.dataset import Dataset
from polymer_gc import Monomer
from typing import TypeAlias
from collections.abc import Callable
import numpy as np

MonomerFeaturizer: Callable[[Monomer],np.ndarray]

def build_model_from_dataset(dataset:Dataset,
        monomer_featurizer:MonomerFeaturizer,
):
    pass



