# from model import RandomArchitectureDataset, RandomArchitectureEntry
from polymer_gc.core.dataset import AnyPolymerArchitecture, AnyPolymerSequence
import numpy as np
from typing import Union, get_origin, get_args

N_POLYMERS = 100
rng = np.random.RandomState(42)

entries = []

for seq in get_args(AnyPolymerSequence):
    # get default value of seq.sequence_type
    print(seq.fields)
