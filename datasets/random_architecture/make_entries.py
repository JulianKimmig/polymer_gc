from model import RandomArchitectureDataset, RandomArchitectureEntry
from polymer_gc.core.dataset import (
    AnyPolymerArchitecture,
    AnyPolymerSequence,
    Homopolymer,
    Copolymer,
    GradientCopolymer,
)
import numpy as np
from typing import get_args
import pandas as pd
from polymer_gc.graph.linear import norm_sigmoid
from tqdm import tqdm
from pathlib import Path
import pickle
import os
import json

json_path = Path(__file__).parent / "RandomArchitecture.json"
if json_path.exists():
    exit()

N_POLYMERS = 100
min_mass, max_mass = 10_000, 1_000_000
min_pdi, max_pdi = 1.1, 4.0
MAX_MONOMERS = 3
GRADIENT_RESOLUTION = 50
rng = np.random.RandomState(42)

lin_monomers = pd.read_csv("LAMALAB_CURATED_Tg_structured_polymerclass.csv")["PSMILES"]

log_masses = np.log(min_mass), np.log(max_mass)
n = (
    len(get_args(AnyPolymerSequence))
    * len(get_args(AnyPolymerArchitecture))
    * N_POLYMERS
)
masses = (
    rng.rand(n) * (log_masses[1] - log_masses[0]) + log_masses[0]
)  # min,max=log(min_mass),log(max_mass)
masses = np.exp(masses)  # min,max=min_mass,max_mass
gradients = norm_sigmoid(
    np.linspace(0, 1, GRADIENT_RESOLUTION)[:, np.newaxis],
    rng.rand(n * MAX_MONOMERS) * 40 - 20,
)
r = 2
pdi = rng.rand(n) * r  # min,max=0,r
pdi = pdi + 1  # min,max=1,r+1
pdi = 1 / pdi  # min,max=1/(r+1),1
pdi -= 1 / (r + 1)  # min,max=0,1-1/(r+1)
pdi = pdi / (1 - 1 / (r + 1))  # min,max=0,1

pdi *= max_pdi - min_pdi  # min,max=0, max_pdi-min_pdi
pdi += min_pdi  # min,max=min_pdi, max_pdi

k = 0
entries = []


with tqdm(total=n) as pbar:
    for seq in get_args(AnyPolymerSequence):
        # get default value of seq.sequence_type
        sequence = seq.model_fields["sequence_type"].default
        for arch in get_args(AnyPolymerArchitecture):
            for N in range(N_POLYMERS):
                if seq is Homopolymer:
                    n_monos = 1
                else:
                    n_monos = rng.randint(2, MAX_MONOMERS)

                mn = masses[k]
                mw = mn * pdi[k]
                monomers = [
                    {"smiles": lin_monomers[s]}
                    for s in rng.choice(len(lin_monomers), n_monos, replace=False)
                ]
                seq_data = {}
                if issubclass(seq, Copolymer):
                    seq_data["ratios"] = rng.random(n_monos)
                if issubclass(seq, GradientCopolymer):
                    seq_data["gradient"] = [
                        gradients[:, k * (1 + i)] for i in range(n_monos)
                    ]

                data = dict(
                    mn=mn,
                    mw=mw,
                    architecture=arch(),
                    sequence=seq(**seq_data),
                    monomers=monomers,
                )

                entry = RandomArchitectureEntry(**data)
                # entry.sec = make_sec(
                #     mn=entry.mn,
                #     mw=entry.mw,

                # )
                entries.append(entry)

                k += 1
                pbar.update(1)


dataset = RandomArchitectureDataset(name="RandomArchitecture", items=entries)

with open("RandomArchitecture.pkl", "wb") as f:
    pickle.dump(dataset, f)


json_path.write_text(dataset.model_dump_json(indent=2))
