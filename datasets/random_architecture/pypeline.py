


import matplotlib

from polymer_gc.graph.maker import LinearHomopolymer

matplotlib.use("Agg")  # Must be called BEFORE importing pyplot
import numpy as np
from pathlib import Path
import pandas as pd
from polymcsim import (
    visualize_polymer,
)
from polymer_gc.data.dataset import (
    PgDatasetConfig,
    Dataset,
)
from polymer_gc.data.database import SessionManager
from pathlib import Path

N_POLYMERS = 100  # number of polymers per class
GRAPHS_PER_POLYMER = 5 
GRAPHS_PER_SIMULATION = 10
MIN_MASS, MAX_MASS = 10_000, 1_000_000  # mass range
MIN_PDI, MAX_PDI = 1.2, 4.0  # PDI range
PDI_R = 2  # distribution control parameter for PDI generation
VISUALIZE_GRAPHS = False

MAX_SAMPLE_MASS = 50_000
MIN_SAMPLE_MASS = 4_000
MAX_CO_MONOMERS = 3  # maximum number of monomers in a polymer
MAX_ARMS = 5  # maximum number of arms in a star polymer

db_path = Path(__file__).parent / "database.db"
rng = np.random.RandomState(42)
imagedir = Path(__file__).parent / "images"
imagedir.mkdir(exist_ok=True)



# generate a list of monomers from the tg datafile, as this is what we have now
lin_monomers = pd.read_csv(Path(__file__).parent / "LAMALAB_CURATED_Tg_structured_polymerclass.csv")[
    "PSMILES"
].tolist()

pg_dataset_config = PgDatasetConfig(
    embedding="random_64",
    num_bins=100,
    log_start=1,
    log_end=7,
    n_graphs=2,
    targets=["hot_encoded_architecture", "hot_encoded_structure"],
    additional_features=[],
    target_classes={
        "hot_encoded_architecture": ["linear", "star", "cross_linked", "branching"],
        "hot_encoded_structure": [
            "homopolymer",
            "random_copolymer",
            "gradient",
            "block",
        ],
    },
)

from polymer_gc.graph.maker import (
    LinearHomopolymer,
    LinearRandomCopolymer,
    LinearGradient,
    LinearBlock,
    StarGradientCopolymer,
    StarHomopolymer,
    StarRandomCopolymer,
    StarBlock,
    BranchingHomopolymer,
    BranchingRandomCopolymer,
    BranchingGradient,
    BranchingBlock,
    CrossLinkedRandomCopolymer,
    CrossLinkedHomopolymer,
)

with SessionManager(db_path) as session:

    dataset = Dataset.get_or_create(
        name="RandomArchitecture", set_kwargs=dict(config=pg_dataset_config)
    )

  

    common_kwargs = dict(
        graphs_per_simulation=GRAPHS_PER_SIMULATION,
        allowed_node_deviation=1.15,
        max_sample_mass=MAX_SAMPLE_MASS,
        min_sample_mass=MIN_SAMPLE_MASS,
        min_mass=MIN_MASS,
        max_mass=MAX_MASS,
        min_pdi=MIN_PDI,
        max_pdi=MAX_PDI,
        pdi_r=PDI_R,
        dataset=dataset,
    # gradient mixin    
    lin_monomers=lin_monomers,
    max_monomers=MAX_CO_MONOMERS,
     )

    common_kwargs_call = dict(graphs_per_polymer=GRAPHS_PER_POLYMER,n_polymers=N_POLYMERS,visualize_graphs=VISUALIZE_GRAPHS,image_dir=imagedir,
    
    )



    rints = rng.randint(0, 1000000, 10000).tolist()
    entries = []
    ge = None
    try:
        for i in range(1):
            try:
                entries += LinearHomopolymer(rng=rints.pop(),**common_kwargs)(**common_kwargs_call)
            except Exception as e:
                raise e
                print(e)
                ge = e
                pass
            try:
                entries += LinearRandomCopolymer(rng=rints.pop(),**common_kwargs)(**common_kwargs_call)
            except Exception as e:
                raise e
                print(e)
                ge = e
                pass
            try:
                entries += LinearGradient(rng=rints.pop(),**common_kwargs)(**common_kwargs_call)
            except Exception as e:
                raise e
                print(e)
                ge = e
                pass
            try:
                entries += LinearBlock(rng=rints.pop(),**common_kwargs)(**common_kwargs_call)
            except Exception as e:
                raise e
                print(e)
                ge = e
                pass
            try:
                entries += StarGradientCopolymer(rng=rints.pop(),**common_kwargs,max_arms=MAX_ARMS)(**common_kwargs_call)
            except Exception as e:
                raise e
                print(e)
                ge = e
                pass
            try:
                entries += StarHomopolymer(rng=rints.pop(),**common_kwargs,max_arms=MAX_ARMS)(**common_kwargs_call)
            except Exception as e:
                raise e
                print(e)
                ge = e
                pass
            try:
                entries += StarRandomCopolymer(rng=rints.pop(),**common_kwargs,max_arms=MAX_ARMS)(**common_kwargs_call)
            except Exception as e:
                raise e
                print(e)
                ge = e
                pass
            try:
                entries += StarBlock(rng=rints.pop(),**common_kwargs,max_arms=MAX_ARMS)(**common_kwargs_call)
            except Exception as e:
                raise e
                print(e)
                ge = e
                pass

            try:
                entries += BranchingHomopolymer(rng=rints.pop(),**common_kwargs)(**common_kwargs_call)
            except Exception as e:
                raise e
                print(e)
                ge = e
                pass
            try:
                entries += BranchingRandomCopolymer(rng=rints.pop(),**common_kwargs)(**common_kwargs_call)
            except Exception as e:
                raise e
                print(e)
                ge = e
                pass
            try:
                entries += BranchingGradient(rng=rints.pop(),**common_kwargs)(**common_kwargs_call)
            except Exception as e:
                raise e
                print(e)
                ge = e
                pass
            try:
                entries += BranchingBlock(rng=rints.pop(),**common_kwargs)(**common_kwargs_call)
            except Exception as e:
                raise e
                print(e)
                ge = e
                pass

            try:
                entries += CrossLinkedRandomCopolymer(rng=rints.pop(),**common_kwargs)(**common_kwargs_call)
            except Exception as e:
                raise e
                print(e)
                ge = e
                pass
                # try:
                #     entries += CrossLinkedGradient(rng=rints.pop())(dataset)
                # except Exception as e:
                #     raise e
                print(e)
            #     ge=e
            #     pass

            try:
                entries += CrossLinkedHomopolymer(rng=rints.pop(),**common_kwargs)(**common_kwargs_call)
            except Exception as e:
                raise e
                print(e)
                ge = e
                pass

        if ge is not None:
            raise ge
    except Exception as e:
        raise e
