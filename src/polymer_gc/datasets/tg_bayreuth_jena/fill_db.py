from typing import Dict, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx

# Import the necessary components from your polymer_gc library
from polymer_gc.data.dataset import (
    PgDatasetConfig,
    Dataset,
    DatasetItem,
    GraphDatasetItemLink,
)
from polymer_gc.data.basemodels.structures import SQLStructureModel
from polymer_gc.data.graphs import ReusableGraphEntry
from polymer_gc.data.database import SessionManager
from polymer_gc.sec import SimSEC
from polymer_gc.data.sec import SECEntry
from polymer_gc.sec import make_sec
from polymcsim import (
    MonomerDef,
    SimulationInput,
    generate_polymers_with_mass,
    ReactionSchema,
    VarParam,
    Simulation,
    SimParams,
    visualize_polymer,
    SiteDef,
)
# --- Configuration ---

# 1. Define file paths and names
CSV_PATH = Path(__file__).parent / "tg_data.csv"  # The path to your data
FLORY_FOX_PARAMS_CSV_PATH = Path(__file__).parent / "flory_fox_params_ds.csv"


# 3. Define the dataset configuration
#    This tells the database how to interpret the data we are about to add.
default_pg_dataset_config = PgDatasetConfig(
    targets=["Tg"],
)


def flory_fox(Mn, Tg_inf, K):
    """Calculates Tg based on the Flory-Fox equation."""
    # Ensure Mn is treated as a numpy array for vectorization
    Mn = np.asarray(Mn)
    return Tg_inf - K / Mn



def populate(
        db_path:Path,
        dataset_name:str=None,
        graphs_per_entry:int=20,
        min_monos:int=3,
        max_monos:int=100,
        pg_dataset_config:Optional[PgDatasetConfig]=None,
        seed:int=42,
        flory_fox=True,
):
    pg_dataset_config = pg_dataset_config or default_pg_dataset_config
    pg_dataset_config.targets = default_pg_dataset_config.targets
    pg_dataset_config.target_classes = None
    
    if not dataset_name:
        if flory_fox:
            dataset_name = "tg_bayreuth_jena"
        else:
            dataset_name = "tg_bayreuth_jena_no_flory_fox"

    try:
        df = pd.read_csv(CSV_PATH)
        flory_fox_params_ds = pd.read_csv(FLORY_FOX_PARAMS_CSV_PATH, index_col=0)

        if flory_fox:
            for smiles, data in flory_fox_params_ds.iterrows():
                if smiles not in df["canonicalized_PSMILES_rep_u1"].values:
                    continue
                entrymask = (df["canonicalized_PSMILES_rep_u1"] == smiles) & df[
                    "canonicalized_PSMILES_rep_u2"
                ].isna()
                min_tm = df[entrymask]["Mn"].min()
                max_tm = df[entrymask]["Mn"].max()
                pdi_mean = (df[entrymask]["Mw"] / df[entrymask]["Mn"]).mean()
                sampled_mns = np.random.RandomState(seed).uniform(np.log(min_tm), np.log(max_tm), 20)
                sampled_mns = np.exp(sampled_mns)
                sampled_tgs = flory_fox(sampled_mns, data["Tg_inf"], data["K"])

                df = pd.concat(
                    [
                        df,
                        pd.DataFrame(
                            {
                                "Mn": sampled_mns,
                                "Mw": sampled_mns * pdi_mean,
                                "SMILES_start": df[entrymask]["SMILES_start"].iloc[0],
                                "SMILES_end": df[entrymask]["SMILES_end"].iloc[0],
                                "molpercent_rep_u1": 1.0,
                                "Tg": sampled_tgs,
                                "canonicalized_PSMILES_rep_u1": smiles,
                            }
                        ),
                    ]
                )
        
        print(f"Successfully loaded {CSV_PATH}. Found {len(df)} entries.")
    except FileNotFoundError:
        print(f"Error: The file {CSV_PATH} was not found.")
        return

    with SessionManager(db_path) as session:
        # Create or retrieve the main Dataset object
        dataset = Dataset.get_or_create(
            name=dataset_name, set_kwargs=dict(config=pg_dataset_config)
        )

        smiles_2_structs: Dict[str, SQLStructureModel] = {}

        for index, row in tqdm(df.iterrows()):
            for structcol in [
                "SMILES_start",
                "SMILES_end",
                "canonicalized_PSMILES_rep_u1",
                "canonicalized_PSMILES_rep_u2",
            ]:
                # check if nan, None, or empty string
                if (
                    pd.isna(row[structcol])
                    or row[structcol] == ""
                    or row[structcol] is None
                ):
                    continue
                if row[structcol] not in smiles_2_structs:
                    smiles_2_structs[row[structcol]] = SQLStructureModel.get_or_create(
                        smiles=row[structcol],
                    )
        struct_id_to_struct: Dict[int, SQLStructureModel] = {
            s.id: s for s in smiles_2_structs.values()
        }

        for index, row in tqdm(df.iterrows()):
            A = smiles_2_structs[row["SMILES_start"]]
            B = smiles_2_structs[row["SMILES_end"]]
            try:
                C = smiles_2_structs[row["canonicalized_PSMILES_rep_u1"]]
            except KeyError:
                continue
            if (
                pd.isna(row[structcol])
                or row[structcol] == ""
                or row[structcol] is None
            ):
                D = None
            else:
                D = smiles_2_structs[row["canonicalized_PSMILES_rep_u2"]]

            structure_map = {
                "A": A.id,
                "B": B.id,
                "C": C.id,
            }
            reactivity_ratios = [1]
            monomer_ratios = [1]
            if D is not None:
                structure_map["D"] = D.id
                monomer_ratios = np.array(
                    [row["molpercent_rep_u1"], row["molpercent_rep_u2"]]
                )
                monomer_ratios = monomer_ratios / monomer_ratios.sum()
                monomer_ratios = monomer_ratios.tolist()
                reactivity_ratios = [1, 1]

            params = dict(
                n_monomers=2 if D is not None else 1,
                monomer_ratios=monomer_ratios,
                reactivity_ratios=reactivity_ratios,
            )

            entry = DatasetItem.get_or_create(
                dataset=dataset,
                entry_type="random_copolymer" if D is not None else "homopolymer",
                mn=row["Mn"],
                mw=row["Mw"],
                targets=dict(Tg=row["Tg"] + 273.15),
                structure_map=structure_map,
                params=params,
            )
            sec_sim_params = {
                "Mn": entry.mn,
                "Mw": entry.mw,
            }

            if entry.sec_id is None:
                sec_entry = SECEntry.all(sim_params=sec_sim_params)
                if len(sec_entry) == 0:
                    sec = make_sec(mn=entry.mn, mw=entry.mw)
                    sec_entry = SECEntry.from_sec(sec, sim_params=sec_sim_params)
                    session.add(sec_entry)
                    session.commit()
                else:
                    sec_entry = sec_entry[0]

                entry.sec_id = sec_entry.id
                session.commit()

            graphs = GraphDatasetItemLink.get_graphs(entry)
            if len(graphs) < graphs_per_entry:
                missing_graphs = graphs_per_entry - len(graphs)
                samplemasses = entry.sec.sample_masses(n=missing_graphs)
                for samplemass in samplemasses:
                    startgroup = struct_id_to_struct[entry.structure_map["A"]]
                    endgroup = struct_id_to_struct[entry.structure_map["B"]]
                    monomers = [struct_id_to_struct[entry.structure_map["C"]]]
                    if "D" in entry.structure_map:
                        monomers.append(struct_id_to_struct[entry.structure_map["D"]])
                    poly_mass_wo_ends = samplemass - startgroup.mass - endgroup.mass
                    assert poly_mass_wo_ends > 0, "Monomass is negative"
                    monoratio = np.array(entry.params["monomer_ratios"])
                    monomer_masses = np.array([m.mass for m in monomers])
                    mean_monomass = (monomer_masses * monoratio).sum()
                    n_monos = (
                        max(
                            min_monos,
                            min(max_monos, (poly_mass_wo_ends / mean_monomass)),
                        )
                        * monoratio
                    )

                    N = 1
                    ini = MonomerDef.chaingrowth_initiator(
                        name="A",
                        molar_mass=startgroup.mass,
                        count=N,
                    )
                    endgroup = MonomerDef(
                        name="B",
                        molar_mass=endgroup.mass,
                        count=N,
                        sites=[SiteDef(type="EG", status="DORMANT")],
                    )
                    monomers = [
                        MonomerDef.chaingrowth_monomer(
                            name=n,
                            molar_mass=m,
                            count=max(1, int(N * c)),
                        )
                        for n, m, c in zip(["C", "D"], monomer_masses, n_monos)
                    ]

                    sim_input = SimulationInput(
                        monomers=monomers + [ini, endgroup],
                        reactions={
                            **ReactionSchema.chaingrowth_initiation(ini, monomers),
                            frozenset(["R", "EG"]): ReactionSchema(rate=1e-10),
                        },
                    )
                    sim = Simulation(sim_input)
                    sim.run()

                    p = sim.result.get_largest_polymer()
                    graph = p.graph
                    graph = nx.relabel.convert_node_labels_to_integers(
                        graph, first_label=0, ordering="default"
                    )

                    node_labels = [n[1]["monomer_type"] for n in graph.nodes(data=True)]

                    edges = np.array([[e[0], e[1]] for e in graph.edges]).tolist()

                    graph_entry = ReusableGraphEntry(
                        **ReusableGraphEntry.fill_values(
                            nodes=node_labels,
                            edges=edges,
                            name=f"graph_{entry.id}",
                        )
                    )
                    session.add(graph_entry)
                    session.flush()
                    GraphDatasetItemLink.link(entry, graph_entry)
                session.commit()


if __name__ == "__main__":
    populate()
