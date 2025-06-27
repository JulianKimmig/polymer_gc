import matplotlib

matplotlib.use("Agg")  # Must be called BEFORE importing pyplot
import matplotlib.pyplot as plt
import numpy as np
from pydantic import Field
from typing import Literal, TypeAlias, get_args, List, Dict, Tuple, Union
import polymer_gc
from pathlib import Path
import pandas as pd
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
from tqdm import tqdm
from polymer_gc.data.dataset import (
    GraphDatasetItemLink,
    PgDatasetConfig,
    Dataset,
    DatasetItem,
)
from polymer_gc.data.basemodels.structures import SQLStructureModel
from polymer_gc.data.graphs import ReusableGraphEntry
from polymer_gc.data.database import SessionManager, SessionRegistry
from polymer_gc.data.sec import SECEntry
from polymer_gc.sec import make_sec
from polymer_gc.utils.pdi import generate_pdi_distribution
import networkx as nx
import json
from pathlib import Path
import concurrent.futures
import time

N_POLYMERS = 100  # number of polymers per class
GRAPHS_PER_POLYMER = 5
GRAPHS_PER_SIMULATION = 100

min_mass, max_mass = 10_000, 1_000_000  # mass range
min_pdi, max_pdi = 1.1, 4.0  # PDI range
MAX_CO_MONOMERS = 3  # maximum number of monomers in a polymer
PDI_R = 2  # distribution control parameter for PDI generation
rng = np.random.RandomState(42)
db_path = "database.db"
MAX_SAMPLE_MASS = 50_000
MIN_SAMPLE_MASS = 4_000
imagedir = Path("images")
imagedir.mkdir(exist_ok=True)

BLOCK_FACTOR = 5

architecture_one_hot_targets = ["linear", "star", "cross_linked", "branching"]
structure_one_hot_targets = ["homopolymer", "random_copolymer", "gradient", "block"]

# generate a list of monomers from the tg datafile, as this is what we have now
lin_monomers = pd.read_csv("LAMALAB_CURATED_Tg_structured_polymerclass.csv")[
    "PSMILES"
].tolist()
last_graph = [None]

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


log_masses = np.log(min_mass), np.log(max_mass)

VISUALIZE_GRAPHS = False

with SessionManager(db_path) as session:
    # for smiles in tqdm(lin_monomers):
    #     SQLStructureModel.get_or_create(smiles=smiles, commit=False)
    # session.commit()

    dataset = Dataset.get_or_create(
        name="RandomArchitecture", set_kwargs=dict(config=pg_dataset_config)
    )


class PolymerMaker:
    architecture_one_hot_target: int
    structure_one_hot_target: int

    graphs_per_simulation: int = GRAPHS_PER_SIMULATION

    allowed_node_deviation: float = 1.15

    def evaluate_entry(self, entry: DatasetItem) -> bool:
        return True

    @property
    def graph_name(self) -> str:
        return f"{architecture_one_hot_targets[self.architecture_one_hot_target]}_{structure_one_hot_targets[self.structure_one_hot_target]}"

    def graph_description(self, entry: DatasetItem) -> dict:
        raise NotImplementedError("graph_description is not implemented")

    def _formatted_graph_description(self, entry: DatasetItem) -> str:
        dat = self.graph_description(entry)
        return json.dumps(dat, sort_keys=True)

    def get_params(self, entry_index: int) -> dict:
        return dict()

    def get_graph_params(self) -> dict:
        return dict()

    def __init__(self, rng: Union[int, np.random.RandomState] = 42):
        if self.architecture_one_hot_target is None:
            raise ValueError("architecture_one_hot_target is not set")
        if self.structure_one_hot_target is None:
            raise ValueError("structure_one_hot_target is not set")
        if self.graph_name is None:
            raise ValueError("graph_name is not set")
        if isinstance(rng, int):
            self.rng = np.random.RandomState(rng)
        else:
            self.rng = rng
        self._inner_seed = self.rng.randint(0, 1000000)

    def get_smiles(self, n: int) -> List[Dict[str, str]]:
        raise NotImplementedError("get_smiles is not implemented")

    def make_graphs(self, mass, entry: DatasetItem):
        # print("Make graphs of mass", mass)
        sim_config = self.get_simulation(mass, entry)
        sim_config.params.random_seed = self.rng.randint(0, 1000000)
        # print(sim_config.model_dump_json(indent=2))
        monomers = sim_config.monomers
        total_mass = (
            sum([m.molar_mass * m.count for m in monomers]) / self.graphs_per_simulation
        )

        # assert np.abs(total_mass - mass) / mass < 0.05, (
        #     f"total_mass {total_mass} != mass {mass} p/m 5%"
        # )

        sim = Simulation(sim_config)
        res = sim.run()

        polys = res.get_polymers()
        # print("PMW:",[p.molecular_weight for p in polys])
        # print("PL:",[len(p.graph) for p in polys])
        last_graph[0] = polys[0].graph
        for p in polys:
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
                    name=self.graph_name,
                    description=self._formatted_graph_description(entry),
                )
            )
            session.add(graph_entry)
        session.commit()

    def get_simulation(self, mass, entry: DatasetItem) -> SimulationInput:
        raise NotImplementedError("get_simulation is not implemented")

    def _get_structures(self, n: int) -> List[Dict[str, SQLStructureModel]]:
        monomer_smiles: List[Dict[str, SQLStructureModel]] = []
        with SessionManager(db_path) as session:
            for smilespack in self.get_smiles(n):
                newpack = {}
                for k, smiles in smilespack.items():
                    structure = SQLStructureModel.get_or_create(
                        smiles=smiles, commit=False
                    )
                    newpack[k] = structure
                monomer_smiles.append(newpack)
            session.commit()
        return monomer_smiles

    def _get_masses(self, n) -> Tuple[List[int], List[int], List[float]]:
        mn = self.rng.rand(n) * (log_masses[1] - log_masses[0]) + log_masses[0]
        mn = np.exp(mn).astype(int)
        pdi = generate_pdi_distribution(n, min_pdi, max_pdi, r=PDI_R, rng=self.rng)
        mws = (mn * pdi).astype(int)
        return mn, mws, pdi

    def _get_entry(
        self,
        i: int,
        mn: int,
        mw: int,
        structure: Dict[str, SQLStructureModel],
    ) -> DatasetItem:
        entry = DatasetItem.get_or_create(
            dataset=dataset,
            entry_type=self.graph_name,
            mn=mn,
            mw=mw,
            new_kwargs=dict(
                targets=dict(
                    hot_encoded_architecture=self.architecture_one_hot_target,
                    hot_encoded_structure=self.structure_one_hot_target,
                ),
                structure_map={k: v.id for k, v in structure.items()},
                params=self.get_params(i),
            ),
        )
        if not self.evaluate_entry(entry):
            raise ValueError(f"Entry {entry} is not valid")
        return entry

    def _get_sec(self, entry: DatasetItem) -> SECEntry:
        if entry.sec is None:
            session = SessionRegistry.get_session()
            sec = SECEntry.get(
                sim_params=dict(mn=int(entry.mn), mw=int(entry.mw)),
            )
            if sec is None:
                sec = SECEntry.from_sec(
                    make_sec(entry.mn, entry.mw),
                    sim_params=dict(mn=int(entry.mn), mw=int(entry.mw)),
                )
            entry.sec = sec
            session.add(entry)
            session.commit()
        return entry.sec

    def _get_graphs(self, entry: DatasetItem) -> List[ReusableGraphEntry]:
        pass

    def get_number_nodes(
        self, entry: DatasetItem, samplemasses: List[float]
    ) -> List[int]:
        raise NotImplementedError("get_number_nodes is not implemented")

    def _get_entry_graphs(self, entry: DatasetItem) -> List[ReusableGraphEntry]:
        existing_graphs = GraphDatasetItemLink.get_graphs(entry)

        if len(existing_graphs) >= GRAPHS_PER_POLYMER:
            return existing_graphs

        samplemasses = np.clip(
            np.array(
                entry.sec.sample_masses(
                    n=GRAPHS_PER_POLYMER - len(existing_graphs),
                    random_state=int(entry.mn + entry.mw),
                )
            ),
            MIN_SAMPLE_MASS,
            MAX_SAMPLE_MASS,
        )
        n_nodes = self.get_number_nodes(entry, samplemasses)
        # print("Required nodes", n_nodes)

        found_graphs = set()
        for n, mass_idx, c in zip(
            *np.unique(n_nodes, return_counts=True, return_index=True)
        ):
            requ = dict(
                n_nodes_min=max(int(n * 1 / self.allowed_node_deviation), 1),
                n_nodes_max=int(n * self.allowed_node_deviation),
                name=self.graph_name,
                exclude_entries=found_graphs,
                description=self._formatted_graph_description(entry),
                # n=c,
            )
            graphs = ReusableGraphEntry.get_possible_entries(**requ)
            for i in range(10):
                if len(graphs) >= c:
                    break
                self.make_graphs(samplemasses[mass_idx], entry)
                graphs = ReusableGraphEntry.get_possible_entries(**requ)

            if len(graphs) < c:
                raise ValueError(
                    f"Found {len(graphs)} graphs for length {n} but required {c}\n{requ}"
                )

            gids = [g.id for g in graphs]
            # shuffle gids
            gids = self.rng.permutation(gids)

            found_graphs.update(gids[:c].tolist())

        session = SessionRegistry.get_session()
        found_graphs = session.exec(
            ReusableGraphEntry.select().where(ReusableGraphEntry.id.in_(found_graphs))
        ).all()

        for g in found_graphs:
            GraphDatasetItemLink.link(entry, g)

        return found_graphs

    def __call__(self, dataset: Dataset):
        with SessionManager(db_path) as session:
            entries = DatasetItem.all(dataset=dataset, entry_type=self.graph_name)
        if len(entries) >= N_POLYMERS:
            return entries

        rem = N_POLYMERS - len(entries)

        structures = self._get_structures(rem)
        mns, mws, pdis = self._get_masses(rem)

        if VISUALIZE_GRAPHS:
            all_img_files = True
            for j in range(rem):
                for i in range(GRAPHS_PER_POLYMER):
                    if not (
                        imagedir / f"g_{self.__class__.__name__}_{j}_{i}.png"
                    ).exists():
                        all_img_files = False
                        break
            if all_img_files:
                return []
        with SessionManager(db_path) as session:
            for i in tqdm(
                range(rem), desc=f"Generating {self.__class__.__name__} entries"
            ):
                entry = self._get_entry(
                    i,
                    mns[i],
                    mws[i],
                    structures[i],
                )
                sec = self._get_sec(entry)

                try:
                    graphs = self._get_entry_graphs(entry)
                except Exception as e:
                    raise ValueError(
                        f"Error getting graphs for entry {i}: {entry}"
                    ) from e

                entries.append(entry)

                assert len(graphs) > 0, f"No graphs found for entry {i}"
                if VISUALIZE_GRAPHS:
                    for j, g in enumerate(graphs):
                        try:
                            with concurrent.futures.ThreadPoolExecutor() as executor:
                                future = executor.submit(
                                    visualize_polymer,
                                    g.to_nx(),
                                    save_path=imagedir
                                    / f"g_{self.__class__.__name__}_{i}_{j}.png",
                                )
                                # Wait for 30 seconds, then timeout
                                result = future.result(timeout=30)
                                print(
                                    f"Visualization completed for {self.__class__.__name__}_{i}_{j}"
                                )
                        except concurrent.futures.TimeoutError:
                            print(
                                f"Visualization timed out for {self.__class__.__name__}_{i}_{j} after 30 seconds"
                            )
                        except Exception as e:
                            print(
                                f"Visualization failed for {self.__class__.__name__}_{i}_{j}: {e}"
                            )

        return entries


class GradientMixin:
    structure_one_hot_target = structure_one_hot_targets.index("gradient")

    MAX_MONOMERS = MAX_CO_MONOMERS
    MIN_MONOMERS = 2

    def get_params(self, entry_index: int) -> dict:
        n_monomers = np.random.RandomState(self._inner_seed + entry_index).randint(
            self.MIN_MONOMERS, self.MAX_MONOMERS + 1
        )
        min_threshold = 1 / (4 * n_monomers)
        while True:
            monomer_ratios = self.rng.dirichlet(np.ones(n_monomers), size=1)[0]
            monomer_ratios = np.round(monomer_ratios, 1)
            monomer_ratios[-1] = 1 - np.sum(monomer_ratios[:-1])
            # Check if all ratios meet the minimum threshold
            if all(ratio >= min_threshold for ratio in monomer_ratios):
                break

        reactivity_ratios = [1]
        for i in range(n_monomers - 1):
            reactivity_ratios.append(reactivity_ratios[-1] * (self.rng.rand() * 10 + 1))

        # shuffle reactivity ratios
        reactivity_ratios = self.rng.permutation(reactivity_ratios).tolist()

        params = dict(
            n_monomers=n_monomers,
            monomer_ratios=monomer_ratios.tolist(),
            reactivity_ratios=reactivity_ratios,
        )
        return params

    def graph_description(self, entry: DatasetItem) -> str:
        keys = list(entry.structure_map.keys())
        dat = {k: f"backbone" for k in keys}
        dat["monomer_ratios"] = {
            k: r for r, k in zip(entry.params["monomer_ratios"], keys)
        }
        dat["reactivity_ratios"] = {
            k: r for r, k in zip(entry.params["reactivity_ratios"], keys)
        }
        return dat

    def get_smiles(self, n: int) -> List[Dict[str, str]]:
        monomer_smiles = []
        for i in range(n):
            params = self.get_params(i)
            n_monomers = params["n_monomers"]
            monomer_smiles.append(
                {
                    k: s
                    for k, s in zip(
                        [chr(65 + i) for i in range(n_monomers)],
                        self.rng.choice(lin_monomers, n_monomers, replace=False),
                    )
                }
            )
        return monomer_smiles


class HomopolymerMixin(GradientMixin):
    structure_one_hot_target = structure_one_hot_targets.index("homopolymer")

    MAX_MONOMERS = 1
    MIN_MONOMERS = 1


class RandomCopolymerMixin(GradientMixin):
    structure_one_hot_target = structure_one_hot_targets.index("random_copolymer")

    def get_params(self, entry_index: int) -> dict:
        params = super().get_params(entry_index)
        params["reactivity_ratios"] = np.ones(params["n_monomers"]).tolist()
        return params


class BlockMixin(GradientMixin):
    structure_one_hot_target = structure_one_hot_targets.index("block")

    def get_params(self, entry_index: int) -> dict:
        params = super().get_params(entry_index)
        params["reactivity_ratios"] = (
            10 ** (np.arange(0, len(params["reactivity_ratios"])) * BLOCK_FACTOR)
        ).tolist()
        # shuffle reactivity ratios
        params["reactivity_ratios"] = self.rng.permutation(
            params["reactivity_ratios"]
        ).tolist()
        return params


class LinearGradient(GradientMixin, PolymerMaker):
    architecture_one_hot_target = architecture_one_hot_targets.index("linear")

    def evaluate_entry(self, entry: DatasetItem) -> bool:
        return len(entry.structure_map) == len(entry.params["monomer_ratios"])

    def get_simulation(self, mass, entry: DatasetItem) -> SimulationInput:
        structs = [entry.structuremapentry(k) for k in entry.structure_map.keys()]
        ratios = entry.params["monomer_ratios"]
        reactivity_ratios = entry.params["reactivity_ratios"]
        highest_reactivity_index = np.argmax(reactivity_ratios)
        inimass = structs[highest_reactivity_index].mass
        remmass = mass - inimass
        mean_mass = np.sum(np.array(ratios) * np.array([s.mass for s in structs]))
        total_units = remmass / mean_mass
        intis = np.round(
            np.array(ratios) * total_units * self.graphs_per_simulation
        ).astype(int)

        pseudoinis = [
            MonomerDef.chaingrowth_initiator(
                name=chr(65 + highest_reactivity_index),
                molar_mass=inimass,
                count=self.graphs_per_simulation,
                active_site_name=f"R_{chr(65 + highest_reactivity_index)}",
            )
        ]

        monos = [
            MonomerDef.chaingrowth_monomer(
                name=k,
                molar_mass=s.mass,
                count=ini,
                head_name=f"H_{k}",
                tail_name=f"T_{k}",
            )
            for k, s, ini in zip(entry.structure_map.keys(), structs, intis)
        ]

        r = {}
        for i, k in enumerate(entry.structure_map.keys()):
            for j, l in enumerate(entry.structure_map.keys()):
                r[frozenset([f"R_{k}", f"H_{l}"])] = ReactionSchema(
                    rate=reactivity_ratios[j],
                    activation_map={f"T_{l}": f"R_{l}"},
                )

        return SimulationInput(
            monomers=pseudoinis + monos,
            reactions=r,
            params=SimParams(),
        )

    def get_number_nodes(
        self, entry: DatasetItem, samplemasses: List[float]
    ) -> List[int]:
        params = entry.params
        n_monomers = params["n_monomers"]
        monomer_ratios = np.array(params["monomer_ratios"])
        monomer_masses = np.array(
            [entry.structuremapentry(k).mass for k in entry.structure_map.keys()]
        )

        mean_mass = np.sum(monomer_masses * monomer_ratios) / n_monomers
        total_units = np.array(samplemasses) / (mean_mass * n_monomers)

        return np.round(total_units).astype(int)


class LinearHomopolymer(HomopolymerMixin, LinearGradient):
    pass


class LinearRandomCopolymer(RandomCopolymerMixin, LinearGradient):
    pass


class LinearBlock(BlockMixin, LinearRandomCopolymer):
    pass


class StarGradientCopolymer(GradientMixin, PolymerMaker):
    architecture_one_hot_target = architecture_one_hot_targets.index("star")

    MAX_ARMS = 10

    def get_smiles(self, n: int) -> List[Dict[str, str]]:
        monomer_smiles = super().get_smiles(n)
        for ps in monomer_smiles:
            ps[chr(65 + len(ps))] = self.rng.choice(lin_monomers)
        return monomer_smiles

    def get_params(self, entry_index: int) -> dict:
        params = super().get_params(entry_index)
        params["n_arms"] = self.rng.randint(3, self.MAX_ARMS + 1)
        return params

    def graph_description(self, entry: DatasetItem) -> str:
        dat = super().graph_description(entry)
        keys = list(entry.structure_map.keys())
        corekey = keys[-1]
        dat[corekey] = "core"
        dat["n_arms"] = entry.params["n_arms"]
        return dat

    def get_number_nodes(
        self, entry: DatasetItem, samplemasses: List[float]
    ) -> List[int]:
        params = entry.params
        n_monomers = params["n_monomers"]
        monomer_ratios = np.array(params["monomer_ratios"])
        monomer_ratios = monomer_ratios / np.sum(monomer_ratios)
        keys = list(entry.structure_map.keys())
        monokeys = keys[:-1]
        corekey = keys[-1]
        monomer_masses = np.array([entry.structuremapentry(k).mass for k in monokeys])

        masses_wo_core = np.array(samplemasses) - entry.structuremapentry(corekey).mass
        # print(monomer_masses)
        mean_mass = np.sum(monomer_masses * monomer_ratios)
        total_units = 1 + masses_wo_core / (mean_mass)
        # print(mean_mass, total_units)

        return np.round(total_units).astype(int)

    def get_simulation(self, mass, entry: DatasetItem) -> SimulationInput:
        keys = list(entry.structure_map.keys())
        monokeys = keys[:-1]
        corekey = keys[-1]
        structs = [entry.structuremapentry(k) for k in keys]
        corestruct = structs[-1]
        monomerstructs = structs[:-1]

        ratios = np.array(entry.params["monomer_ratios"])
        arm_mass = mass - corestruct.mass
        monomer_masses = np.array([s.mass for s in monomerstructs])
        mean_mass = np.sum(monomer_masses * ratios)
        arm_units = arm_mass / mean_mass
        reactivity_ratios = entry.params["reactivity_ratios"]
        intis = np.round(
            np.array(ratios) * arm_units * self.graphs_per_simulation
        ).astype(int)
        coreini = [
            MonomerDef(
                name=corekey,
                molar_mass=corestruct.mass,
                count=self.graphs_per_simulation,
                sites=[
                    SiteDef(type=f"R_{corekey}", status="ACTIVE")
                    for i in range(entry.params["n_arms"])
                ],
            )
        ]

        monos = [
            MonomerDef.chaingrowth_monomer(
                name=k,
                molar_mass=s.mass,
                count=int(ini),
                head_name=f"H_{k}",
                tail_name=f"T_{k}",
            )
            for k, s, ini in zip(monokeys, monomerstructs, intis)
        ]

        r = {}
        for i, k in enumerate(monokeys):
            r[frozenset([f"R_{corekey}", f"H_{k}"])] = ReactionSchema(
                rate=reactivity_ratios[i],
                activation_map={f"T_{k}": f"R_{k}"},
            )

            for j, l in enumerate(monokeys):
                r[frozenset([f"R_{k}", f"H_{l}"])] = ReactionSchema(
                    rate=min(reactivity_ratios[i], reactivity_ratios[j]),
                    activation_map={f"T_{l}": f"R_{l}"},
                )

        sim = SimulationInput(
            monomers=coreini + monos,
            reactions=r,
            params=SimParams(),
        )
        # print(sim.model_dump_json(indent=2))
        return sim


class StarHomopolymer(HomopolymerMixin, StarGradientCopolymer):
    pass


class StarRandomCopolymer(RandomCopolymerMixin, StarGradientCopolymer):
    pass


class StarBlock(BlockMixin, StarGradientCopolymer):
    pass


class CrossLinkedGradient(LinearGradient):
    architecture_one_hot_target = architecture_one_hot_targets.index("cross_linked")
    allowed_node_deviation = 1.5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.repeat_simulation = self.graphs_per_simulation
        self.graphs_per_simulation = 10

    def get_simulation(self, mass, entry: DatasetItem) -> SimulationInput:
        sim_config = super().get_simulation(mass, entry)
        mono_0 = sim_config.monomers[-1]
        sites = list([s.model_dump() for s in mono_0.sites])
        ser_sim_config = sim_config.model_dump_json()
        for site in sites:
            oldtype = site["type"]
            newtype = f"{site['type']}_2"
            ser_sim_config = ser_sim_config.replace(oldtype, newtype)
            site["type"] = newtype
            mono_0.sites.append(SiteDef(**site))

        deser_sim_config = SimulationInput.model_validate_json(ser_sim_config)
        sim_config.reactions.update(deser_sim_config.reactions)
        for mono in sim_config.monomers:
            mono.count = int(max(1, mono.count / self.graphs_per_simulation))

        return sim_config

    def make_graphs(self, mass, entry: DatasetItem):
        # as one simulation gives always one graph, we need to run it multiple times
        for _ in range(self.repeat_simulation):
            super().make_graphs(mass, entry)


class CrossLinkedHomopolymer(HomopolymerMixin, CrossLinkedGradient):
    pass


class CrossLinkedRandomCopolymer(RandomCopolymerMixin, CrossLinkedGradient):
    pass


class BranchingGradient(LinearGradient):
    architecture_one_hot_target = architecture_one_hot_targets.index("branching")
    allowed_node_deviation = 1.15

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.repeat_simulation = self.graphs_per_simulation
        # self.graphs_per_simulation = 10

    def get_simulation(self, mass, entry: DatasetItem) -> SimulationInput:
        sim_config = super().get_simulation(mass, entry)
        mono_0 = sim_config.monomers[-1]
        sites = list([s.model_dump() for s in mono_0.sites])
        for site in sites[1:]:
            oldtype = site["type"]
            newtype = f"{site['type']}_2"
            site["type"] = newtype
            mono_0.sites.append(SiteDef(**site))
            for reaction in sim_config.reactions.values():
                if reaction.activation_map.get(oldtype):
                    reaction.activation_map[newtype] = reaction.activation_map[oldtype]

        # for mono in sim_config.monomers:
        # mono.count = int(max(1,mono.count/self.graphs_per_simulation))

        # print(sim_config.model_dump_json(indent=2))
        return sim_config

    # def make_graphs(self, mass, entry: DatasetItem):
    #     # as one simulation gives always one graph, we need to run it multiple times
    #     for _ in range(self.repeat_simulation):
    #         super().make_graphs(mass, entry)


class BranchingHomopolymer(HomopolymerMixin, BranchingGradient):
    pass


class BranchingRandomCopolymer(RandomCopolymerMixin, BranchingGradient):
    pass


class BranchingBlock(BlockMixin, BranchingGradient):
    pass


rints = rng.randint(0, 1000000, 10000).tolist()
entries = []
ge = None
try:
    for i in range(1):
        try:
            entries += LinearHomopolymer(rng=rints.pop())(dataset)
        except Exception as e:
            raise e
            print(e)
            ge = e
            pass
        try:
            entries += LinearRandomCopolymer(rng=rints.pop())(dataset)
        except Exception as e:
            raise e
            print(e)
            ge = e
            pass
        try:
            entries += LinearGradient(rng=rints.pop())(dataset)
        except Exception as e:
            raise e
            print(e)
            ge = e
            pass
        try:
            entries += LinearBlock(rng=rints.pop())(dataset)
        except Exception as e:
            raise e
            print(e)
            ge = e
            pass
        try:
            entries += StarGradientCopolymer(rng=rints.pop())(dataset)
        except Exception as e:
            raise e
            print(e)
            ge = e
            pass
        try:
            entries += StarHomopolymer(rng=rints.pop())(dataset)
        except Exception as e:
            raise e
            print(e)
            ge = e
            pass
        try:
            entries += StarRandomCopolymer(rng=rints.pop())(dataset)
        except Exception as e:
            raise e
            print(e)
            ge = e
            pass
        try:
            entries += StarBlock(rng=rints.pop())(dataset)
        except Exception as e:
            raise e
            print(e)
            ge = e
            pass

        try:
            entries += BranchingHomopolymer(rng=rints.pop())(dataset)
        except Exception as e:
            raise e
            print(e)
            ge = e
            pass
        try:
            entries += BranchingRandomCopolymer(rng=rints.pop())(dataset)
        except Exception as e:
            raise e
            print(e)
            ge = e
            pass
        try:
            entries += BranchingGradient(rng=rints.pop())(dataset)
        except Exception as e:
            raise e
            print(e)
            ge = e
            pass
        try:
            entries += BranchingBlock(rng=rints.pop())(dataset)
        except Exception as e:
            raise e
            print(e)
            ge = e
            pass

        try:
            entries += CrossLinkedRandomCopolymer(rng=rints.pop())(dataset)
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
            entries += CrossLinkedHomopolymer(rng=rints.pop())(dataset)
        except Exception as e:
            raise e
            print(e)
            ge = e
            pass

    if ge is not None:
        raise ge
except Exception as e:
    if last_graph[0]:
        visualize_polymer(last_graph[0], save_path=f"last_graph.png")

    raise e
