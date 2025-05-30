from tqdm import tqdm
import os
import os.path as osp
import json
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from typing import Dict, Generic, TypeVar
import polymer_gc
from hashlib import md5

from torch_geometric.io import fs
import tempfile

T = TypeVar("T", bound=polymer_gc.dataset.Dataset)


def atomic_torch_save(obj, path):
    dir_name = osp.dirname(path)
    with tempfile.NamedTemporaryFile(dir=dir_name, delete=False) as tmp:
        temp_path = tmp.name
    try:
        torch.save(obj, temp_path)
        os.replace(temp_path, path)  # atomic on most platforms
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


class PolymerGraphDataset(Dataset, Generic[T]):
    def graphconfighash(self):
        return md5(
            json.dumps(
                {
                    k: getattr(self.config, k, None)
                    for k in ["targets", "additional_features"]
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()

    def __init__(self, dataset: T, root, config: polymer_gc.dataset.PgDatasetConfig):
        self.dataset = dataset
        self.config = (
            config
            if isinstance(config, polymer_gc.dataset.PgDatasetConfig)
            else polymer_gc.dataset.PgDatasetConfig(**config)
        )
        self.entryhash_map = self.make_entryhash_map()
        self._processed_file_names = self.make_processed_file_names()
        self.basedir = root
        super().__init__(root=osp.join(root, self.graphconfighash()))
        self.embeddings = self.load_emebddings()
        self.mass_distributions = self.load_mass_distributions()

    def make_entryhash_map(self) -> Dict[str, polymer_gc.dataset.DataSetEntry]:
        entry: polymer_gc.dataset.DataSetEntry
        ehm = {}
        print("Making entry hash map...")
        for entry in self.dataset.items:
            ehm[md5(entry.model_dump_json().encode()).hexdigest()] = entry
        return ehm

    def make_processed_file_names(self):
        print("Making processed file names...")
        processed_file_names = []

        for entryhash, entry in self.entryhash_map.items():
            for i in range(self.config.n_graphs):
                processed_file_names.append(f"{entryhash}_{i}.pt")

        return processed_file_names

    @property
    def processed_file_names(self):
        return self._processed_file_names

    @property
    def embeddings_dir(self):
        emb_dir = osp.join(self.root, "embeddings")
        if not osp.exists(emb_dir):
            fs.makedirs(emb_dir)
        return emb_dir

    def make_embeddings(self):
        print("Making embeddings...")
        embedding = self.config.embedding
        monomerembeddings = {}
        self.dataset.unique_monomers[0].add_batch_embeddings(
            self.dataset.unique_monomers,
            embedding,
            values=[
                monomerembeddings.get(monomer.smiles, None)
                for monomer in self.dataset.unique_monomers
            ],
        )
        for monomer in self.dataset.unique_monomers:
            monomerembeddings[monomer.smiles] = torch.Tensor(
                monomer.embeddings[embedding].value
            ).float()

        atomic_torch_save(
            monomerembeddings,
            osp.join(
                self.embeddings_dir, f"monomer_embeddings_{self.config.embedding}.pt"
            ),
        )

    def prepare(self):
        for entryhash, entry in tqdm(
            self.entryhash_map.items(),
            desc="Preparing data (mass dist)",
            total=len(self.entryhash_map),
        ):
            self.get_mass_distribution(entryhash, save=False)
        # save mass distributions
        atomic_torch_save(
            self.mass_distributions, osp.join(self.root, "mass_distributions.pt")
        )

    def load_mass_distributions(self):
        file = osp.join(self.root, "mass_distributions.pt")
        if not osp.exists(file):
            return {}
        print("laoding", file)
        mass_distributions = torch.load(file)
        return mass_distributions

    def load_emebddings(self):
        file = osp.join(
            self.embeddings_dir, f"monomer_embeddings_{self.config.embedding}.pt"
        )
        if not osp.exists(file):
            print("Embeddings not found, making embeddings...")
            self.make_embeddings()
        monomerembeddings = torch.load(file)
        try:
            for monomer in self.dataset.unique_monomers:
                if monomer.smiles not in monomerembeddings:
                    raise ValueError()
        except ValueError as e:
            print("Monomer embeddings not found, making embeddings...")
            self.make_embeddings()

        monomerembeddings = torch.load(file)

        return monomerembeddings

    def process(self):
        entry: polymer_gc.dataset.DataSetEntry
        processdir = self.processed_dir
        self.make_embeddings()
        for entry in tqdm(self.dataset.items):
            entryhash = md5(entry.model_dump_json().encode()).hexdigest()
            targetfiles = [
                osp.join(processdir, f"{entryhash}_{i}.pt")
                for i in range(self.config.n_graphs)
            ]
            # if all files exist, skip
            if all(osp.exists(targetfile) for targetfile in targetfiles):
                continue

            graphs = entry.make_graphs(n=self.config.n_graphs)

            for i, graph in enumerate(graphs):
                targetfile = targetfiles[i]
                smiles = [mono.smiles for mono in graph.monomers]

                additional_features = []
                for feature in self.config.additional_features:
                    fval = getattr(entry, feature)
                    fval = np.array(fval).flatten()
                    additional_features.append(fval)

                if len(additional_features) == 0:
                    additional_features = torch.zeros(1, 0).float()
                else:
                    additional_features = (
                        np.concatenate(additional_features).astype(np.float32).flatten()
                    )
                    additional_features = (
                        torch.Tensor(additional_features).float().unsqueeze(0)
                    )

                data = Data(
                    x=torch.Tensor(graph.nodes).long(),
                    edge_index=torch.Tensor(graph.edges.T).long(),
                    y=torch.cat(
                        [
                            torch.Tensor([getattr(entry, target)])
                            .flatten()
                            .float()
                            .reshape(1, -1)
                            for target in self.config.targets
                        ],
                        axis=1,
                    ),
                    additional_features=additional_features,
                    _entryhash=entryhash,
                    smiles=smiles,
                )
                atomic_torch_save(data, targetfile)

    def get_mass_distribution(self, entryhash, save=True):
        if entryhash in self.mass_distributions:
            return self.mass_distributions[entryhash]
        sec = self.entryhash_map[entryhash].sec
        bin_edges = np.linspace(
            self.config.log_start,
            self.config.log_end,
            self.config.num_bins + 1,
        )

        log_mwd, mwdmasses = sec.calc_logMW(
            10**self.config.log_start, 10**self.config.log_end
        )
        log_masses = np.log10(mwdmasses)
        mask = (log_masses >= self.config.log_start) & (
            log_masses <= self.config.log_end
        )
        filtered_logx = log_masses[mask]
        filtered_y = log_mwd[mask, 0]
        bin_indices = np.digitize(filtered_logx, bin_edges) - 1
        hist = np.zeros(self.config.num_bins)
        for i in range(self.config.num_bins):
            values = filtered_y[bin_indices == i]
            if len(values) > 0:
                hist[i] = values.mean()

        self.mass_distributions[entryhash] = torch.Tensor(hist).float().unsqueeze(0)
        if save:
            atomic_torch_save(
                self.mass_distributions, osp.join(self.root, "mass_distributions.pt")
            )
        return self.mass_distributions[entryhash]

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(
            osp.join(self.processed_dir, self._processed_file_names[idx]),
            weights_only=False,
        )
        data.mass_distribution = self.get_mass_distribution(data._entryhash)
        rel_embeddings = torch.stack([self.embeddings[smile] for smile in data.smiles])
        data.x = rel_embeddings[data.x]
        del data.smiles
        del data._entryhash
        return data
