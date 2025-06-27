from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from sqlmodel import Column, JSON, Field, Relationship, UniqueConstraint, select
from pydantic import model_validator
from polymcsim.schemas import SimulationInput
from .basemodels.core import Base
from .sec import SECEntry
from .database import SessionRegistry
from .basemodels.structures import SQLStructureModel
from .graphs import ReusableGraphEntry
import numpy as np
from tqdm import tqdm
from sqlalchemy.orm import joinedload

class PgDatasetConfig(BaseModel):
    id: Optional[int] = Field(default=None, primary_key=True)

    embedding: str = Field(
        "PolyBERT",
        description="Embedding method to be used. PolyBERT as default.",
    )
    num_bins: int = Field(
        100,
        description="Number of bins for the mass distribution histogram.",
    )
    log_start: float = Field(
        1.0,
        description="Start of the logarithmic scale for the mass distribution.",
    )
    log_end: float = Field(
        7.0,
        description="End of the logarithmic scale for the mass distribution.",
    )
    n_graphs: int = Field(
        5,
        description="Number of graphs to be generated from the dataset.",
    )
    targets: List[str] = Field(
        ...,
        description="List of target properties to be predicted.",
    )
    additional_features: List[str] = Field(
        ...,
        description="List of additional features to be included in the dataset.",
        default_factory=list,
    )
    target_classes: Optional[Dict[str, List[str]]] = Field(
        None,
        description="List of classes for each target property. E.g., {'hot_encoded_architecture': ['linear', 'star', 'cross_linked', 'branching'], 'hot_encoded_structure': ['homopolymer', 'random_copolymer', 'gradient', 'block']}",
    )

    def to_json(self):
        return self.model_dump()


class DatasetItem(Base, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    dataset_id: int = Field(foreign_key="datasets.id")
    dataset: "Dataset" = Relationship(back_populates="items")
    targets: Dict[str, Any] = Field(sa_column=Column(JSON))
    entry_type: str = Field(default="")
    mn: float = Field()
    mw: float = Field()
    sec_id: Optional[int] = Field(foreign_key="secentry.id")
    structure_map: Optional[Dict[str, int]] = Field(
        sa_column=Column(JSON)
    )  # key: monomer name, value: monomer id
    params: Dict[str, Any] = Field(
        sa_column=Column(JSON)
    )  # {"n_monomers": 2, "monomer_ratios": [0.5, 0.5], "reactivity_ratios": [1.0, 1.0]}

    @property
    def sec(self):
        session = SessionRegistry.get_session()
        if self.sec_id is None:
            return None
        return session.get(SECEntry, self.sec_id) if session else None

    @sec.setter
    def sec(self, value: SECEntry):
        self.sec_id = value.id

    def structuremapentry(self, key):
        if not hasattr(self, "_resolved_structure_map"):
            if self.structure_map is None:
                raise ValueError(
                    "structure_map is None, cannot resolve structure entries"
                )
            self._resolved_structure_map = {
                k: SQLStructureModel.get(id=v) for k, v in self.structure_map.items()
            }
        sentry = self._resolved_structure_map[key]
        if sentry is None:
            self._resolved_structure_map[key] = SQLStructureModel.get(
                id=self.structure_map[key]
            )
        sentry = self._resolved_structure_map[key]
        if sentry is None:
            raise ValueError(
                f"Structure not found for key:{key} and id:{self.structure_map[key]}"
            )
        return sentry


class Dataset(Base, table=True):
    __tablename__ = "datasets"
    id: Optional[int] = Field(default=None, primary_key=True)
    dict_config: Dict[str, Any] = Field(sa_column=Column(JSON))
    name: str = Field(unique=True)
    items: List[DatasetItem] = Relationship(back_populates="dataset")

    @model_validator(mode="before")
    @classmethod
    def check_config(cls, data):
        if "config" in data:
            data["dict_config"] = PgDatasetConfig.model_validate(
                data["config"]
            ).model_dump()
        return data

    @property
    def config(self):
        return (
            PgDatasetConfig.model_validate(self.dict_config)
            if self.dict_config is not None
            else None
        )

    @config.setter
    def config(self, value: PgDatasetConfig):
        self.dict_config = value.model_dump()

    def load_entries_data(self):
        all_embeddings = {}
        entries = []
        all_graphs = []
        targets = {k: [] for k in self.config.targets}
        all_entry_graphs = []
        cg = 0
        required_structureids = set()
        for entry in self.items:
            required_structureids.update(list(entry.structure_map.values()))
        required_structureids = list(required_structureids)
        required_structureids.sort()
        with SessionRegistry.get_session() as session:
            required_structures = session.exec(
            select(SQLStructureModel).where(
                SQLStructureModel.id.in_(required_structureids)
            ).options(joinedload(SQLStructureModel.embeddings))).unique().all()
            required_structures_dict = {s.id: s for s in required_structures}

            # load all embeddings
            SQLStructureModel.batch_get_embedding(required_structures, self.config.embedding)

            # required structures ny id in order
            for i, entry in tqdm(enumerate(self.items), total=len(self.items), desc="Loading entries data"):
                entries.append(entry)
                entry_structures: Dict[str_, SQLStructureModel] = {
                    k: required_structures_dict[v] for k, v in entry.structure_map.items()
                }
                entry_graphs = []
                for k, v in targets.items():
                    v.append(entry.targets[k])

                for k, str_ in entry_structures.items():
                    if str_.id in all_embeddings:
                        continue
                    all_embeddings[str_.id] = str_.get_embedding(self.config.embedding)

                graphs = GraphDatasetItemLink.get_graphs(entry)
                for graph in graphs:
                    nodes = np.array([entry.structure_map[n] for n in graph.nodes])
                    edges = np.array(graph.edges)
                    all_graphs.append(
                        {
                            "nodes": nodes,
                            "edges": edges,
                            "entry_id": entry.id,
                            "graph_id": graph.id,
                            "entry_pos": i,
                        }
                    )
                    entry_graphs.append(cg)
                    cg += 1
                all_entry_graphs.append(entry_graphs)

        structure_ids = np.array(list(all_embeddings.keys()))
        all_embeddings = np.array([all_embeddings[sid] for sid in structure_ids])

        data = {
            "structure_ids": structure_ids,
            "all_embeddings": all_embeddings,
            "graphs": all_graphs,
            "entry_graphs": all_entry_graphs,
            "targets": {k: np.array(v) for k, v in targets.items()},
        }

        return data


class GraphDatasetItemLink(Base, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    dataset_item_id: int = Field(foreign_key=DatasetItem.__tablename__ + ".id")
    dataset_item: DatasetItem = Relationship()
    graph_id: int = Field(foreign_key=ReusableGraphEntry.__tablename__ + ".id")
    graph: ReusableGraphEntry = Relationship()

    # graph item pairs are unique
    __table_args__ = (
        UniqueConstraint("dataset_item_id", "graph_id", name="graph_item_unique"),
    )

    @classmethod
    def link(cls, dataset_item: DatasetItem, graph: ReusableGraphEntry):
        session = SessionRegistry.get_session()
        if session is None:
            raise ValueError("No session found")
        # if link already exists, return it
        link = session.exec(
            select(cls).where(
                cls.dataset_item_id == dataset_item.id, cls.graph_id == graph.id
            )
        ).first()
        if link:
            return link
        link = cls(dataset_item_id=dataset_item.id, graph_id=graph.id)
        session.add(link)
        session.commit()
        return link

    @classmethod
    def get_graphs(cls, dataset_item: DatasetItem) -> List[ReusableGraphEntry]:
        session = SessionRegistry.get_session()
        if session is None:
            raise ValueError("No session found")
        return [
            l.graph
            for l in session.exec(
                select(cls).where(cls.dataset_item_id == dataset_item.id)
            ).all()
        ]
