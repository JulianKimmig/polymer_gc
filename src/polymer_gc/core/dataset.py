from pydantic import BaseModel, Field, field_validator, ConfigDict, model_validator
from typing import List, TypeVar, Generic, Optional, Union, Literal
from .monomer import Monomer
from abc import ABC, abstractmethod
from ..sec import SECDataBase
import pandas as pd
from typing_extensions import (
    TypedDict,
)  # loading from typing_extensions for compatibility pydantic

from ..sec import SimSEC
from polymer_gc.graph import make_linear_polymer_graphs


class SmallMoleculeEntry(BaseModel):
    smiles: str = Field(..., description="SMILES representation of the molecule.")
    name: Optional[str] = None


class SplitOrientedDataFrame(TypedDict):
    index: List[Union[str, int, float]]
    columns: List[str]
    data: List[List[Union[str, int, float]]]


class PolymerSequence(BaseModel):
    """
    Describes the sequence of monomers within a polymer segment.
    This is typically used for linear segments or components of more complex architectures.
    """

    sequence_type: str = Field(..., description="Type of monomer sequence.")


class Homopolymer(PolymerSequence):
    sequence_type: Literal["homopolymer"] = Field(
        "homopolymer", description="Sequence is a homopolymer."
    )


class Copolymer(PolymerSequence):
    sequence_type: Literal["copolymer"] = Field(
        "copolymer", description="Sequence is a copolymer."
    )

    ratios: List[Union[int, float]] = Field(
        ...,
        description="Ratios of the monomers in the copolymer. Must sum to 1.",
    )


class RandomCopolymer(Copolymer):
    sequence_type: Literal["random_copolymer"] = Field(
        "random_copolymer", description="Sequence is a random copolymer."
    )


class BlockCopolymer(Copolymer):
    sequence_type: Literal["block_copolymer"] = Field(
        "block_copolymer", description="Sequence of the polymer."
    )


class AlternatingCopolymer(PolymerSequence):
    sequence_type: Literal["alternating_copolymer"] = Field(
        "alternating_copolymer", description="Sequence is an alternating copolymer."
    )


class GradientCopolymer(Copolymer):
    sequence_type: Literal["gradient_copolymer"] = Field(
        "gradient_copolymer", description="Sequence of the polymer."
    )

    gradient: List[List[float]] = Field(
        ...,
        description="Gradient of the copolymer. Must be a list of lists, where each inner list represents a gradient",
    )


AnyPolymerSequence = Union[
    Homopolymer,
    RandomCopolymer,
    AlternatingCopolymer,
    BlockCopolymer,
    GradientCopolymer,
]


class PolymerArchitecture(BaseModel, ABC):
    """
    Base model for defining the overall polymer architecture/topology.
    """

    architecture_type: str = Field(
        ..., description="The primary architectural classification of the polymer."
    )

    @abstractmethod
    def make_graphs(
        self, masses: List[float], sequence: AnyPolymerSequence, monomers: List[Monomer]
    ):
        """
        Generate graphs for the polymer architecture.
        This method should be implemented by subclasses to create specific graphs.
        """
        raise NotImplementedError("Subclasses must implement the make_graphs method.")


class LinearPolymer(PolymerArchitecture):
    architecture_type: Literal["linear"] = Field(
        "linear", description="Linear polymer architecture."
    )

    def make_graphs(
        self, masses: List[float], sequence: AnyPolymerSequence, monomers: List[Monomer]
    ):
        if sequence.sequence_type == "homopolymer":
            return make_linear_polymer_graphs(masses, monomers)
        elif sequence.sequence_type == "random_copolymer":
            return make_linear_polymer_graphs(
                masses,
                monomers,
                rel_content=[[i] for i in sequence.ratios],
            )
        else:
            raise ValueError(
                f"Sequence type {sequence.sequence_type} not supported for linear polymer."
            )


class BranchedPolymer(PolymerArchitecture):
    architecture_type: Literal["branched"] = Field(
        "branched", description="Branched polymer architecture."
    )


class CrosslinkedPolymer(PolymerArchitecture):
    architecture_type: Literal["crosslinked"] = Field(
        "crosslinked", description="Crosslinked polymer architecture."
    )


class StarPolymer(PolymerArchitecture):
    architecture_type: Literal["star"] = Field(
        "star", description="Star polymer architecture."
    )


class CombPolymer(PolymerArchitecture):
    architecture_type: Literal["comb"] = Field(
        "comb", description="Comb polymer architecture."
    )


class DendriticPolymer(PolymerArchitecture):
    architecture_type: Literal["dendritic"] = Field(
        "dendritic", description="Dendritic polymer architecture."
    )
    branchings: int = Field(
        2,
        description="Number of branchings in the dendritic polymer. Default is 2.",
    )


class CyclicPolymer(LinearPolymer):
    architecture_type: Literal["cyclic"] = Field(
        "cyclic", description="Cyclic polymer architecture."
    )


AnyPolymerArchitecture = Union[
    LinearPolymer,
    BranchedPolymer,
    CrosslinkedPolymer,
    StarPolymer,
    CombPolymer,
    DendriticPolymer,
    CyclicPolymer,
]


class DataSetEntry(BaseModel):
    monomers: List[Monomer]
    mn: float = Field(..., description="Number average molecular weight.")
    mw: float = Field(..., description="Weight average molecular weight.")
    architecture: AnyPolymerArchitecture = Field(
        ...,
        description="Architecture of the polymer.",
        default_factory=LinearPolymer,
        discriminator="architecture_type",
    )
    sequence: AnyPolymerSequence = Field(
        ...,
        description="Sequence of the polymer.",
        default_factory=Homopolymer,
        discriminator="sequence_type",
    )
    sec_raw: Optional[SplitOrientedDataFrame] = Field(
        None,
        description="Raw SEC data for the polymer. If not provided, SEC data will be generated.",
    )
    sec_calibration_params: Optional[List[float]] = Field(
        None,
        description="Calibration parameters for the SEC data. If not provided, default values will be used.",
    )

    @model_validator(mode="before")
    @classmethod
    def calculate_missing_weight_params(cls, data: dict):
        mn = data.get("mn")
        mw = data.get("mw")
        if mn is not None and mw is not None:
            return data
        pdi = data.get("pdi")

        if pdi is None or (mn is None and mw is None):
            raise ValueError(
                "Either both 'mn' and 'mw' or one of them and 'pdi' have to be provided."
            )

        if mn is None:
            data["mn"] = mw / pdi
            return data

        if mw is None:
            data["mw"] = mn * pdi
            return data

        if mn > mw:
            raise ValueError(
                "The number average molecular weight (mn) must be less than the weight average molecular weight (mw)."
            )

    @model_validator(mode="after")
    def check_sequence(self) -> "DataSetEntry":
        """
        After the Dataset is created, iterate through all monomers
        and ensure identical monomers are the same object.
        """
        if len(self.monomers) == 0:
            raise ValueError("At least one monomer must be provided.")
        if len(self.monomers) == 1 and not isinstance(self.sequence, Homopolymer):
            raise ValueError(
                "If only one monomer is provided, the architecture must be a homopolymer."
            )

        if len(self.monomers) > 1 and isinstance(self.sequence, Homopolymer):
            raise ValueError(
                "If more than one monomer is provided, the architecture must not be a homopolymer."
            )

        if not isinstance(
            self.sequence, (Copolymer, Homopolymer, AlternatingCopolymer)
        ):
            raise ValueError(f"This shoudl not happen: {self.sequence}")

        if isinstance(self.sequence, Copolymer):
            if not len(self.monomers) == len(self.sequence.ratios):
                raise ValueError(
                    "The number of monomers must match the number of ratios in the copolymer."
                )

        return self

    @model_validator(mode="after")
    def check_architecture(self) -> "DataSetEntry":
        """
        After the Dataset is created, iterate through all monomers
        and ensure identical monomers are the same object.
        """
        if not isinstance(self.architecture, AnyPolymerArchitecture):
            raise ValueError("This shoudl not happen")

        if isinstance(self.architecture, LinearPolymer):
            if not isinstance(
                self.sequence, (Copolymer, Homopolymer, AlternatingCopolymer)
            ):
                raise ValueError(
                    "The architecture must be a linear polymer if the sequence is a copolymer or homopolymer."
                )
        elif isinstance(self.architecture, BranchedPolymer):
            if not isinstance(
                self.sequence, (Copolymer, Homopolymer, AlternatingCopolymer)
            ):
                raise ValueError(
                    "The architecture must be a branched polymer if the sequence is a copolymer or homopolymer."
                )
        return self

    @property
    def sec(self) -> Union[SECDataBase, None]:
        """
        Returns the SEC data for the entry.
        """
        if hasattr(self, "_sec"):
            return self._sec
        if self.sec_raw is not None:
            # Assuming SECDataBase can be initialized with raw data

            self._sec = SimSEC(
                pd.DataFrame(
                    self.sec_raw["data"],
                    columns=self.sec_raw["columns"],
                    index=self.sec_raw["index"],
                ),
                calibration_params=self.sec_calibration_params,
            )
            return self._sec
        return None

    @sec.setter
    def sec(self, value: SECDataBase):
        """
        Sets the SEC data for the entry.
        """
        if not isinstance(value, SECDataBase):
            raise ValueError("SEC data must be an instance of SECDataBase.")
        # self.sec_raw = json.loads(sec._raw_data.to_json(orient="split"))
        self._sec = value

    def make_graphs(self, n: int):
        """
        Generates graphs for the entry.
        """
        if self.sec is None:
            raise ValueError("SEC data is not available.")
        # Placeholder for graph generation logic
        # self._graphs = generate_graphs(self.sec, n)
        masses = self.sec.sample(n=n)

        return self.architecture.make_graphs(
            masses=masses,
            sequence=self.sequence,
            monomers=self.monomers,
        )


T = TypeVar("T", bound=DataSetEntry)


class Dataset(BaseModel, Generic[T]):
    name: str
    items: List[T]

    @property
    def unique_monomers(self) -> List[Monomer]:
        _unique_monomers = set()
        for entry in self.items:
            _unique_monomers.update(entry.monomers)
        return list(_unique_monomers)

    @model_validator(mode="after")
    def _consolidate_monomers(self) -> "Dataset[T]":
        """
        After the Dataset is created, iterate through all monomers
        and ensure identical monomers are the same object.
        """

        interning_cache: dict[str, Monomer] = {}
        for entry in self.items:
            object.__setattr__(
                entry,
                "monomers",
                [interning_cache.setdefault(m.smiles, m) for m in entry.monomers],
            )

        return self


class PgDatasetConfig(BaseModel):
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
