from pydantic import BaseModel, Field, field_validator, ConfigDict, model_validator
from typing import List, TypeVar, Generic, Optional, Union, Literal
from .monomer import Monomer
from ..sec import SECDataBase
import pandas as pd
from typing_extensions import (
    TypedDict,
)  # loading from typing_extensions for compatibility pydantic

from ..sec import SimSEC


class SmallMoleculeEntry(BaseModel):
    smiles: str = Field(..., description="SMILES representation of the molecule.")
    name: Optional[str] = None


class SplitOrientedDataFrame(TypedDict):
    index: List[Union[str, int, float]]
    columns: List[str]
    data: List[List[Union[str, int, float]]]


class DataSetEntry(BaseModel):
    monomers: List[Monomer]
    mn: float = Field(..., description="Number average molecular weight.")
    mw: float = Field(..., description="Weight average molecular weight.")
    architecture: Literal["linear"] = Field(
        "linear", description="Architecture of the polymer."
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
