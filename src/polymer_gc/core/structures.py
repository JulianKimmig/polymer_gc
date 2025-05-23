from pydantic import (
    BaseModel,
    Field,
    field_validator,
    ConfigDict,
    model_validator,
    PrivateAttr,
)
from typing import List, Dict, Optional, Type, Any
from abc import ABC, abstractmethod
from rdkit import Chem
from rdkit.Chem import Descriptors


class StructureEmbedding(BaseModel, ABC):
    name: Optional[str] = Field(
        default=None,
        description="Name of the embedding. Automatically set based on class.",
    )
    value: List[float] = Field(..., description="Value of the embedding.")

    @field_validator("value")
    @classmethod
    def check_value_length(cls, value):
        if len(value) == 0:
            raise ValueError("Embedding value must not be empty.")
        return value

    @classmethod
    def default_name(cls) -> str:
        """
        Generate a default name for the embedding based on the class name.
        """
        return cls.__module__ + "." + cls.__name__

    @model_validator(mode="before")
    @classmethod
    def _set_name(cls, data: Any) -> Any:
        auto_name = cls.default_name()
        user_name = data.get("name")

        if user_name and user_name != auto_name:
            raise ValueError(
                f"Cannot override embedding name. Expected: '{auto_name}', got: '{user_name}'"
            )

        data["name"] = auto_name
        return data

    @classmethod
    @abstractmethod
    def calculate_embedding(cls, structure: "StructureModel") -> List[float]:
        pass

    @classmethod
    def batch_calculate_embedding(
        cls,
        structures: List["StructureModel"],
    ) -> List[List[float]]:
        """
        Calculate embeddings for a batch of structures.
        """
        return [cls.calculate_embedding(structure) for structure in structures]


DEFAULT_EMBEDDINGS = {}


def register_embedding(cls, alias: Optional[str] = None):
    """
    Register a new embedding class.
    """
    if not issubclass(cls, StructureEmbedding):
        raise ValueError(f"{cls} is not a subclass of StructureEmbedding.")

    if alias is None:
        alias = cls.default_name()

    if alias in DEFAULT_EMBEDDINGS:
        raise ValueError(f"Embedding with name '{alias}' already registered.")

    DEFAULT_EMBEDDINGS[alias] = cls
    DEFAULT_EMBEDDINGS[cls.default_name()] = cls
    return cls


try:
    from sentence_transformers import SentenceTransformer

    class PolyBERTEmbedding(StructureEmbedding):
        """
        PolyBERT embedding using SentenceTransformer.
        """

        @classmethod
        def calculate_embedding(cls, structure: "StructureModel") -> List[float]:
            if not structure.smiles:
                raise ValueError("SMILES string is required for PolyBERT embedding.")
            polyBERT = SentenceTransformer("kuelumbus/polyBERT")
            return polyBERT.encode([structure.smiles])[0]

        @classmethod
        def batch_calculate_embedding(
            cls,
            structures: List["StructureModel"],
        ) -> List[List[float]]:
            if not all(s.smiles for s in structures):
                raise ValueError(
                    "All structures must have a SMILES string for PolyBERT embedding."
                )
            polyBERT = SentenceTransformer("kuelumbus/polyBERT")
            return polyBERT.encode([s.smiles for s in structures])

    register_embedding(PolyBERTEmbedding, "PolyBERT")
except ImportError:
    pass


def calculate_mass_from_smiles(smiles: str) -> float:
    """
    Calculate the mass of a polymer from its SMILES representation.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    # Use RDKit to calculate the molecular weight
    mass = Descriptors.MolWt(mol)
    return mass


class StructureModel(BaseModel):
    smiles: str = Field(None, description="SMILES representation of the structure.")
    name: Optional[str] = Field(None, description="Name of the structure.")

    embeddings: Dict[str, StructureEmbedding] = Field(
        ...,
        description="Embeddings for the structure, can be used for various ML tasks.",
        default_factory=dict,
        exclude=True,
    )

    mass: float = Field(..., description="Mass of the monomer.")

    @field_validator("mass")
    @classmethod
    def check_mass_positive(cls, value):
        if value <= 0:
            raise ValueError("Mass must be a positive number.")
        return value

    @model_validator(mode="before")
    @classmethod
    def fill_mass_if_needed(cls, data: dict):
        mass = data.get("mass")
        smiles = data.get("smiles")
        if mass is None:
            if not smiles:
                raise ValueError("Either 'mass' or 'smiles' must be provided.")
            data["mass"] = calculate_mass_from_smiles(smiles)
        return data

    def add_embedding(
        self,
        name: str,
        embedding_cls: Optional[Type[StructureEmbedding]] = None,
        value: Optional[List[float]] = None,
    ):
        """
        Add an embedding to the structure.
        """
        if embedding_cls is None:
            embedding_cls = DEFAULT_EMBEDDINGS.get(name)
            if embedding_cls is None:
                raise ValueError(f"Embedding class for {name} not found.")
        if not issubclass(embedding_cls, StructureEmbedding):
            raise ValueError(
                f"{embedding_cls} is not a subclass of StructureEmbedding."
            )

        if value is None:
            value = embedding_cls.calculate_embedding(self)
        self.embeddings[name] = embedding_cls(value=value)

    # Tell Python how to compare/ hash them
    def __hash__(self) -> int:  # allows use as dict keys
        return hash(self.smiles)

    def __eq__(self, other) -> bool:
        return isinstance(other, StructureModel) and self.smiles == other.smiles

    @classmethod
    def add_batch_embeddings(
        cls,
        structures: List["StructureModel"],
        name: str,
        embedding_cls: Optional[Type[StructureEmbedding]] = None,
        values: Optional[List[Optional[List[float]]]] = None,
    ):
        """
        Add a batch of embeddings to the structures.
        """
        # make sure all structures are of type cls
        if not all(isinstance(s, cls) for s in structures):
            raise ValueError("All structures must be of the same type.")
        if embedding_cls is None:
            embedding_cls = DEFAULT_EMBEDDINGS.get(name)
            if embedding_cls is None:
                raise ValueError(f"Embedding class for {name} not found.")
        if not issubclass(embedding_cls, StructureEmbedding):
            raise ValueError(
                f"{embedding_cls} is not a subclass of StructureEmbedding."
            )

        if values is None:
            values = [None] * len(structures)

        if len(structures) != len(values):
            raise ValueError(
                f"Number of structures ({len(structures)}) must match number of values ({len(values)})."
            )
        missing_values = [i for i, v in enumerate(values) if v is None]

        filled_embeddings = embedding_cls.batch_calculate_embedding(
            structures=[structures[i] for i in missing_values],
        )

        for i, v in zip(missing_values, filled_embeddings):
            values[i] = v

        for structure, value in zip(structures, values):
            structure.add_embedding(
                name=name,
                embedding_cls=embedding_cls,
                value=value,
            )
