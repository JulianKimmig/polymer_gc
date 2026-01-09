from abc import ABC, abstractmethod
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, MolFromSmiles
import numpy as np
from pydantic import BaseModel, Field, model_validator, field_validator
from typing import List, Optional, Any


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
    def calculate_embedding(cls, smiles: str) -> List[float]:
        pass

    @classmethod
    def batch_calculate_embedding(
        cls,
        smiles: List[str],
    ) -> List[List[float]]:
        """
        Calculate embeddings for a batch of smiles.
        """
        return [cls.calculate_embedding(structure) for structure in smiles]


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


class random_64(StructureEmbedding):
    @classmethod
    def calculate_embedding(cls, smiles: str) -> List[float]:
        """
        Generate a random 64-dimensional embedding.
        """
        return np.random.rand(64).tolist()

    @classmethod
    def batch_calculate_embedding(
        cls,
        smiles: List[str],
    ) -> List[List[float]]:
        """
        Generate random 64-dimensional embeddings for a batch of smiles.
        """
        return np.random.rand(len(smiles), 64).tolist()


register_embedding(random_64, "random_64")

try:
    from sentence_transformers import SentenceTransformer

    class PolyBERTEmbedding(StructureEmbedding):
        """
        PolyBERT embedding using SentenceTransformer.
        """

        @classmethod
        def calculate_embedding(cls, smiles: str) -> List[float]:
            polyBERT = SentenceTransformer("kuelumbus/polyBERT")
            return polyBERT.encode([smiles])[0]

        @classmethod
        def batch_calculate_embedding(
            cls,
            smiles: List[str],
        ) -> List[List[float]]:
            print("batch calculating polyBERT embedding")
            polyBERT = SentenceTransformer("kuelumbus/polyBERT")
            return polyBERT.encode([s for s in smiles]).tolist()

    register_embedding(PolyBERTEmbedding, "PolyBERT")
except ImportError:
    pass


fpgen = AllChem.GetRDKitFPGenerator()


class RDKitFP(StructureEmbedding):
    @classmethod
    def calculate_embedding(cls, smiles: str) -> List[float]:
        return np.array(fpgen.GetFingerprint(MolFromSmiles(smiles))).tolist()

    @classmethod
    def batch_calculate_embedding(
        cls,
        smiles: List[str],
    ) -> List[List[float]]:
        return [cls.calculate_embedding(s) for s in smiles]


register_embedding(RDKitFP, "RDKitFP")
