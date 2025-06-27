from typing import Optional, List
from sqlmodel import Column, JSON, Field, Relationship, UniqueConstraint
from pydantic import model_validator
from rdkit import Chem
from rdkit.Chem import Descriptors
from polymer_gc.embeddings import DEFAULT_EMBEDDINGS
from polymer_gc.data.database import SessionRegistry
from .core import Base
import numpy as np

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


class SQLStructureEmbedding(Base, table=True):
    __tablename__ = "structure_embeddings"
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(max_length=255)
    value: List[float] = Field(default_factory=list, sa_column=Column(JSON))
    structure_id: Optional[int] = Field(default=None, foreign_key="structures.id")
    structure: Optional["SQLStructureModel"] = Relationship(back_populates="embeddings")

    # name and structure_id must be unique together
    __table_args__ = (
        UniqueConstraint("name", "structure_id", name="uix_name_structure_id"),
    )


class SQLStructureModel(Base, table=True):
    __tablename__ = "structures"
    id: Optional[int] = Field(default=None, primary_key=True)
    smiles: str = Field(max_length=1000, unique=True)
    name: Optional[str] = Field(default=None, max_length=255)
    mass: float = Field(default=None)
    embeddings: List["SQLStructureEmbedding"] = Relationship(back_populates="structure")

    @classmethod
    def fill_values(cls, **kwargs):
        kwargs = super().fill_values(**kwargs)
        if "mass" not in kwargs:
            kwargs["mass"] = calculate_mass_from_smiles(kwargs["smiles"])
        return kwargs

    @property
    def name_or_smiles(self):
        return self.name or self.smiles
    
    @classmethod
    def batch_get_embedding(cls, structs, embedding_name: str):
        smiles = [struct.smiles for struct in structs]
        with SessionRegistry.get_session() as session:
            embeddings = DEFAULT_EMBEDDINGS[embedding_name].batch_calculate_embedding(smiles).tolist()
            emb= [
                SQLStructureEmbedding.get_or_create(
                name=embedding_name,
                structure_id=struct.id,
                set_kwargs={
                    "value": embedding,
                    "structure": struct,
                },
                commit=False
            )
                for struct, embedding in zip(structs, embeddings)]
            session.commit()
        return embeddings

    def get_embedding(self, embedding_name: str, create_if_not_exists: bool = True):
        for embedding in self.embeddings:
            if embedding.name == embedding_name:
                return embedding.value
        if not create_if_not_exists:
            raise ValueError(
                f"Embedding {embedding_name} not found for structure {self.smiles}"
            )

        if embedding_name not in DEFAULT_EMBEDDINGS:
            raise ValueError(
                f"Embedding {embedding_name} not found in DEFAULT_EMBEDDINGS"
            )
        embedding = DEFAULT_EMBEDDINGS[embedding_name].calculate_embedding(self.smiles)
        if isinstance(embedding, np.ndarray):
            embedding =  embedding.tolist()
        emb = SQLStructureEmbedding.get_or_create(
            name=embedding_name,
            structure_id=self.id,
            set_kwargs={
                "value": embedding,
                "structure": self,
            },
        )
        self.embeddings.append(emb)
        return emb.value
