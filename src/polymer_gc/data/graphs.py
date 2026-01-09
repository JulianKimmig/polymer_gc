from sqlmodel import Column, JSON, Field, Integer, Relationship, select, and_, String
from .basemodels.core import Base
import pandas as pd
from ..sec import SimSEC
from .database import SessionRegistry
from typing import List, Tuple, Optional, Dict, Any
from pydantic import model_validator
import networkx as nx
import json


class ReusableGraphEntry(Base, table=True):
    """
    This class is used to store reusable graph entries.
    The graph is stored as a list of nodes and edges.
    The nodes are stored as a list of monomer names.
    The edges are stored as a list of tuples, where each tuple represents an edge between two nodes.
    The n_nodes and n_edges are stored as integers.
    The name is stored as a string.
    The description is stored as a string.

    Example entry:
    id	nodes	edges	n_nodes	n_edges	name	description
    8199	["A", "A", "A", "A", "A"]	[[0, 3], [1, 2], [1, 4], [3, 4]]	5	4	branching_homopolymer	{"A": "backbone", "monomer_ratios": {"A": 1.0}, "reactivity_ratios": {"A": 1}}


    """

    __tablename__ = "graph_entries"

    id: Optional[int] = Field(default=None, primary_key=True)
    nodes: List[int] = Field(sa_column=Column(JSON))
    edges: List[Tuple[int, int]] = Field(sa_column=Column(JSON))
    n_nodes: int = Field(default=None)
    n_edges: int = Field(default=None)
    name: str = Field(default=None)
    description: str = Field(default="{}", sa_column=Column(String))

    @classmethod
    def get_possible_entries(
        cls,
        n_nodes_min: Optional[int] = None,
        n_nodes_max: Optional[int] = None,
        n_edges_min: Optional[int] = None,
        n_edges_max: Optional[int] = None,
        name: Optional[str] = None,
        exclude_entries: Optional[List[int]] = None,
        description: Optional[str] = None,
        n: Optional[int] = None,
    ):
        session = SessionRegistry.get_session()

        # Use a list to collect filter conditions
        filters = []

        if n_nodes_min is not None:
            filters.append(cls.n_nodes >= int(n_nodes_min))
        if n_nodes_max is not None:
            filters.append(cls.n_nodes <= int(n_nodes_max))
        if n_edges_min is not None:
            filters.append(cls.n_edges >= int(n_edges_min))
        if n_edges_max is not None:
            filters.append(cls.n_edges <= int(n_edges_max))
        if name is not None:
            filters.append(cls.name == str(name))
        if exclude_entries:  # You can check for truthiness directly
            filters.append(cls.id.not_in(list(exclude_entries)))
        if description is not None:
            filters.append(cls.description == str(description))

        statement = select(cls)
        # If the filters list is not empty, apply them all with AND
        if filters:
            statement = statement.where(and_(*filters))

        if n is not None:
            statement = statement.limit(n)

        res = session.exec(statement).all()
        return res

    @classmethod
    def fill_values(cls, **data):
        """Automatically calculate n_nodes and n_edges from nodes and edges data"""
        data = super().fill_values(**data)
        if "n_nodes" not in data:
            data["n_nodes"] = len(data["nodes"])
        if "n_edges" not in data:
            data["n_edges"] = len(data["edges"])

        return data

    def to_nx(self):
        g = nx.Graph()
        g.add_nodes_from(
            [(i, {"monomer_type": self.nodes[i]}) for i in range(self.n_nodes)]
        )
        g.add_edges_from(self.edges)
        return g
