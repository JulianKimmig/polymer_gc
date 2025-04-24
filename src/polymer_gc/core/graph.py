from typing import Dict
from rdkit import Chem
import networkx as nx
from .monomer import Monomer, _label_from_index
from .exceptions import InvalidValenceError


class PolymerGraph:
    """Graph‑based representation of a *single* polymer chain/network.

    Internally we store a :class:`networkx.MultiGraph` where:
        • **Nodes** carry RDKit *Atom* objects plus bookkeeping fields.
        • **Edges** carry bond order, stereo, and origin metadata.

    Methods are *chemistry‑aware*: they respect valence when connecting
    monomers and will raise :class:`InvalidValenceError` if an illegal bond
    is attempted.
    """

    def __init__(self, *, name: str = "polymer", rng=None):
        self.name = name
        self.g: nx.MultiGraph = nx.MultiGraph(name=name)
        self._rng = rng
        # Mapping dummy atom → global node id for quick attachment
        self._open_sites: Dict[int, int] = {}
        self._next_node_id: int = 0

    # ---------------------------------------------------------------------
    # Low‑level utilities
    # ---------------------------------------------------------------------

    def _add_monomer(self, mon: Monomer) -> Dict[str, int]:
        """Add **all atoms** from *mon* as a disconnected fragment.

        Returns a mapping *attach_label → node_id* in the main graph so that
        caller can subsequently *connect* them.
        """
        self.g.add_node(
            len(self.g.nodes),
            type="monomer",
            monomer=mon,
            attachment_points=mon.attachment_points.copy(),
            open_sites=list(mon.attachment_points.values()),
        )

        mol = mon.to_rdkit()
        amap = {}
        for atom in mol.GetAtoms():
            new_id = self._next_node_id
            self.g.add_node(new_id, rd_atom=atom, element=atom.GetSymbol())
            if atom.GetIdx() in mon.attachment_points.values():
                # record mapping for later bonding
                label = _label_from_index(mon, atom.GetIdx())
                amap[label] = new_id
            self._next_node_id += 1

        # Add bonds internal to the monomer
        for bond in mol.GetBonds():
            a = bond.GetBeginAtomIdx()
            b = bond.GetEndAtomIdx()
            self.g.add_edge(a, b, order=bond.GetBondTypeAsDouble())

        return amap

    def connect(self, node_a: int, node_b: int, *, order: int = 1):
        """Create a bond between **existing** nodes *node_a* and *node_b*."""

        # Simple valence check: RDKit provides helper
        def _remaining_valence(node):
            atom: Chem.Atom = self.g.nodes[node]["rd_atom"]
            return atom.GetTotalValence() - atom.GetTotalDegree()

        if _remaining_valence(node_a) <= 0:
            raise InvalidValenceError(
                f"Cannot connect – valence exhausted for node_a {self.g.nodes[node_a]}."
            )
        if _remaining_valence(node_b) <= 0:
            raise InvalidValenceError(
                f"Cannot connect – valence exhausted for node_b {self.g.nodes[node_b]}."
            )
        self.g.add_edge(node_a, node_b, order=order)

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def to_networkx(self) -> nx.MultiGraph:
        """Return the raw NetworkX graph (read‑only)."""
        return self.g.copy()

    def to_smiles(self) -> str:
        """Generate a canonical SMILES for the entire polymer (single snapshot).

        *Warning*: For large networks this can be slow; future versions may
        rely on external toolkits for macromolecular SMILES (BigSMILES).
        """
        # Convert networkx → RDKit editable Mol
        mol = Chem.RWMol()
        node_map = {}
        for n, data in self.g.nodes(data=True):
            atom: Chem.Atom = data["rd_atom"]
            new_idx = mol.AddAtom(atom)
            node_map[n] = new_idx
        for u, v, edata in self.g.edges(data=True):
            mol.AddBond(node_map[u], node_map[v], _bondtype_from_order(edata["order"]))
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)


def _bondtype_from_order(order: int) -> Chem.BondType:
    mapping = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
    }
    return mapping.get(order, Chem.BondType.SINGLE)
