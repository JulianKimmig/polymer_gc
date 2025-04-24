from typing import Dict, Union
from dataclasses import dataclass, field
from rdkit import Chem
from .exceptions import AttachmentPointError


@dataclass(slots=True, frozen=True)
class Monomer:
    """Lightweight description of a repeat unit.

    Parameters
    ----------
    name
        Short human‑readable identifier (e.g. *STY* for styrene).
    smiles
        Canonical SMILES string **with** dummy atoms labelled ``[\*]``
        marking attachment points (RDKit convention).
    attachment_points
        Mapping *label → dummy atom index* (0‑based).  Labels are arbitrary
        strings ("head", "tail", "branch1" …) understood by topology
        generators.
    meta
        Optional user metadata (MW, vendor, CAS…).
    """

    name: str
    smiles: str
    attachment_points: Dict[str, int]
    meta: Dict[str, Union[str, float, int]] = field(default_factory=dict)

    # RDKit molecules are cached after first build for performance.
    _rdkit_mol: Chem.Mol | None = field(default=None, init=False, repr=False)

    def to_rdkit(self) -> Chem.Mol:
        """Return an RDKit *editable* molecule with attachment atoms tagged."""
        if self._rdkit_mol is None:
            mol = Chem.MolFromSmiles(self.smiles, sanitize=True)
            if mol is None:
                raise ValueError(
                    f"Invalid SMILES for monomer {self.name}: {self.smiles}"
                )
            object.__setattr__(self, "_rdkit_mol", mol)  # bypass frozen
        return Chem.Mol(self._rdkit_mol)  # return *copy* so edits are safe

    def __hash__(self) -> int:  # allows set/dict usage
        return hash((self.name, self.smiles))

    def __iter__(self):  # unpack as tuple
        yield self.name
        yield self.smiles
        yield self.attachment_points


def _label_from_index(mon: Monomer, idx: int) -> str:
    for label, atm_idx in mon.attachment_points.items():
        if atm_idx == idx:
            return label
    raise AttachmentPointError(
        f"Atom index {idx} not in attachment_points of {mon.name}"
    )
