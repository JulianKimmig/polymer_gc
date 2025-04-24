from typing import Dict, Union, List
import json
import pathlib
from .monomer import Monomer


class MonomerRegistry:
    """In‑memory/global registry for :class:`Monomer` objects.

    Supports *lazy* loading from JSON/YAML files.  The JSON schema is simply:
    ``[{"name": "STY", "smiles": "[*]c1cc...", "attachment_points": {"head":0,"tail":1}}]``
    """

    _REGISTRY: Dict[str, Monomer] = {}
    _SEARCH_PATHS: List[pathlib.Path] = []  # extra dirs added by user

    # ------------------------------------------------------------
    # Read / write API
    # ------------------------------------------------------------
    @classmethod
    def add(cls, mon: Monomer, *, overwrite: bool = False) -> None:
        if not overwrite and mon.name in cls._REGISTRY:
            if cls._REGISTRY[mon.name].smiles != mon.smiles:
                raise KeyError(f"Monomer '{mon.name}' already registered.")
        cls._REGISTRY[mon.name] = mon

    @classmethod
    def get(cls, name: str) -> Monomer:
        try:
            return cls._REGISTRY[name]
        except KeyError as exc:
            cls._try_autoload(name)
            if name in cls._REGISTRY:
                return cls._REGISTRY[name]
            raise exc

    @classmethod
    def list(cls) -> List[str]:
        return sorted(cls._REGISTRY)

    # ------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------
    @classmethod
    def add_search_path(cls, path: Union[str, pathlib.Path]):
        p = pathlib.Path(path)
        if not p.exists():
            raise FileNotFoundError(p)
        cls._SEARCH_PATHS.append(p)

    @classmethod
    def _try_autoload(cls, name: str):
        for directory in cls._SEARCH_PATHS:
            json_file = directory / f"{name}.json"
            if json_file.exists():
                with open(json_file) as fh:
                    data = json.load(fh)
                mon = Monomer(**data)
                cls._REGISTRY[name] = mon
                return
