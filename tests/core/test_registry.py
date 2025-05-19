import pytest
import json
import pathlib
from polymer_gc.core.monomer import Monomer
from polymer_gc.core.registry import MonomerRegistry


def test_add_monomer(simple_monomer):
    MonomerRegistry.add(simple_monomer)
    assert MonomerRegistry.get(simple_monomer.name) == simple_monomer
    assert simple_monomer.name in MonomerRegistry.list()


def test_add_monomer_overwrite_false_new(simple_monomer, another_monomer):
    MonomerRegistry.add(simple_monomer)
    MonomerRegistry.add(another_monomer, overwrite=False)  # New monomer
    assert MonomerRegistry.get(another_monomer.name) == another_monomer


def test_add_monomer_overwrite_false_existing_different_smiles_bug(simple_monomer):
    """
    Tests the behavior when adding a monomer with the same name if overwrite=False.
    The original code has a line `if cls._REGISTRY[mon.name].smiles != mon.smiles:`.
    Since Monomer class does not have a `smiles` attribute, this will raise an AttributeError.
    This test confirms that AttributeError is raised.
    """
    MonomerRegistry.add(simple_monomer)
    m_same_name_diff_mass = Monomer(
        name=simple_monomer.name, mass=simple_monomer.mass + 10
    )

    with pytest.raises(
        AttributeError
    ):  # Expecting AttributeError due to missing 'smiles'
        MonomerRegistry.add(m_same_name_diff_mass, overwrite=False)

    # The registry should still contain the original monomer due to the error
    assert MonomerRegistry.get(simple_monomer.name).mass == simple_monomer.mass


def test_add_monomer_overwrite_true(simple_monomer):
    MonomerRegistry.add(simple_monomer)
    m_updated = Monomer(name=simple_monomer.name, mass=200.0)
    MonomerRegistry.add(m_updated, overwrite=True)
    retrieved = MonomerRegistry.get(simple_monomer.name)
    assert retrieved.mass == 200.0
    assert retrieved == m_updated  # Checks if it's the new object


def test_get_monomer_exists(simple_monomer, registered_monomers):
    retrieved = MonomerRegistry.get(simple_monomer.name)
    assert retrieved == simple_monomer


def test_get_monomer_not_exists():
    with pytest.raises(KeyError):
        MonomerRegistry.get("NONEXISTENT")


def test_list_monomers(registered_monomers, simple_monomer, another_monomer):
    names = MonomerRegistry.list()
    expected_names = sorted([simple_monomer.name, another_monomer.name])
    assert names == expected_names


def test_add_search_path_valid(tmp_path):
    MonomerRegistry.add_search_path(tmp_path)
    assert tmp_path in MonomerRegistry._SEARCH_PATHS


def test_add_search_path_invalid():
    with pytest.raises(FileNotFoundError):
        MonomerRegistry.add_search_path("non_existent_path_for_sure")


def test_try_autoload(tmp_path, simple_monomer):
    # Prepare a JSON file
    monomer_data = {"name": simple_monomer.name, "mass": simple_monomer.mass}
    json_file = tmp_path / f"{simple_monomer.name}.json"
    with open(json_file, "w") as f:
        json.dump(monomer_data, f)

    MonomerRegistry.add_search_path(tmp_path)

    # Monomer should not be in registry yet
    assert simple_monomer.name not in MonomerRegistry._REGISTRY

    # Trigger autoload by trying to get it
    retrieved_monomer = MonomerRegistry.get(simple_monomer.name)

    assert retrieved_monomer is not None
    assert retrieved_monomer.name == simple_monomer.name
    assert retrieved_monomer.mass == simple_monomer.mass
    assert simple_monomer.name in MonomerRegistry._REGISTRY


def test_try_autoload_file_not_found(tmp_path):
    MonomerRegistry.add_search_path(tmp_path)  # Path exists, but file won't
    with pytest.raises(KeyError):
        MonomerRegistry.get("NONEXISTENT_JSON_MONOMER")


def test_get_monomer_after_failed_autoload(tmp_path):
    # Add a search path but ensure the specific monomer file doesn't exist
    MonomerRegistry.add_search_path(tmp_path)
    monomer_name_not_in_file = "NotInFile"

    # Attempt to get, which will try to autoload and fail silently for this file
    with pytest.raises(KeyError) as excinfo:
        MonomerRegistry.get(monomer_name_not_in_file)
    assert str(excinfo.value) == f"'{monomer_name_not_in_file}'"
    assert monomer_name_not_in_file not in MonomerRegistry._REGISTRY
