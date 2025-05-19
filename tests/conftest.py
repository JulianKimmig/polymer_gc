import pytest
import numpy as np
from polymer_gc.core.registry import MonomerRegistry
from polymer_gc.core.monomer import Monomer


@pytest.fixture(autouse=True)
def clear_monomer_registry():
    """Clears the MonomerRegistry before and after each test."""
    MonomerRegistry._REGISTRY.clear()
    MonomerRegistry._SEARCH_PATHS.clear()
    yield
    MonomerRegistry._REGISTRY.clear()
    MonomerRegistry._SEARCH_PATHS.clear()


@pytest.fixture
def simple_monomer():
    return Monomer(name="MMA", mass=100.12)


@pytest.fixture
def another_monomer():
    return Monomer(name="STY", mass=104.15)


@pytest.fixture
def yet_another_monomer():
    return Monomer(name="BA", mass=128.17)


@pytest.fixture
def monomers_list(simple_monomer, another_monomer, yet_another_monomer):
    return [simple_monomer, another_monomer, yet_another_monomer]


@pytest.fixture
def registered_monomers(simple_monomer, another_monomer):
    MonomerRegistry.add(simple_monomer)
    MonomerRegistry.add(another_monomer)
    return [simple_monomer, another_monomer]
