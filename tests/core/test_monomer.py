import pytest
from polymer_gc.core.monomer import Monomer


def test_monomer_creation():
    m = Monomer(name="TestMon", mass=50.0)
    assert m.name == "TestMon"
    assert m.mass == 50.0


def test_monomer_properties(simple_monomer):
    assert simple_monomer.name == "MMA"
    assert simple_monomer.mass == 100.12


def test_monomer_mass_validation():
    with pytest.raises(ValueError, match="Mass must be a positive number."):
        Monomer(name="InvalidMass", mass=0)
    with pytest.raises(ValueError, match="Mass must be a positive number."):
        Monomer(name="InvalidMassNegative", mass=-10.0)


def test_monomer_private_attributes():
    m = Monomer(name="Test", mass=1.0)
    assert m._name == "Test"
    assert m._mass == 1.0
    # Try to modify (shouldn't be directly settable if truly private, but Python allows it)
    # Properties are the correct way to access

    with pytest.raises(AttributeError):
        m.name = "NewName"
    with pytest.raises(AttributeError):
        m.mass = 2.0
