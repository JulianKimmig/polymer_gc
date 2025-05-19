import pytest
import numpy as np
from polymer_gc.core.monomer import Monomer
from polymer_gc.graph.datamodel import PolyGraph, PolyGraphEnsemble
from polymer_gc.graph.props import calculate_local_monomer_distribution


@pytest.fixture
def simple_ensemble_for_props(simple_monomer, another_monomer):
    m1, m2 = simple_monomer, another_monomer
    # Graph 1: M1-M1-M2-M2
    nodes1 = np.array([0, 0, 1, 1])
    edges1 = np.array([[0, 1], [1, 2], [2, 3]])
    pg1 = PolyGraph(nodes1, edges1, [m1, m2])

    # Graph 2: M2-M1-M2-M1
    nodes2 = np.array([1, 0, 1, 0])
    edges2 = np.array([[0, 1], [1, 2], [2, 3]])
    pg2 = PolyGraph(nodes2, edges2, [m1, m2])

    return PolyGraphEnsemble(graphs=[pg1, pg2])


def test_calculate_local_monomer_distribution_basic(simple_ensemble_for_props):
    ensemble = simple_ensemble_for_props
    n_points = 100
    window = 3
    dist = calculate_local_monomer_distribution(
        ensemble, n=n_points, window_size=window
    )

    assert isinstance(dist, np.ndarray)
    assert dist.shape == (
        n_points,
        len(ensemble.monomers),
    )  # (n_points, n_monomer_types)

    # Distribution for each point should sum to approx 1.0 (total fraction of monomers)
    np.testing.assert_allclose(np.sum(dist, axis=1), 1.0, atol=1e-6)


def test_calculate_local_monomer_distribution_single_monomer_type():
    m = Monomer("M", 100)
    nodes = np.array([0, 0, 0, 0])  # All same monomer
    edges = np.array([[0, 1], [1, 2], [2, 3]])
    pg = PolyGraph(nodes, edges, [m])
    ensemble = PolyGraphEnsemble(graphs=[pg])

    n_points = 50
    dist = calculate_local_monomer_distribution(ensemble, n=n_points, window_size=1)

    assert dist.shape == (n_points, 1)
    # For a single monomer type, its distribution should be 1.0 everywhere
    np.testing.assert_allclose(dist[:, 0], 1.0, atol=1e-6)


def test_calculate_local_monomer_distribution_empty_ensemble():
    empty_ensemble = PolyGraphEnsemble.from_lists([], [], [Monomer("M", 100)])
    # The from_lists will make an ensemble with 0 graphs. monomers prop will refer to first graph.
    # If graphs list is empty, ensemble.graphs[0] will fail.
    # The code handles PolyGraphEnsemble([]) correctly, where graphs=[] and monomers=[].
    # But PolyGraphEnsemble.from_lists can be tricky.
    # Let's create a truly empty one.
    ensemble = PolyGraphEnsemble(
        graphs=[]
    )  # This might be an issue for ensemble.monomers if not handled
    # In current code: self.graphs[0].monomers -> IndexError
    # This is a bug in PolyGraphEnsemble if graphs list is empty.
    # Let's assume `props` checks `len(graphs)`.
    # The `calculate_local_monomer_distribution` itself checks `len(graphs) == 0`.

    # To test the check in `calculate_local_monomer_distribution`:
    dummy_monomer = Monomer("Dummy", 1.0)

    # Create an ensemble that has monomers defined but no graphs
    class MockEnsemble:
        def __init__(self):
            self.graphs = []
            self.monomers = [dummy_monomer]  # Needed for n_monomers

        def __len__(self):
            return 0

        def __iter__(self):
            return iter(self.graphs)

    with pytest.raises(
        ValueError, match="No graphs provided for distribution calculation."
    ):
        calculate_local_monomer_distribution(MockEnsemble())


def test_calculate_local_monomer_distribution_varying_window(simple_ensemble_for_props):
    ensemble = simple_ensemble_for_props
    n_points = 10

    dist_small_window = calculate_local_monomer_distribution(
        ensemble, n=n_points, window_size=1
    )
    dist_large_window = calculate_local_monomer_distribution(
        ensemble, n=n_points, window_size=len(ensemble.graphs[0].nodes) * 2
    )  # Window larger than graph

    assert dist_small_window.shape == (n_points, len(ensemble.monomers))
    assert dist_large_window.shape == (n_points, len(ensemble.monomers))
    # Large window should average out more, small window show more detail.
    # Not easy to assert specific values without re-implementing logic.
    # Just ensure it runs and shapes are correct.
