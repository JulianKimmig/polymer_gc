import pytest
import numpy as np
import networkx as nx
from polymer_gc.core.monomer import Monomer
from polymer_gc.graph.datamodel import PolyGraph, PolyGraphEnsemble, make_nx_polygraph
from unittest.mock import MagicMock, patch


@pytest.fixture
def sample_nodes_edges_monomers(monomers_list):
    nodes = np.array([0, 1, 0, 2])
    edges = np.array([[0, 1], [1, 2], [2, 3]])
    return nodes, edges, monomers_list


@pytest.fixture
def sample_polygraph(sample_nodes_edges_monomers):
    nodes, edges, monomers = sample_nodes_edges_monomers
    return PolyGraph(nodes, edges, monomers)


def test_polygraph_creation(sample_polygraph, sample_nodes_edges_monomers):
    nodes, edges, monomers = sample_nodes_edges_monomers
    pg = sample_polygraph

    np.testing.assert_array_equal(pg.nodes, nodes)
    np.testing.assert_array_equal(pg.edges, edges)
    assert pg.monomers == monomers
    assert len(pg) == len(nodes)


def test_polygraph_nodes_edges_readonly_internal_and_copy_writable(
    sample_nodes_edges_monomers
):
    nodes_orig, edges_orig, monomers = sample_nodes_edges_monomers
    pg = PolyGraph(nodes_orig.copy(), edges_orig.copy(), monomers)

    # Internal arrays should be read-only
    with pytest.raises(
        ValueError
    ):  # NumPy raises ValueError when trying to write to read-only array
        pg._nodes[0] = 5
    with pytest.raises(ValueError):
        pg._edges[0, 0] = 5

    # Copies returned by properties should be writable
    nodes_copy = pg.nodes
    edges_copy = pg.edges
    nodes_copy[0] = 10  # Should not raise error
    edges_copy[0, 0] = 10  # Should not raise error

    # Original internal arrays should be unchanged
    np.testing.assert_array_equal(pg._nodes, nodes_orig)
    np.testing.assert_array_equal(pg._edges, edges_orig)


def test_polygraph_invalid_monomer_list_size(sample_nodes_edges_monomers):
    nodes, edges, monomers = sample_nodes_edges_monomers
    too_few_monomers = monomers[
        :2
    ]  # nodes has index 2, but monomers only up to index 1
    with pytest.raises(
        ValueError,
        match="The number of monomers must be at least the size of the graph",
    ):
        PolyGraph(nodes, edges, too_few_monomers)


def test_polygraph_iteration(sample_polygraph, sample_nodes_edges_monomers):
    nodes_orig, _, _ = sample_nodes_edges_monomers
    iterated_nodes = [n for n in sample_polygraph]
    np.testing.assert_array_equal(np.array(iterated_nodes), nodes_orig)

    assert list(sample_polygraph.iter_nodes()) == list(nodes_orig)
    np.testing.assert_array_equal(
        np.array(list(sample_polygraph.iter_edges())), sample_polygraph.edges
    )


def test_polygraph_nx_graph(sample_polygraph):
    nx_g = sample_polygraph.nx_graph
    assert isinstance(nx_g, nx.Graph)
    assert len(nx_g.nodes) == len(sample_polygraph.nodes)
    assert len(nx_g.edges) == len(sample_polygraph.edges)
    # Check caching
    assert sample_polygraph._nx_graph is nx_g
    assert sample_polygraph.nx_graph is nx_g


def test_polygraph_mass(sample_polygraph, monomers_list):
    expected_mass = (
        monomers_list[0].mass
        + monomers_list[1].mass
        + monomers_list[0].mass
        + monomers_list[2].mass
    )
    assert pytest.approx(sample_polygraph.mass) == expected_mass
    # Check caching
    assert sample_polygraph._mass is not None
    mass1 = sample_polygraph.mass
    assert sample_polygraph.mass == mass1  # Should return cached value


@patch("polymer_gc.graph.datamodel.plot_polygraph")
def test_polygraph_ipython_display(mock_plot_polygraph, sample_polygraph):
    mock_fig = MagicMock()
    mock_plot_polygraph.return_value = mock_fig

    fig = sample_polygraph._ipython_display_()

    mock_plot_polygraph.assert_called_once_with(sample_polygraph.nx_graph)
    mock_fig.show.assert_called_once()
    assert fig is mock_fig


# --- PolyGraphEnsemble Tests ---


@pytest.fixture
def sample_polygraph_ensemble(sample_polygraph, monomers_list):
    # Create another graph with same monomers but different structure
    nodes2 = np.array([1, 2, 0])
    edges2 = np.array([[0, 1], [1, 2]])
    pg2 = PolyGraph(nodes2, edges2, monomers_list)
    return PolyGraphEnsemble(graphs=[sample_polygraph, pg2])


def test_polygraph_ensemble_creation(sample_polygraph_ensemble, sample_polygraph):
    ensemble = sample_polygraph_ensemble
    assert len(ensemble) == 2
    assert ensemble.graphs[0] == sample_polygraph
    assert ensemble.monomers == sample_polygraph.monomers


def test_polygraph_ensemble_from_lists(sample_nodes_edges_monomers):
    nodes1, edges1, monomers = sample_nodes_edges_monomers
    nodes2 = np.array([1, 0])
    edges2 = np.array([[0, 1]])

    ensemble = PolyGraphEnsemble.from_lists(
        nodes=[nodes1, nodes2], edges=[edges1, edges2], monomers=monomers
    )
    assert len(ensemble) == 2
    assert ensemble.monomers == monomers
    np.testing.assert_array_equal(ensemble.graphs[0].nodes, nodes1)
    np.testing.assert_array_equal(ensemble.graphs[1].nodes, nodes2)


def test_polygraph_ensemble_inconsistent_monomers_count(
    sample_polygraph, monomers_list
):
    monomers_alt = monomers_list + [Monomer("EXTRA", 10.0)]
    pg_alt_monomers = PolyGraph(np.array([0, 1]), np.array([[0, 1]]), monomers_alt)
    with pytest.raises(
        ValueError, match="All graphs must have the same number of monomers."
    ):
        PolyGraphEnsemble(graphs=[sample_polygraph, pg_alt_monomers])


def test_polygraph_ensemble_inconsistent_monomers_objects(
    sample_polygraph, monomers_list
):
    m_new_sty = Monomer(
        "STY", 104.15
    )  # Same name/mass, but different object for this test
    monomers_alt = [monomers_list[0], m_new_sty, monomers_list[2]]
    pg_alt_monomers = PolyGraph(
        sample_polygraph.nodes, sample_polygraph.edges, monomers_alt
    )
    with pytest.raises(ValueError, match="All graphs must have the same monomers."):
        PolyGraphEnsemble(graphs=[sample_polygraph, pg_alt_monomers])


def test_polygraph_ensemble_iteration(sample_polygraph_ensemble):
    count = 0
    for g in sample_polygraph_ensemble:
        assert isinstance(g, PolyGraph)
        count += 1
    assert count == len(sample_polygraph_ensemble)


def test_polygraph_ensemble_masses(sample_polygraph_ensemble):
    masses = sample_polygraph_ensemble.masses
    assert len(masses) == len(sample_polygraph_ensemble)
    assert pytest.approx(masses[0]) == sample_polygraph_ensemble.graphs[0].mass
    assert pytest.approx(masses[1]) == sample_polygraph_ensemble.graphs[1].mass


def test_polygraph_ensemble_mass_distribution(sample_polygraph_ensemble):
    hist, edges = sample_polygraph_ensemble.mass_distribution(bins=5)
    assert isinstance(hist, np.ndarray)
    assert isinstance(edges, np.ndarray)
    assert len(hist) == 5
    assert len(edges) == 6
    assert np.sum(hist) == len(sample_polygraph_ensemble)  # Total count in histogram


def test_make_nx_polygraph(sample_polygraph):
    # Tested indirectly via PolyGraph.nx_graph, but can test directly too
    g_nx = make_nx_polygraph(sample_polygraph)
    assert len(g_nx.nodes) == len(sample_polygraph.nodes)
    for i, data in g_nx.nodes(data=True):
        assert data["idx"] == sample_polygraph.nodes[i]
    assert g_nx.number_of_edges() == sample_polygraph.edges.shape[0]
