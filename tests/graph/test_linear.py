import pytest
import numpy as np
from polymer_gc.core.monomer import Monomer
from polymer_gc.graph.linear import (
    make_linear_polymer_graphs,
    make_linear_gradient_polymer,
    merge_linear_polymers_to_block,
    make_unique_monomers,
    sigmoid,
    norm_sigmoid,
    default_gradient_sigmoid,
)
from polymer_gc.graph.datamodel import PolyGraphEnsemble, PolyGraph


# --- Test make_unique_monomers ---
def test_make_unique_monomers_no_duplicates(simple_monomer, another_monomer):
    ex_monomers = [simple_monomer, another_monomer]
    created_nodes_list = [np.array([0, 1, 0])]

    remapped_nodes, unique_list = make_unique_monomers(ex_monomers, created_nodes_list)

    assert unique_list == [simple_monomer, another_monomer]
    np.testing.assert_array_equal(remapped_nodes[0], np.array([0, 1, 0]))


def test_make_unique_monomers_with_duplicates(simple_monomer, another_monomer):
    # m1, m2, m1 (same object)
    ex_monomers = [simple_monomer, another_monomer, simple_monomer]
    created_nodes_list = [np.array([0, 1, 2]), np.array([2, 0])]
    # original nodes refer to: [m1, m2, m1], [m1, m1]

    remapped_nodes, unique_list = make_unique_monomers(ex_monomers, created_nodes_list)

    assert len(unique_list) == 2
    assert simple_monomer in unique_list
    assert another_monomer in unique_list

    # Expected: unique_list = [simple_monomer, another_monomer] (order might vary, map based)
    # Let's find their new indices
    idx_m1 = unique_list.index(simple_monomer)
    idx_m2 = unique_list.index(another_monomer)

    np.testing.assert_array_equal(remapped_nodes[0], np.array([idx_m1, idx_m2, idx_m1]))
    np.testing.assert_array_equal(remapped_nodes[1], np.array([idx_m1, idx_m1]))


def test_make_unique_monomers_empty_input():
    remapped_nodes, unique_list = make_unique_monomers([], [])
    assert unique_list == []
    assert remapped_nodes == []

    remapped_nodes, unique_list = make_unique_monomers([], [np.array([])])
    assert unique_list == []
    assert len(remapped_nodes) == 1
    np.testing.assert_array_equal(remapped_nodes[0], np.array([], dtype=int))


def test_make_unique_monomers_empty_nodes_list(simple_monomer):
    ex_monomers = [simple_monomer]
    created_nodes_list = [np.array([], dtype=int), np.array([], dtype=int)]

    remapped_nodes, unique_list = make_unique_monomers(ex_monomers, created_nodes_list)

    assert unique_list == []
    assert len(remapped_nodes) == 2
    np.testing.assert_array_equal(remapped_nodes[0], np.array([], dtype=int))
    np.testing.assert_array_equal(remapped_nodes[1], np.array([], dtype=int))


# --- Test sigmoid functions ---
def test_sigmoid():
    assert sigmoid(0, a=1, b=0) == 0.5
    assert sigmoid(0.5, a=1, b=0.5) == 0.5
    assert sigmoid(1e6) > 0.99


def test_norm_sigmoid():
    assert norm_sigmoid(0.5, a=10) == 0.5
    assert norm_sigmoid(0, a=10) < 0.1
    assert norm_sigmoid(1, a=10) > 0.9


def test_default_gradient_sigmoid():
    assert default_gradient_sigmoid(0.5) == 0.5


# --- Test make_linear_polymer_graphs ---
@pytest.fixture
def default_monomers():
    return [Monomer("M1", 100), Monomer("M2", 120)]


@pytest.fixture
def start_end_groups():
    start = Monomer("Start", 50)
    end = Monomer("End", 30)
    return start, end


def test_mlp_basic_float_M(default_monomers):
    ensemble = make_linear_polymer_graphs(
        M=1000, monomers=default_monomers, n=3, seed=42
    )
    assert isinstance(ensemble, PolyGraphEnsemble)
    assert len(ensemble) == 3
    for graph in ensemble:
        assert isinstance(graph, PolyGraph)
        assert graph.mass > 0  # Should be close to 1000
        assert len(graph.monomers) <= len(default_monomers)  # Unique monomers used


def test_mlp_list_M(default_monomers):
    target_masses = [500, 1000, 1500]
    ensemble = make_linear_polymer_graphs(
        M=target_masses, monomers=default_monomers, seed=42
    )
    assert len(ensemble) == len(target_masses)
    # n parameter should be ignored
    ensemble_n_ignored = make_linear_polymer_graphs(
        M=target_masses, monomers=default_monomers, n=10, seed=42
    )
    assert len(ensemble_n_ignored) == len(target_masses)


def test_mlp_np_array_M(default_monomers):
    target_masses = np.array([600.0, 1200.0])
    ensemble = make_linear_polymer_graphs(
        M=target_masses, monomers=default_monomers, seed=42
    )
    assert len(ensemble) == len(target_masses)


def test_mlp_with_start_end_groups(default_monomers, start_end_groups):
    start, end = start_end_groups
    ensemble = make_linear_polymer_graphs(
        M=1000, monomers=default_monomers, n=2, startgroup=start, endgroup=end, seed=42
    )
    assert len(ensemble) == 2
    for graph in ensemble:
        assert graph.mass > (start.mass + end.mass)
        # Check if start/end groups are part of the graph's monomers
        graph_monomer_names = [m.name for m in graph.monomers]
        assert start.name in graph_monomer_names
        assert end.name in graph_monomer_names
        # Check if start/end groups are at the ends of the node list (after remapping)
        start_idx_remapped = graph.monomers.index(start)
        end_idx_remapped = graph.monomers.index(end)

        assert (
            graph.nodes[0] == start_idx_remapped
            or graph.nodes[-1] == start_idx_remapped
        )
        assert graph.nodes[-1] == end_idx_remapped or graph.nodes[0] == end_idx_remapped
        if len(graph.nodes) > 1:  # if it's not just start+end
            assert graph.nodes[0] != graph.nodes[-1]


def test_mlp_with_rel_content(default_monomers):
    rel_content = [[0.1, 0.5, 0.9], [0.9, 0.5, 0.1]]  # M1 increases, M2 decreases
    ensemble = make_linear_polymer_graphs(
        M=2000, monomers=default_monomers, n=1, rel_content=rel_content, seed=42
    )
    assert len(ensemble) == 1
    # Hard to verify exact distribution, but runs without error


def test_mlp_mass_adjustment(default_monomers):
    # Target mass slightly above one monomer, but less than two min monomers
    # Min mass of default_monomers is 100.
    # Max allowed mass for target M=110 would be 110 + 100/2 = 160
    ensemble = make_linear_polymer_graphs(
        M=110, monomers=default_monomers, n=10, seed=42
    )
    for graph in ensemble:
        assert 0 < graph.mass <= 160  # Max allowed mass check
        if graph.mass > default_monomers[0].mass / 2:  # if any monomer is there
            assert len(graph.nodes) >= 1


def test_mlp_errors_M(default_monomers):
    with pytest.raises(ValueError, match="Target mass list M cannot be empty."):
        make_linear_polymer_graphs(M=[], monomers=default_monomers)
    with pytest.raises(ValueError, match="Target mass array M cannot be empty."):
        make_linear_polymer_graphs(M=np.array([]), monomers=default_monomers)
    with pytest.raises(
        TypeError, match="M must be a float or a list of floats or numpy array."
    ):
        make_linear_polymer_graphs(M="not a mass", monomers=default_monomers)
    with pytest.raises(
        ValueError,
        match="Target mass\\(es\\) must be greater than 0 after accounting for start/end groups.",
    ):
        make_linear_polymer_graphs(
            M=10, monomers=default_monomers, startgroup=Monomer("S", 50)
        )


def test_mlp_no_monomers_positive_mass_error(start_end_groups):
    start, end = start_end_groups
    with pytest.raises(
        ValueError,
        match="Cannot generate polymers with positive mass if no monomers are provided.",
    ):
        make_linear_polymer_graphs(
            M=start.mass + end.mass + 100, monomers=[], startgroup=start, endgroup=end
        )


def test_mlp_no_monomers_only_start_end(start_end_groups):
    start, end = start_end_groups
    target_mass = start.mass + end.mass

    with pytest.raises(
        ValueError,
        match="Cannot generate polymers with positive mass if no monomers are provided.",
    ):
        make_linear_polymer_graphs(
            M=target_mass + 1, monomers=[], startgroup=start, endgroup=end, seed=42
        )

    with pytest.raises(
        ValueError,
        match="Target mass.* must be greater than 0 after accounting for start/end groups.",
    ):
        make_linear_polymer_graphs(
            M=target_mass, monomers=[], startgroup=start, endgroup=end, seed=42
        )


def test_mlp_rel_content_zero_sum_fallback(default_monomers):
    # Create rel_content that sums to zero at some point, forcing fallback
    # This is tricky to construct perfectly for the interpolation, but test for no error
    # For example, if max_length = 1, and rel_content for that point is [0,0]
    # This usually means all monomers have 0 probability.
    # The code has a fallback `p = np.ones_like(p) / len(p) if len(p) > 0 else np.array([])`

    # Use M small enough that max_length is small
    # If M is small enough, max_length might be 1.
    # local_content[i] can be all zeros if _rel_content_processed is all zeros.
    rel_content_zeros = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    ensemble = make_linear_polymer_graphs(
        M=default_monomers[0].mass * 1.5,
        monomers=default_monomers,
        n=1,
        rel_content=rel_content_zeros,
        seed=42,
    )
    assert len(ensemble) == 1
    # The main check is that it runs without division by zero or choice from empty errors
    # if there are actually monomers to choose from (n_ru > 0).
    # If len(p) == 0, then nodes_array[:, i] = rng.choice(ru_indices, size=num_graphs, p=p)
    # will fail if ru_indices is not empty.
    # This means if len(p)==0 and n_ru > 0, it implies an issue.
    # Test with rel_content causing empty `p`
    if len(default_monomers) > 0:  # Only if there are monomers to choose from
        # This will create `p = np.array([])` because `len(p)` (which is `n_ru`) `> 0`
        # but then `p` is empty. `rng.choice(ru_indices, p=np.array([]))` might error.
        # `np.random.choice` with empty `p` and non-empty `a` gives ValueError: probabilities are empty
        # The code is: if len(p) > 0: nodes_array[:, i] = rng.choice(ru_indices, size=num_graphs, p=p)
        # If rel_content_zeros leads to p being [0,0,...], sum(p)=0, then p becomes uniform.
        # So, to test the `else: p = np.array([])` branch of `if len(p) > 0`, we need `n_ru = 0`.
        # But this is handled by the `if not monomer_actual_masses:` block.
        # So, the fallback to uniform should always prevent `p` from being empty if `n_ru > 0`.
        # Let's ensure no error like "ValueError: probabilities are empty".
        graph = ensemble.graphs[0]
        assert graph.mass > 0  # Ensure some polymer was formed.


# --- Test make_linear_gradient_polymer ---
def test_mlgp_default_gradients(default_monomers):
    ensemble = make_linear_gradient_polymer(
        M=1000, monomers=default_monomers, n=2, seed=42
    )
    assert isinstance(ensemble, PolyGraphEnsemble)
    assert len(ensemble) == 2


def test_mlgp_custom_gradients(default_monomers):
    custom_fns = [lambda x: x, lambda x: 1 - x]  # Simple linear gradients
    ensemble = make_linear_gradient_polymer(
        M=1000, monomers=default_monomers, n=1, gadient_functions=custom_fns, seed=42
    )
    assert len(ensemble) == 1


def test_mlgp_gradient_function_mismatch(default_monomers):
    custom_fns = [lambda x: x]  # Only one function for two monomers
    with pytest.raises(ValueError, match="The number of gradient functions must match"):
        make_linear_gradient_polymer(
            M=1000, monomers=default_monomers, gadient_functions=custom_fns
        )


# --- Test merge_linear_polymers_to_block ---
@pytest.fixture
def ensemble1(default_monomers, start_end_groups):
    s1, e1 = Monomer("S1", 10), Monomer("E1", 11)
    return make_linear_polymer_graphs(
        M=[500, 600], monomers=[default_monomers[0]], startgroup=s1, endgroup=e1, seed=1
    )


@pytest.fixture
def ensemble2(default_monomers, start_end_groups):
    s2, e2 = (
        Monomer("S2", 12),
        Monomer("E2", 13),
    )  # Potentially e1 and s2 are "overlapping"
    return make_linear_polymer_graphs(
        M=[400, 700], monomers=[default_monomers[1]], startgroup=s2, endgroup=e2, seed=2
    )


def test_merge_basic(ensemble1, ensemble2):
    merged_ensemble = merge_linear_polymers_to_block([ensemble1, ensemble2])
    assert len(merged_ensemble) == len(ensemble1)  # Should be 2

    graph1_block = merged_ensemble.graphs[0]
    graph2_block = merged_ensemble.graphs[1]

    # Check masses (approximate, as endgroups might be removed)
    expected_mass1 = ensemble1.graphs[0].mass + ensemble2.graphs[0].mass
    expected_mass2 = ensemble1.graphs[1].mass + ensemble2.graphs[1].mass
    # Actual mass will be less if endgroups are removed.
    # The heuristic for removal is "appears <= 2 times in its segment"
    # This means if an endgroup monomer is rare in its original segment, it's likely removed.

    # Check monomer composition
    all_monomers_from_source = set(ensemble1.monomers + ensemble2.monomers)
    merged_monomer_names = {m.name for m in merged_ensemble.monomers}
    source_monomer_names = {m.name for m in all_monomers_from_source}
    assert merged_monomer_names.issubset(
        source_monomer_names
    )  # Merged might have fewer if some were never used

    # Check connectivity - at least one edge connecting the blocks
    # Max node index from first block (before merging)
    # This is harder to check precisely without deep diving into node re-indexing
    # But, number of edges should be roughly sum of edges + (num_blocks - 1)
    # or fewer if endgroups were removed.


def test_merge_remove_overlapping_false(ensemble1, ensemble2):
    merged_ensemble_no_remove = merge_linear_polymers_to_block(
        [ensemble1, ensemble2], remove_overlapping_endgroups=False
    )

    g1_merged = merged_ensemble_no_remove.graphs[0]
    g1_orig_ens1 = ensemble1.graphs[0]
    g1_orig_ens2 = ensemble2.graphs[0]

    # Expected nodes roughly sum of original nodes
    assert len(g1_merged.nodes) == len(g1_orig_ens1.nodes) + len(g1_orig_ens2.nodes)
    # Expected edges sum of original edges + 1 connecting edge
    assert len(g1_merged.edges) == len(g1_orig_ens1.edges) + len(g1_orig_ens2.edges) + 1


def test_merge_single_ensemble():
    m = Monomer("M", 100)
    ens = make_linear_polymer_graphs(M=200, monomers=[m], n=1)
    merged = merge_linear_polymers_to_block([ens])
    assert merged == ens  # Should return the same ensemble


def test_merge_empty_list():
    with pytest.raises(
        ValueError,
        match="Input list of PolyGraphEnsemble objects is empty. Cannot merge",
    ):
        merge_linear_polymers_to_block([])


def test_merge_ensembles_different_lengths(ensemble1, ensemble2):
    # ensemble1 has 2 graphs, ensemble2 has 2. Make one shorter.
    short_ensemble2 = PolyGraphEnsemble.from_lists(
        nodes=[ensemble2.graphs[0].nodes],
        edges=[ensemble2.graphs[0].edges],
        monomers=ensemble2.monomers,
    )
    with pytest.raises(
        ValueError,
        match="All input PolyGraphEnsembles must have the same number of graphs",
    ):
        merge_linear_polymers_to_block([ensemble1, short_ensemble2])
