import numpy as np
import pytest

from siesta_afm.neighbors import build_neighbor_graph
from siesta_afm.ordering import (
    NonBipartiteError,
    alternating_index,
    checkerboard_ordering,
    detect_layers,
    layer_ordering,
    neighbor_bipartite_ordering,
    manual_groups_ordering,
    propagation_vector_ordering,
)
from siesta_afm.structure import Structure


def make_structure(points, pbc=(False, False, False)) -> Structure:
    return Structure(["Cu"] * len(points), points, np.eye(3) * 10, pbc)


def test_alternating_index_uses_only_magnetic_list() -> None:
    result = alternating_index([1, 4, 7, 10])
    assert result.signs == {1: 1, 4: -1, 7: 1, 10: -1}


def test_layer_tolerance_clusters_close_coordinates() -> None:
    atoms = make_structure([[0, 0, 0], [1, 0, 0.1], [0, 0, 1.0], [1, 0, 1.15]])
    layers = detect_layers(atoms, range(4), "z", 0.2)
    assert layers == [[0, 1], [2, 3]]
    result = layer_ordering(atoms, range(4), axis="z", tolerance=0.2)
    assert result.signs == {0: 1, 1: 1, 2: -1, 3: -1}


def test_neighbor_bipartite_colors_disconnected_components() -> None:
    atoms = make_structure([[0, 0, 0], [1, 0, 0], [5, 0, 0], [6, 0, 0]])
    result = neighbor_bipartite_ordering(atoms, range(4), cutoff=1.01)
    graph, _, _ = build_neighbor_graph(atoms, range(4), 1.01)
    assert all(result.signs[a] != result.signs[b] for a, b in graph.edges)


def test_non_bipartite_requires_explicit_frustrated_flag() -> None:
    root3 = np.sqrt(3.0)
    atoms = make_structure([[0, 0, 0], [1, 0, 0], [0.5, root3 / 2, 0]])
    with pytest.raises(NonBipartiteError):
        neighbor_bipartite_ordering(atoms, range(3), cutoff=1.01)
    result = neighbor_bipartite_ordering(
        atoms, range(3), cutoff=1.01, allow_frustrated=True
    )
    assert result.warnings
    graph, _, _ = build_neighbor_graph(atoms, range(3), 1.01)
    cut = sum(result.signs[a] != result.signs[b] for a, b in graph.edges)
    assert cut == 2
    repeated = neighbor_bipartite_ordering(
        atoms, range(3), cutoff=1.01, allow_frustrated=True, seed=0
    )
    assert repeated.signs == result.signs


def test_propagation_vector_uses_fractional_phase() -> None:
    atoms = make_structure([[0, 0, 0], [5, 0, 0]])
    result = propagation_vector_ordering(atoms, [0, 1], [1, 0, 0])
    assert result.signs == {0: 1, 1: -1}


def test_checkerboard_uses_in_plane_neighbors_per_layer() -> None:
    atoms = make_structure(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 2],
            [1, 0, 2],
            [0, 1, 2],
            [1, 1, 2],
        ]
    )
    result = checkerboard_ordering(atoms, range(8), plane="xy", cutoff="auto")
    assert result.signs[0] != result.signs[1]
    assert result.signs[0] != result.signs[2]
    assert result.signs[4] != result.signs[5]


def test_checkerboard_normal_tolerance_is_configurable() -> None:
    atoms = make_structure([[0, 0, 0], [1, 0, 0.4]])
    with pytest.raises(ValueError, match="no in-plane magnetic pairs"):
        checkerboard_ordering(
            atoms, [0, 1], plane="xy", cutoff=1.1, normal_tolerance=0.25
        )
    result = checkerboard_ordering(
        atoms, [0, 1], plane="xy", cutoff=1.1, normal_tolerance=0.5
    )
    assert result.signs[0] != result.signs[1]


def test_fractional_propagation_vector_explains_singular_cell() -> None:
    atoms = Structure(["Cu"], [[0, 0, 0]], np.zeros((3, 3)), (False,) * 3)
    with pytest.raises(ValueError, match="--cartesian-coordinates"):
        propagation_vector_ordering(atoms, [0], [0.5, 0, 0])


def test_manual_groups_reject_overlap_and_missing_atoms() -> None:
    with pytest.raises(ValueError, match="overlap"):
        manual_groups_ordering([0, 1], [1, 2], [2])
    with pytest.raises(ValueError, match="omit"):
        manual_groups_ordering([0, 1], [1], [])


def test_manual_groups_assign_all_magnetic_atoms() -> None:
    result = manual_groups_ordering([0, 2, 4], [1, 5], [3])
    assert result.signs == {0: 1, 4: 1, 2: -1}
