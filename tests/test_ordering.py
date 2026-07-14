import numpy as np
import pytest

from siesta_afm.neighbors import build_neighbor_graph
from siesta_afm.ordering import (
    NonBipartiteError,
    alternating_index,
    by_species_ordering,
    checkerboard_ordering,
    coordination_ordering,
    detect_layers,
    direction_layer_ordering,
    layer_ordering,
    neighbor_bipartite_ordering,
    manual_groups_ordering,
    propagation_vector_ordering,
    random_ordering,
)
from siesta_afm.structure import Structure
from siesta_afm.workflows import generate_assignment


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


def test_propagation_vector_warns_with_one_based_node_indices() -> None:
    atoms = make_structure([[0, 0, 0], [2.5, 0, 0], [5, 0, 0]])
    result = propagation_vector_ordering(atoms, range(3), [1, 0, 0])
    warning = "\n".join(result.warnings)
    assert "near a node" in warning
    assert "indices: 2" in warning
    assert "input cell" in warning and "supercell" in warning


def test_periodic_odd_layers_warn_but_slab_normal_does_not() -> None:
    points = [[0, 0, 1], [0, 0, 4], [0, 0, 7]]
    periodic = make_structure(points, pbc=(False, False, True))
    slab = make_structure(points, pbc=(True, True, False))
    assert "odd layer count" in "\n".join(
        layer_ordering(periodic, range(3), axis="z").warnings
    )
    assert not layer_ordering(slab, range(3), axis="z").warnings


def test_periodic_boundary_layers_merge_and_share_sign() -> None:
    atoms = make_structure(
        [[0, 0, 0.1], [0, 0, 5.0], [0, 0, 9.9]],
        pbc=(False, False, True),
    )
    result = layer_ordering(atoms, range(3), axis="z", tolerance=0.25)
    assert result.metadata["wrapped_layer_merged"] is True
    assert result.metadata["layers"] == [[2, 0], [1]]
    assert result.signs[0] == result.signs[2] == -result.signs[1]


def test_small_periodic_cell_warns_for_self_image_inside_cutoff() -> None:
    atoms = Structure(
        ["Cu"], [[0, 0, 0]], np.eye(3), (True, True, True)
    )
    result = neighbor_bipartite_ordering(atoms, [0], cutoff=1.1)
    assert "cell too small" in "\n".join(result.warnings)
    assert result.metadata["periodic_self_image_distance"] == pytest.approx(1.0)


def test_by_species_requires_exact_partition_and_preserves_element_moments() -> None:
    atoms = Structure(
        ["Ni", "O", "Co"],
        [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
        np.eye(3) * 10,
        (False, False, False),
    )
    assignment = by_species_ordering(atoms, [0, 2], ["Ni", "Co"], ["ni"], ["CO"])
    assert assignment.signs == {0: 1, 2: -1}
    _, _, spins = generate_assignment(
        atoms,
        ["Ni", "Co"],
        "by-species",
        ["Ni=2", "Co=3"],
        up_species=["Ni"],
        down_species=["Co"],
    )
    assert spins == {0: 2.0, 2: -3.0}
    with pytest.raises(ValueError, match="exactly cover|missing"):
        by_species_ordering(atoms, [0, 2], ["Ni", "Co"], ["Ni"], ["Fe"])


def _synthetic_spinel(*, inverse: bool = False) -> Structure:
    a = 1.0 / np.sqrt(3.0)
    tetrahedral = [
        [a, a, a],
        [a, -a, -a],
        [-a, a, -a],
        [-a, -a, a],
    ]
    octahedral = [
        [11, 0, 0],
        [9, 0, 0],
        [10, 1, 0],
        [10, -1, 0],
        [10, 0, 1],
        [10, 0, -1],
    ]
    metal_symbols = ["Fe", "Fe"] if inverse else ["Co", "Fe"]
    return Structure(
        [*metal_symbols, *(["O"] * 10)],
        [[0, 0, 0], [10, 0, 0], *tetrahedral, *octahedral],
        np.eye(3) * 30,
        (False, False, False),
    )


def test_coordination_classifies_tetrahedral_and_octahedral_sites() -> None:
    atoms = _synthetic_spinel()
    result = coordination_ordering(atoms, [0, 1])
    assert result.metadata["coordination_numbers"] == {0: 4, 1: 6}
    assert result.signs == {0: -1, 1: 1}


def test_inverse_spinel_coordination_moments_and_warning(tmp_path) -> None:
    atoms = _synthetic_spinel(inverse=True)
    site_moments = tmp_path / "site_moments.csv"
    site_moments.write_text(
        "atom_index,element,moment\n1,Fe,5.0\n", encoding="utf-8"
    )
    _, assignment, spins = generate_assignment(
        atoms,
        ["Fe"],
        "by-coordination",
        ["Fe@4=3", "Fe@6=4", "Fe=1", "0.5"],
        site_moment_file=str(site_moments),
    )
    assert spins == {0: -5.0, 1: 4.0}
    assert "inverse spinel" in "\n".join(assignment.warnings)


def test_coordination_error_names_atom_element_and_cn() -> None:
    atoms = _synthetic_spinel()
    with pytest.raises(ValueError, match=r"atom 1 \(Co, CN=4\)"):
        coordination_ordering(
            atoms, [0, 1], up_coordination=[8], down_coordination=[6]
        )


def test_afm_g_preset_equals_explicit_q_vector() -> None:
    atoms = Structure(
        ["Ni"] * 4,
        [[0, 0, 0], [5, 0, 0], [0, 5, 0], [0, 0, 5]],
        np.eye(3) * 10,
        (True, True, True),
    )
    _, preset, _ = generate_assignment(
        atoms, ["Ni"], "propagation-vector", "1", afm_type="G"
    )
    _, explicit, _ = generate_assignment(
        atoms, ["Ni"], "propagation-vector", "1", q_vector=(0.5, 0.5, 0.5)
    )
    assert preset.signs == explicit.signs
    assert preset.metadata["afm_type"] == "G"
    with pytest.raises(ValueError, match="mutually exclusive"):
        generate_assignment(
            atoms,
            ["Ni"],
            "propagation-vector",
            "1",
            afm_type="G",
            q_vector=(0.5, 0.5, 0.5),
        )
    with pytest.raises(ValueError, match="fractional coordinates"):
        generate_assignment(
            atoms,
            ["Ni"],
            "propagation-vector",
            "1",
            afm_type="G",
            fractional_coordinates=False,
        )


def test_111_layer_direction_gives_fcc_afm_ii_planes() -> None:
    atoms = Structure(
        ["Ni"] * 4,
        [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]],
        np.eye(3),
        (True, True, True),
    )
    result = direction_layer_ordering(atoms, range(4), [1, 1, 1], tolerance=0.1)
    assert result.metadata["layers"] == [[0], [1, 2, 3]]
    assert result.signs == {0: 1, 1: -1, 2: -1, 3: -1}


def test_random_ordering_is_seeded_and_warns_about_physical_meaning() -> None:
    first = random_ordering(range(32), seed=7)
    repeated = random_ordering(range(32), seed=7)
    different = random_ordering(range(32), seed=8)
    assert first.signs == repeated.signs
    assert first.signs != different.signs
    assert "not a physical" in "\n".join(first.warnings)
