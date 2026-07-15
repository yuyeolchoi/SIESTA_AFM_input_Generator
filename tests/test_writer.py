import numpy as np

from siesta_afm.fdf_writer import patch_fdf_text, render_dm_init_spin
from siesta_afm.io import parse_dm_init_spin
from siesta_afm.structure import Structure


def test_writer_uses_one_based_indices_and_exact_moments() -> None:
    text = render_dm_init_spin(
        {1: 0.5, 4: -1.25},
        method="alternating-index",
        magnetic_species=["Cu", "Ni"],
    )
    assert "Spin polarized" in text
    assert "SpinPolarized" not in text
    assert parse_dm_init_spin(text) == [(2, 0.5), (5, -1.25)]


def test_writer_preserves_atom_order_and_can_write_zero_spins() -> None:
    atoms = Structure(["O", "Cu", "H", "Cu"], np.zeros((4, 3)), np.eye(3), (False,) * 3)
    text = render_dm_init_spin(
        {1: 0.5, 3: -0.5},
        method="layer",
        magnetic_species=["Cu"],
        structure=atoms,
        write_zero_spins=True,
    )
    assert parse_dm_init_spin(text) == [(1, 0.0), (2, 0.5), (3, 0.0), (4, -0.5)]
    assert "     1     0.000000  # O " in text
    assert "     2     0.500000  # Cu" in text
    assert "     3     0.000000  # H " in text
    assert "     4    -0.500000  # Cu" in text


def test_writer_site_comments_roundtrip_and_show_coordination() -> None:
    atoms = Structure(
        ["Ni", "O", "Co", "O", "Co"],
        np.zeros((5, 3)),
        np.eye(3),
        (False,) * 3,
    )
    metadata = {
        "coordination_numbers": {0: 6, 2: 6, 4: 4},
        "sublattice_classification": {0: "up", 2: "up", 4: "down"},
    }
    text = render_dm_init_spin(
        {0: 2.0, 2: 0.0, 4: -2.0},
        method="by-coordination",
        magnetic_species=["Ni", "Co"],
        metadata=metadata,
        structure=atoms,
    )
    assert "     1     2.000000  # Ni  (Oh, CN=6)" in text
    assert "     3     0.000000  # Co  (Oh, CN=6)" in text
    assert "     5    -2.000000  # Co  (Td, CN=4)" in text
    warnings: list[str] = []
    assert parse_dm_init_spin(text, warnings=warnings) == [
        (1, 2.0),
        (3, 0.0),
        (5, -2.0),
    ]
    assert warnings == []
    assert "# Co  (Td, CN=4)" in patch_fdf_text("SystemName test\n", text)


def test_writer_preserves_legacy_rows_without_structure_or_when_opted_out() -> None:
    without_structure = render_dm_init_spin(
        {0: 0.5}, method="layer", magnetic_species=["Cu"]
    )
    with_opt_out = render_dm_init_spin(
        {0: 0.5},
        method="layer",
        magnetic_species=["Cu"],
        structure=Structure(["Cu"], [[0, 0, 0]]),
        site_comments=False,
    )
    assert "     1     0.500000\n" in without_structure
    assert "     1     0.500000\n" in with_opt_out
    assert "# Cu" not in without_structure
    assert "# Cu" not in with_opt_out
