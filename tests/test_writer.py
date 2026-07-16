import numpy as np
import pytest

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
        "coordination_geometry": {0: "Oh", 2: "Oh", 4: "Td"},
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


def test_writer_does_not_guess_geometry_when_metadata_has_only_cn() -> None:
    text = render_dm_init_spin(
        {0: 1.0},
        method="by-coordination",
        magnetic_species=["Cu"],
        metadata={"coordination_numbers": {0: 4}},
        structure=Structure(["Cu"], [[0, 0, 0]]),
    )
    assert "# Cu  (CN=4)" in text
    assert "Td" not in text


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


def test_writer_renders_mixed_noncollinear_rows_and_preserves_comments() -> None:
    structure = Structure(
        ["Cu", "Ni"], np.zeros((2, 3)), np.eye(3), (False,) * 3
    )
    text = render_dm_init_spin(
        {0: -1.0, 1: -0.5},
        method="graph-coloring",
        magnetic_species=["Cu", "Ni"],
        angles={0: (90.0, 120.0)},
        structure=structure,
    )

    assert "Spin non-collinear" in text
    assert "     1     1.000000    90.0000   120.0000  # Cu" in text
    assert "     2    -0.500000  # Ni" in text
    assert parse_dm_init_spin(text, include_angles=True) == [
        (1, 1.0, 90.0, 120.0),
        (2, 0.5, 180.0, 0.0),
    ]


def test_writer_empty_angles_keeps_collinear_output() -> None:
    plain = render_dm_init_spin(
        {0: 0.5}, method="layer", magnetic_species=["Cu"]
    )
    empty = render_dm_init_spin(
        {0: 0.5}, method="layer", magnetic_species=["Cu"], angles={}
    )

    assert empty == plain
    assert "Spin polarized" in empty


@pytest.mark.parametrize(
    "base",
    ["SystemName test\n", "NonCollinearSpin true\n", "Spin non-collinear\n"],
)
def test_patch_accepts_noncollinear_spin_blocks(base: str) -> None:
    spin_text = render_dm_init_spin(
        {0: 1.0},
        method="graph-coloring",
        magnetic_species=["Cu"],
        angles={0: (90.0, 0.0)},
    )

    patched = patch_fdf_text(base, spin_text)

    assert patched.count("Spin non-collinear") == 1
    assert "NonCollinearSpin" not in patched
    assert patch_fdf_text(patched, spin_text) == patched


def test_patch_keeps_soc_blocked_for_noncollinear_spin_blocks() -> None:
    spin_text = render_dm_init_spin(
        {0: 1.0},
        method="graph-coloring",
        magnetic_species=["Cu"],
        angles={0: (90.0, 0.0)},
    )

    with pytest.raises(ValueError, match="spinorbit"):
        patch_fdf_text("SpinOrbit true\n", spin_text)


def test_patch_still_rejects_noncollinear_base_for_collinear_spin_block() -> None:
    spin_text = render_dm_init_spin(
        {0: 1.0}, method="layer", magnetic_species=["Cu"]
    )

    with pytest.raises(ValueError, match="collinear DM.InitSpin"):
        patch_fdf_text("NonCollinearSpin true\n", spin_text)
