import numpy as np

from siesta_afm.fdf_writer import render_dm_init_spin
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
