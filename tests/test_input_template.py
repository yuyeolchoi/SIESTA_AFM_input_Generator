from pathlib import Path

import numpy as np

from siesta_afm.input_template import automatic_kgrid, render_complete_input
from siesta_afm.io import parse_dm_init_spin, read_structure
from siesta_afm.structure import Structure
from siesta_afm.workflows import generate_assignment


ROOT = Path(__file__).parents[1]


def test_complete_input_roundtrips_structure_order_coordinates_and_spins(
    tmp_path: Path,
) -> None:
    structure = Structure(
        ["O", "Ni", "Co", "O", "Ni"],
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]],
        np.diag([8.0, 9.0, 10.0]),
        (True, True, True),
    )
    spins = {1: 2.0, 2: -3.0, 4: -2.0}
    result = render_complete_input(
        structure,
        spins,
        method="manual-groups",
        magnetic_species=("Ni", "Co"),
    )
    path = tmp_path / "input.fdf"
    path.write_text(result.text, encoding="utf-8")

    reread = read_structure(path)
    assert reread.symbols == structure.symbols
    assert np.allclose(reread.positions, structure.positions)
    assert parse_dm_init_spin(path) == [(2, 2.0), (3, -3.0), (5, -2.0)]
    coordinate_rows = result.text.split(
        "%block AtomicCoordinatesAndAtomicSpecies", 1
    )[1].split("%endblock AtomicCoordinatesAndAtomicSpecies", 1)[0]
    assert [line.split("#", 1)[1].strip() for line in coordinate_rows.splitlines() if "#" in line] == [
        "O 1",
        "Ni 2",
        "Co 3",
        "O 4",
        "Ni 5",
    ]


def test_complete_input_preserves_fdf_species_ids(tmp_path: Path) -> None:
    structure = Structure(
        ["Ni", "O", "Ni"],
        [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
        np.diag([5.0, 5.0, 5.0]),
        (True, True, True),
        [7, 3, 7],
    )
    result = render_complete_input(
        structure,
        {0: 2.0, 2: -2.0},
        method="alternating-index",
        magnetic_species=("Ni",),
    )
    path = tmp_path / "preserved.fdf"
    path.write_text(result.text, encoding="utf-8")
    assert read_structure(path).species_ids == [7, 3, 7]


def test_same_element_species_labels_receive_the_same_hubbard_u() -> None:
    structure = Structure(
        ["Co", "O", "Co"],
        [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
        np.diag([5.0, 5.0, 5.0]),
        (True, True, True),
        [4, 2, 9],
    )
    result = render_complete_input(
        structure,
        {0: 3.0, 2: -3.0},
        method="alternating-index",
        magnetic_species=("Co",),
    )
    assert "   4   27 Co_4" in result.text
    assert "   9   27 Co_9" in result.text
    assert "Co_4 1  # species label" in result.text
    assert "Co_9 1  # species label" in result.text
    assert result.text.count(" 3.32 0.00") == 2


def test_automatic_kgrid_scales_and_respects_nonperiodic_axes() -> None:
    small = Structure(["Ni"], [[0, 0, 0]], np.diag([3.0, 5.0, 7.0]), (True, True, True))
    large = Structure(["Ni"], [[0, 0, 0]], np.diag([15.0, 20.0, 40.0]), (True, True, True))
    slab = Structure(["Ni"], [[0, 0, 0]], np.diag([3.0, 5.0, 25.0]), (True, True, False))
    assert automatic_kgrid(small) == (10, 6, 5)
    assert automatic_kgrid(large) == (2, 2, 1)
    assert automatic_kgrid(slab) == (10, 6, 1)


def test_dftu_defaults_overrides_omission_and_unsupported_comment() -> None:
    structure = Structure(
        ["Ni", "Co", "O", "Cu"],
        np.zeros((4, 3)),
    )
    default = render_complete_input(
        structure,
        {0: 2.0, 1: -3.0, 3: 1.0},
        method="manual-groups",
        magnetic_species=("Ni", "Co", "Cu"),
        hubbard_u=("Ni=6.0", "Co=3.3"),
    )
    assert "%block LDAU.proj" in default.text
    assert " 6.00 0.00" in default.text
    assert " 3.30 0.00" in default.text
    assert "# no default U for Cu" in default.text
    assert "O 1\n n=" not in default.text

    modern = render_complete_input(
        structure,
        {0: 2.0},
        method="alternating-index",
        magnetic_species=("Ni",),
        dftu_keyword="dftu",
    )
    assert "%block DFTU.Proj" in modern.text
    assert "DFTU.ProjectorGenerationMethod 2" in modern.text

    omitted = render_complete_input(
        structure,
        {0: 2.0},
        method="alternating-index",
        magnetic_species=("Ni",),
        lda_u=False,
    )
    assert "%block LDAU.proj" not in omitted.text
    assert "DFT+U omitted" in omitted.text


def test_inverse_spinel_warns_that_hubbard_u_is_per_species() -> None:
    structure = read_structure(
        ROOT / "tests" / "fixtures" / "inverse_spinel_coordination.cif",
        slab=True,
    )
    _, assignment, spins = generate_assignment(
        structure,
        ("Ni", "Co"),
        "by-coordination",
        ("Ni@6=2.0", "Co@4=2.0", "Co@6=0.0"),
        anion_species=("O",),
    )
    result = render_complete_input(
        structure,
        spins,
        method=assignment.method,
        magnetic_species=("Ni", "Co"),
        metadata=assignment.metadata,
    )
    assert any(
        "element Co occupies two coordination sublattices but LDA+U is per species"
        in warning
        for warning in result.warnings
    )
