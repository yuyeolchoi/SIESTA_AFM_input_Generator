from pathlib import Path

import numpy as np
import pytest

from siesta_afm.input_template import (
    automatic_kgrid,
    parse_hubbard_u,
    render_complete_input,
)
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


def test_coordination_species_split_defaults_off_byte_for_byte() -> None:
    structure = Structure(
        ["Co", "O", "Co"],
        [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
        np.diag([5.0, 5.0, 5.0]),
        (True, True, True),
    )
    options = dict(
        method="by-coordination",
        magnetic_species=("Co",),
        metadata={
            "coordination_numbers": {0: 4, 2: 6},
            "coordination_geometry": {0: "Td", 2: "Oh"},
        },
    )
    implicit = render_complete_input(structure, {0: 3.0, 2: -3.0}, **options)
    explicit = render_complete_input(
        structure,
        {0: 3.0, 2: -3.0},
        split_species_by_coordination=False,
        **options,
    )
    assert explicit.text.encode("utf-8") == implicit.text.encode("utf-8")
    assert explicit.species_ids == implicit.species_ids


def test_parse_hubbard_u_accepts_element_coordination_keys() -> None:
    assert parse_hubbard_u(("co@4=3.0", "Co@6=5.0", "ni=6.0")) == {
        "Co@4": 3.0,
        "Co@6": 5.0,
        "Ni": 6.0,
    }


def test_inverse_spinel_coordination_split_roundtrips_order_and_hubbard_u(
    tmp_path: Path,
) -> None:
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
        hubbard_u=("Ni@6=6.0", "Co@4=3.0", "Co@6=5.0"),
        split_species_by_coordination=True,
    )

    assert result.species_ids == (1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4)
    assert "   2   27 Co_2  # Co CN=6 (Oh)" in result.text
    assert "   3   27 Co_3  # Co CN=4 (Td)" in result.text
    assert (
        "Co_2.psf and Co_3.psf must both be provided "
        "(copies of the Co pseudopotential"
    ) in result.text
    co_2_projector = result.text.split("Co_2 1  # species label", 1)[1]
    co_3_projector = result.text.split("Co_3 1  # species label", 1)[1]
    assert " 5.00 0.00  # U_eff" in co_2_projector.split("Co_3 1", 1)[0]
    assert " 3.00 0.00  # U_eff" in co_3_projector.split("%endblock", 1)[0]

    coordinate_rows = result.text.split(
        "%block AtomicCoordinatesAndAtomicSpecies", 1
    )[1].split("%endblock AtomicCoordinatesAndAtomicSpecies", 1)[0]
    assert [
        line.split("#", 1)[1].strip()
        for line in coordinate_rows.splitlines()
        if "#" in line
    ] == [f"{symbol} {index}" for index, symbol in enumerate(structure.symbols, 1)]

    destination = tmp_path / "split.fdf"
    destination.write_text(result.text, encoding="utf-8")
    reread = read_structure(destination)
    assert reread.symbols == structure.symbols
    assert np.allclose(reread.positions, structure.positions)
    assert reread.species_ids == list(result.species_ids)


def test_coordination_hubbard_u_requires_split_flag() -> None:
    structure = Structure(["Co"], [[0, 0, 0]])
    with pytest.raises(
        ValueError,
        match="@CN Hubbard U requires --split-species-by-coordination",
    ):
        render_complete_input(
            structure,
            {0: 3.0},
            method="alternating-index",
            magnetic_species=("Co",),
            hubbard_u=("Co@4=3.0",),
        )


def test_plain_hubbard_u_applies_to_all_coordination_split_species() -> None:
    structure = Structure(["Co", "Co"], [[0, 0, 0], [1, 1, 1]])
    result = render_complete_input(
        structure,
        {0: 3.0, 1: -3.0},
        method="by-coordination",
        magnetic_species=("Co",),
        metadata={"coordination_numbers": {0: 4, 1: 6}},
        hubbard_u=("Co=4.0",),
        split_species_by_coordination=True,
    )
    assert result.text.count(" 4.00 0.00  # U_eff") == 2
    assert any(
        "plain Hubbard U Co=4 applies the same U to split species Co_1 and Co_2"
        in warning
        for warning in result.warnings
    )


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
    assert any(
        "pass --split-species-by-coordination" in warning
        for warning in result.warnings
    )
