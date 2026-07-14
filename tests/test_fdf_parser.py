from pathlib import Path

import numpy as np
import pytest

from siesta_afm.fdf_writer import patch_fdf_text
from ase.units import Bohr

from siesta_afm.io import parse_dm_init_spin, parse_fdf_structure, parse_xv_structure


def test_fdf_parser_preserves_species_and_order(tmp_path: Path) -> None:
    path = tmp_path / "input.fdf"
    path.write_text(
        """LatticeConstant 2.0 Ang
%block ChemicalSpeciesLabel
1 29 Cu
2 8 O
%endblock ChemicalSpeciesLabel
%block LatticeVectors
2 0 0
0 2 0
0 0 2
%endblock LatticeVectors
AtomicCoordinatesFormat Fractional
%block AtomicCoordinatesAndAtomicSpecies
0.0 0.0 0.0 2
0.5 0.5 0.5 1
%endblock AtomicCoordinatesAndAtomicSpecies
""",
        encoding="utf-8",
    )
    structure = parse_fdf_structure(path)
    assert structure.symbols == ["O", "Cu"]
    assert structure.species_ids == [2, 1]
    assert np.allclose(structure.cell, np.eye(3) * 4.0)
    assert np.allclose(structure.positions[1], [2.0, 2.0, 2.0])
    assert [row["siesta_index"] for row in structure.mapping] == [1, 2]


def test_fdf_include_recursion(tmp_path: Path) -> None:
    (tmp_path / "species.fdf").write_text(
        "%block ChemicalSpeciesLabel\n1 29 Cu\n%endblock ChemicalSpeciesLabel\n",
        encoding="utf-8",
    )
    (tmp_path / "coordinates.fdf").write_text(
        "%block AtomicCoordinatesAndAtomicSpecies\n0 0 0 1\n"
        "%endblock AtomicCoordinatesAndAtomicSpecies\n",
        encoding="utf-8",
    )
    main = tmp_path / "main.fdf"
    main.write_text(
        "%include species.fdf\n%include coordinates.fdf\n", encoding="utf-8"
    )
    structure = parse_fdf_structure(main)
    assert structure.symbols == ["Cu"]


def test_recursive_include_is_rejected(tmp_path: Path) -> None:
    first = tmp_path / "a.fdf"
    second = tmp_path / "b.fdf"
    first.write_text("%include b.fdf\n", encoding="utf-8")
    second.write_text("%include a.fdf\n", encoding="utf-8")
    try:
        parse_fdf_structure(first)
    except ValueError as exc:
        assert "recursive FDF include" in str(exc)
    else:
        raise AssertionError("recursive include was not rejected")


def test_patch_replaces_existing_spin_configuration() -> None:
    original = "SpinPolarized false\n%block DM.InitSpin\n1 0.1\n%endblock DM.InitSpin\n"
    generated = (
        "SpinPolarized true\n%block DM.InitSpin\n2 -0.5\n%endblock DM.InitSpin\n"
    )
    patched = patch_fdf_text(original, generated)
    assert "Spin polarized" in patched
    assert patched.lower().count("%block dm.initspin") == 1
    assert parse_dm_init_spin(patched) == [(2, -0.5)]


def test_patch_normalizes_legacy_spin_keyword_variants_and_is_idempotent() -> None:
    generated = "%block DM.InitSpin\n1 0.5\n%endblock DM.InitSpin\n"
    for keyword in ("Spin.Polarized false", "spin_polarized F", "SPIN-POLARIZED no"):
        once = patch_fdf_text(keyword + "\n", generated)
        twice = patch_fdf_text(once, generated)
        assert once == twice
        assert once.count("Spin polarized") == 1
        assert "SpinPolarized" not in once


def test_patch_updates_modern_spin_keyword_without_duplication() -> None:
    generated = "%block DM.InitSpin\n1 0.5\n%endblock DM.InitSpin\n"
    patched = patch_fdf_text("Spin non-polarized\n", generated)
    assert patched.count("Spin polarized") == 1


def test_patch_rejects_noncollinear_and_spin_orbit_inputs() -> None:
    generated = "%block DM.InitSpin\n1 0.5\n%endblock DM.InitSpin\n"
    for mode in ("non-collinear", "non-colinear", "spin-orbit"):
        try:
            patch_fdf_text(f"Spin {mode}\n", generated)
        except ValueError as exc:
            assert "collinear DM.InitSpin" in str(exc)
        else:
            raise AssertionError(f"Spin {mode} was not rejected")


@pytest.mark.parametrize(
    "keyword",
    [
        "NonCollinearSpin true",
        "NonCollinearSpin T",
        "NonCollinearSpin .true.",
        "NonCollinearSpin yes",
        "NonCollinearSpin 1",
        "SpinOrbit true",
    ],
)
def test_patch_rejects_enabled_legacy_noncollinear_controls(keyword: str) -> None:
    generated = "%block DM.InitSpin\n1 0.5\n%endblock DM.InitSpin\n"
    with pytest.raises(ValueError, match="enabled legacy"):
        patch_fdf_text(keyword + "\n", generated)


@pytest.mark.parametrize(
    "keyword", ["NonCollinearSpin F", "NonCollinearSpin .false.", "SpinOrbit no"]
)
def test_patch_removes_disabled_legacy_noncollinear_controls(keyword: str) -> None:
    generated = "%block DM.InitSpin\n1 0.5\n%endblock DM.InitSpin\n"
    original = f"SystemName test\n{keyword}\nSpinPolarized false\n"
    once = patch_fdf_text(original, generated)
    twice = patch_fdf_text(once, generated)
    assert once == twice
    assert once.count("Spin polarized") == 1
    normalized = once.lower().replace(".", "").replace("_", "").replace("-", "")
    assert "noncollinearspin" not in normalized
    assert "spinorbit" not in normalized
    assert "spinpolarized" not in normalized


def test_xv_fallback_parser_reads_bohr_and_preserves_order(tmp_path: Path) -> None:
    path = tmp_path / "sample.XV"
    path.write_text(
        "1 0 0\n0 1 0\n0 0 1\n2\n1 29 0.0 0.0 0.0 0 0 0\n2 8 1.0 0.0 0.0 0 0 0\n",
        encoding="utf-8",
    )
    structure = parse_xv_structure(path)
    assert structure.symbols == ["Cu", "O"]
    assert structure.species_ids == [1, 2]
    assert np.isclose(structure.positions[1, 0], Bohr)


def test_dm_init_spin_accepts_sign_shorthand_and_noncollinear_angles() -> None:
    warnings: list[str] = []
    rows = parse_dm_init_spin(
        "%block DM.InitSpin\n1 +\n2 -\n3 0.75 90 180\n"
        "4 -0.25 45\n%endblock DM.InitSpin\n",
        warnings=warnings,
    )
    assert rows == [(1, 1.0), (2, -1.0), (3, 0.75), (4, -0.25)]
    assert len(warnings) == 2
    assert all("theta/phi" in warning for warning in warnings)


def test_fdf_label_separators_are_ignored() -> None:
    text = "%block D_M-Init.Spin\n1 +\n%endblock D.M_Init-Spin\n"
    assert parse_dm_init_spin(text) == [(1, 1.0)]


@pytest.mark.parametrize(
    ("coordinate_format", "expected"),
    [
        ("Ang", 0.5),
        ("Bohr", 0.5 * Bohr),
        ("Fractional", 1.0),
        ("ScaledCartesian", 1.0),
    ],
)
def test_fdf_coordinate_formats(
    tmp_path: Path, coordinate_format: str, expected: float
) -> None:
    path = tmp_path / f"{coordinate_format}.fdf"
    path.write_text(
        "LatticeConstant 2.0 Ang\n"
        "%block ChemicalSpeciesLabel\n1 29 Cu\n%endblock ChemicalSpeciesLabel\n"
        "%block LatticeVectors\n1 0 0\n0 1 0\n0 0 1\n%endblock LatticeVectors\n"
        f"AtomicCoordinatesFormat {coordinate_format}\n"
        "%block AtomicCoordinatesAndAtomicSpecies\n0.5 0.5 0.5 1\n"
        "%endblock AtomicCoordinatesAndAtomicSpecies\n",
        encoding="utf-8",
    )
    structure = parse_fdf_structure(path)
    assert np.allclose(structure.positions[0], [expected] * 3)
