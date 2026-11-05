import csv
import json
from pathlib import Path

import numpy as np
import pytest

from siesta_afm.cli import main
from siesta_afm.defects import (
    SLAB_SYMMETRY_NOTICE,
    apply_vacancies,
    compare_magnetic_environments,
    enumerate_vacancy_sites,
    format_environment_report,
    format_vacancy_table,
    index_map,
    remap_spin_rows,
    write_index_map_csv,
)
from siesta_afm.io import parse_dm_init_spin, read_structure, write_structure
from siesta_afm.structure import Structure
from siesta_afm.symmetry import symmetry_orbits


ROOT = Path(__file__).parents[1]
CO3O4 = ROOT / "examples" / "Co3O4_spinel_COD1538531.cif"
CUO_SLAB = ROOT / "examples" / "CuO_111_slab.cif"


def _rocksalt_like() -> Structure:
    """Tiny cubic cell: two cations and two anions, all periodic."""

    cell = np.diag([4.0, 4.0, 4.0])
    fractional = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
        ]
    )
    return Structure(
        ["Ni", "Ni", "O", "O"], fractional @ cell, cell=cell, pbc=(True, True, True)
    )


# --- symmetry orbits -------------------------------------------------------


def test_symmetry_orbits_groups_by_permutations_and_respects_index_subset() -> None:
    permutations = [(0, 1, 2, 3), (1, 0, 2, 3), (0, 1, 3, 2)]

    assert symmetry_orbits(permutations) == [[0, 1], [2, 3]]
    assert symmetry_orbits(permutations, indices=[2, 3]) == [[2, 3]]
    assert symmetry_orbits([(0, 1, 2, 3)], indices=[1, 3]) == [[1], [3]]


def test_symmetry_orbits_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one permutation"):
        symmetry_orbits([])


# --- Structure.parent_indices --------------------------------------------


def test_structure_validates_parent_indices() -> None:
    with pytest.raises(ValueError, match="parent_indices and symbols"):
        Structure(["O", "O"], np.zeros((2, 3)), parent_indices=[0])
    with pytest.raises(ValueError, match="duplicates"):
        Structure(["O", "O"], np.zeros((2, 3)), parent_indices=[1, 1])
    with pytest.raises(ValueError, match="nonnegative"):
        Structure(["O", "O"], np.zeros((2, 3)), parent_indices=[-1, 0])


def test_mapping_reports_parent_index_and_with_pbc_preserves_it() -> None:
    structure = Structure(["O", "O"], np.zeros((2, 3)), parent_indices=[0, 5])

    assert [row["parent_index"] for row in structure.mapping] == [0, 5]
    assert structure.with_pbc((True, True, False)).parent_indices == [0, 5]
    assert [row["parent_index"] for row in _rocksalt_like().mapping] == [0, 1, 2, 3]


# --- apply_vacancies / index map / remap -----------------------------------


def test_apply_vacancies_preserves_order_species_ids_and_parent_mapping() -> None:
    parent = _rocksalt_like()
    parent.species_ids = [1, 1, 2, 2]

    child = apply_vacancies(parent, [2])

    assert child.symbols == ["Ni", "Ni", "O"]
    assert child.species_ids == [1, 1, 2]
    assert child.parent_indices == [0, 1, 3]
    assert child.pbc == parent.pbc
    np.testing.assert_allclose(child.positions, parent.positions[[0, 1, 3]])
    assert index_map(child) == [(1, 1, "Ni"), (2, 2, "Ni"), (4, 3, "O")]


def test_apply_vacancies_chains_parent_mapping_to_the_original() -> None:
    first = apply_vacancies(_rocksalt_like(), [0])
    second = apply_vacancies(first, [1])  # removes original atom 2 (index 2)

    assert first.parent_indices == [1, 2, 3]
    assert second.parent_indices == [1, 3]


@pytest.mark.parametrize(
    "remove, message",
    [
        ([], "at least one atom"),
        ([4], "out of range: 5"),
        ([-1], "out of range: 0"),
        ([0, 1, 2, 3], "every atom"),
    ],
)
def test_apply_vacancies_rejects_bad_requests(remove: list[int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        apply_vacancies(_rocksalt_like(), remove)


def test_remap_spin_rows_shifts_indices_and_drops_removed_atoms() -> None:
    child = apply_vacancies(_rocksalt_like(), [0])

    remapped, dropped = remap_spin_rows(
        [(1, 1.0), (2, -1.0), (4, 0.5, 90.0, 0.0)], child, parent_length=4
    )

    assert remapped == [(1, -1.0), (3, 0.5, 90.0, 0.0)]
    assert dropped == [1]
    with pytest.raises(ValueError, match="out of parent range: 5"):
        remap_spin_rows([(5, 1.0)], child, parent_length=4)
    with pytest.raises(ValueError, match="no parent mapping"):
        remap_spin_rows([(1, 1.0)], _rocksalt_like())


def test_write_index_map_csv(tmp_path: Path) -> None:
    child = apply_vacancies(_rocksalt_like(), [2])
    path = write_index_map_csv(tmp_path / "map.csv", child)

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [(r["parent_index"], r["child_index"], r["element"]) for r in rows] == [
        ("1", "1", "Ni"),
        ("2", "2", "Ni"),
        ("4", "3", "O"),
    ]


# --- enumerate_vacancy_sites -------------------------------------------------


def test_enumerate_oxygen_vacancies_in_co3o4_finds_one_orbit() -> None:
    pytest.importorskip("spglib")
    structure = read_structure(CO3O4)

    enumeration = enumerate_vacancy_sites(
        structure, "O", magnetic_species=["Co"], symmetry_dedup=True
    )

    assert enumeration.n_candidates == 32
    assert len(enumeration.sites) == 1
    site = enumeration.sites[0]
    assert site.multiplicity == 32
    assert site.coordination_number == 4
    assert site.coordination_geometry == "Td"
    assert site.neighbor_elements == (("Co", 4),)
    assert len(site.affected_magnetic_atoms) == 4
    # One tetrahedral (index >= 48) and three octahedral Co surround each O.
    assert sum(index >= 48 for index in site.affected_magnetic_atoms) == 1
    assert site.surface is False
    assert any("not defect formation energies" in notice for notice in enumeration.notices)
    assert "17" in format_vacancy_table(enumeration)


def test_enumerate_cobalt_vacancies_in_co3o4_separates_td_and_oh_sites() -> None:
    pytest.importorskip("spglib")
    structure = read_structure(CO3O4)

    enumeration = enumerate_vacancy_sites(structure, "Co", symmetry_dedup=True)

    by_cn = {site.coordination_number: site for site in enumeration.sites}
    assert set(by_cn) == {4, 6}
    assert by_cn[4].multiplicity == 8 and by_cn[4].coordination_geometry == "Td"
    assert by_cn[6].multiplicity == 16 and by_cn[6].coordination_geometry == "Oh"
    assert [site.orbit_id for site in enumeration.sites] == [1, 2]
    assert enumeration.sites[0].coordination_number == 4  # low CN listed first
    assert all(site.affected_magnetic_atoms == () for site in enumeration.sites)


def test_enumerate_without_symmetry_lists_every_atom() -> None:
    structure = read_structure(CO3O4)

    enumeration = enumerate_vacancy_sites(structure, "O", symmetry_dedup=False)

    assert len(enumeration.sites) == 32
    assert all(site.multiplicity == 1 for site in enumeration.sites)
    assert enumeration.magnetic_species == ()


def test_enumerate_slab_marks_surface_layer_and_notes_3d_symmetry() -> None:
    pytest.importorskip("spglib")
    structure = read_structure(CUO_SLAB, slab=True)

    enumeration = enumerate_vacancy_sites(
        structure, "O", magnetic_species=["Cu"], symmetry_dedup=True
    )

    assert SLAB_SYMMETRY_NOTICE in enumeration.notices
    assert enumeration.sites[0].surface is True
    assert enumeration.sites[0].layer == enumeration.sites[0].layer_count - 1
    assert all(site.coordination_number == 2 for site in enumeration.sites)
    assert all(site.affected_magnetic_atoms for site in enumeration.sites)


def test_enumerate_rejects_unknown_species_and_cell_free_symmetry() -> None:
    with pytest.raises(ValueError, match="no atoms of species Zn"):
        enumerate_vacancy_sites(_rocksalt_like(), "Zn")
    molecule = Structure(["Fe", "C", "O"], np.eye(3))
    with pytest.raises(ValueError, match="nonsingular cell"):
        enumerate_vacancy_sites(molecule, "O", symmetry_dedup=True)


# --- compare_magnetic_environments --------------------------------------------


def test_compare_magnetic_environments_detects_cn_drop_around_oxygen_vacancy() -> None:
    parent = read_structure(CO3O4)
    child = apply_vacancies(parent, [16])  # one-based atom 17

    report = compare_magnetic_environments(parent, child, ["Co"])

    assert report.anion_species == ("O",)
    changed = report.changed
    assert len(changed) == 4
    assert {c.anion_coordination_before - c.anion_coordination_after for c in changed} == {1}
    # Octahedral Co (1-16) precede the removed O 17; tetrahedral Co 49-56 shift down.
    assert [c.parent_index + 1 for c in changed] == [4, 12, 14, 50]
    assert [c.child_index + 1 for c in changed] == [4, 12, 14, 49]
    assert all(not c.removed for c in report.changes)
    assert any("--moment Co@3=... Co@5=..." in s for s in report.suggestions)
    text = format_environment_report(report)
    assert "6->5" in text and "4->3" in text
    assert report.to_dict()["changed_parent_indices"] == [
        c.parent_index + 1 for c in changed
    ]


def test_compare_magnetic_environments_reports_removed_magnetic_atom() -> None:
    parent = read_structure(CO3O4)
    child = apply_vacancies(parent, [0])  # octahedral Co 1

    report = compare_magnetic_environments(parent, child, ["Co"])

    removed = [c for c in report.changes if c.removed]
    assert [c.parent_index for c in removed] == [0]
    assert removed[0].child_index is None and removed[0].changed
    # Its six octahedral Co neighbors each lose one magnetic neighbor.
    lost = [c for c in report.changes if not c.removed and c.magnetic_neighbors_before == 6 and c.magnetic_neighbors_after == 5]
    assert len(lost) == 6
    assert any("hole in the magnetic graph" in s for s in report.suggestions)


def test_compare_magnetic_environments_requires_parent_mapping() -> None:
    parent = _rocksalt_like()
    with pytest.raises(ValueError, match="no parent mapping"):
        compare_magnetic_environments(parent, parent, ["Ni"])


# --- write_structure -----------------------------------------------------------


@pytest.mark.parametrize("suffix", [".fdf", ".cif", ".xyz"])
def test_write_structure_roundtrips_defect_model(tmp_path: Path, suffix: str) -> None:
    parent = read_structure(CO3O4)
    child = apply_vacancies(parent, [16])

    path = write_structure(child, tmp_path / f"defect{suffix}")
    reread = read_structure(path)

    assert reread.symbols == child.symbols
    np.testing.assert_allclose(reread.positions, child.positions, atol=1e-4)
    np.testing.assert_allclose(reread.cell, child.cell, atol=1e-4)
    if suffix == ".fdf":
        text = path.read_text(encoding="utf-8")
        assert "NumberOfAtoms 55" in text
        assert "# O 17 (parent 18)" in text
        assert "# Co 1\n" in text  # unchanged indices carry no parent note
    if suffix == ".xyz":
        assert "parent_index" in path.read_text(encoding="utf-8")


def test_write_structure_rejects_unsupported_formats(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="XV"):
        write_structure(_rocksalt_like(), tmp_path / "out.XV")
    with pytest.raises(ValueError, match="unsupported structure output format"):
        write_structure(_rocksalt_like(), tmp_path / "out.bin")


# --- CLI ------------------------------------------------------------------------


def test_defects_cli_prints_table_and_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("spglib")
    report = tmp_path / "sites.json"

    assert (
        main(
            [
                "defects",
                str(CO3O4),
                "--vacancy",
                "O",
                "--magnetic-species",
                "Co",
                "--symmetry-dedup",
                "--json",
                str(report),
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    assert "1 distinct orbits" in captured.out
    assert "not defect formation energies" in captured.err
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["n_distinct_sites"] == 1
    assert data["sites"][0]["representative"] == 17
    assert data["sites"][0]["affected_magnetic_atoms"] == [4, 12, 14, 50]


def test_make_defect_cli_writes_structure_map_spins_and_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pristine = tmp_path / "pristine_spin.fdf"
    assert (
        main(
            [
                "generate",
                str(CO3O4),
                "--magnetic-species",
                "Co",
                "--method",
                "coordination-bipartite",
                "--anion-species",
                "O",
                "--active-coordination",
                "4",
                "--inactive-coordination",
                "6",
                "--moment",
                "Co@4=3.0",
                "Co@6=0",
                "--output",
                str(pristine),
            ]
        )
        == 0
    )
    pristine_rows = dict(parse_dm_init_spin(pristine))
    output = tmp_path / "defect_O17.fdf"
    report = tmp_path / "defect.json"
    capsys.readouterr()

    assert (
        main(
            [
                "make-defect",
                str(CO3O4),
                "--remove-atoms",
                "17",
                "--output",
                str(output),
                "--spin-file",
                str(pristine),
                "--magnetic-species",
                "Co",
                "--json",
                str(report),
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    assert "Removed 1 atom(s): 17 (O)" in captured.err
    assert "6->5" in captured.err
    index_map_path = tmp_path / "defect_O17_index_map.csv"
    spin_path = tmp_path / "defect_O17_spin.fdf"
    assert index_map_path.is_file() and spin_path.is_file()
    child = read_structure(output)
    assert len(child) == 55
    # Octahedral Co 1-16 keep their indices; tetrahedral Co 49-56 shift down by one.
    expected = {
        (index - 1 if index > 17 else index): value for index, value in pristine_rows.items()
    }
    assert dict(parse_dm_init_spin(spin_path)) == expected
    assert "# Parent spin file:" in spin_path.read_text(encoding="utf-8")
    assert main(["validate", str(spin_path), "--structure", str(output), "--magnetic-species", "Co"]) == 0
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["n_atoms_child"] == 55 and data["removed_atoms"] == [17]
    assert data["dropped_spin_rows"] == []
    assert data["magnetic_environment"]["changed_parent_indices"] == [4, 12, 14, 50]


def test_make_defect_cli_renumbers_spins_after_a_cation_vacancy(tmp_path: Path) -> None:
    spin_file = tmp_path / "spins.fdf"
    spin_file.write_text(
        "%block DM.InitSpin\n1 1.0\n2 -1.0\n3 1.0\n%endblock DM.InitSpin\n",
        encoding="utf-8",
    )
    output = tmp_path / "defect.cif"

    assert (
        main(
            [
                "make-defect",
                str(CO3O4),
                "--remove-atoms",
                "2",
                "--output",
                str(output),
                "--spin-file",
                str(spin_file),
            ]
        )
        == 0
    )

    assert parse_dm_init_spin(tmp_path / "defect_spin.fdf") == [(1, 1.0), (2, 1.0)]


def test_make_defect_cli_rejects_out_of_range_and_orphan_spin_output(tmp_path: Path) -> None:
    assert (
        main(
            [
                "make-defect",
                str(CO3O4),
                "--remove-atoms",
                "999",
                "--output",
                str(tmp_path / "x.cif"),
            ]
        )
        == 2
    )
    assert (
        main(
            [
                "make-defect",
                str(CO3O4),
                "--remove-atoms",
                "17",
                "--output",
                str(tmp_path / "y.cif"),
                "--spin-output",
                str(tmp_path / "y_spin.fdf"),
            ]
        )
        == 2
    )
    assert not (tmp_path / "x.cif").exists()
