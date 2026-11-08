import csv
import json
from pathlib import Path

import pytest

from siesta_afm.cli import main
from siesta_afm.defect_spins import (
    ELECTRON_COUNT_WARNING,
    enumerate_defect_spin_candidates,
    format_defect_spin_summary,
    write_defect_spin_candidates,
)
from siesta_afm.io import parse_dm_init_spin, read_structure
from siesta_afm.magnetic_sites import (
    d_electron_count,
    parse_oxidation_state_arguments,
    spin_only_moment,
)
from siesta_afm.validation import validate_spin_file
from siesta_afm.workflows import generate_assignment


ROOT = Path(__file__).parents[1]
CO3O4 = ROOT / "examples" / "Co3O4_spinel_COD1538531.cif"
CUO_SLAB = ROOT / "examples" / "CuO_111_slab.cif"


# --- oxidation-state moment table -----------------------------------------------------


@pytest.mark.parametrize(
    "element, state, spin, expected",
    [
        ("Co", 3, "high", 4.0), ("Co", 3, "low", 0.0),
        ("Co", 2, "high", 3.0), ("Co", 2, "low", 1.0),
        ("Cu", 2, "high", 1.0), ("Cu", 1, "high", 0.0),
        ("Fe", 3, "high", 5.0), ("Fe", 3, "low", 1.0), ("Fe", 2, "high", 4.0),
        ("Mn", 4, "high", 3.0), ("Mn", 2, "high", 5.0),
        ("Ti", 4, "high", 0.0), ("Ti", 3, "high", 1.0),
        ("Ni", 2, "high", 2.0), ("Ni", 3, "low", 1.0),
        ("Co", 2.5, "high", 3.5), ("Co", 2.5, "low", 0.5),
    ],
)
def test_spin_only_moment_table(element: str, state: float, spin: str, expected: float) -> None:
    assert spin_only_moment(element, state, spin) == expected


def test_d_electron_count_validation() -> None:
    assert d_electron_count("Co", 3) == 6.0
    with pytest.raises(ValueError, match="outside d\\^0..d\\^10"):
        d_electron_count("Cu", 0)
    with pytest.raises(ValueError, match="not a d-block element"):
        d_electron_count("O", -2)
    with pytest.raises(ValueError, match="'high' or 'low'"):
        spin_only_moment("Co", 3, "medium")


def test_parse_oxidation_state_arguments() -> None:
    by_element, by_cn = parse_oxidation_state_arguments(["Co@4=+2", "Co@6=+3", "O=-2", "Ni=2"])

    assert by_element == {"o": -2.0, "ni": 2.0}
    assert by_cn == {("co", 4): 2.0, ("co", 6): 3.0}
    with pytest.raises(ValueError, match="use Element=\\+3"):
        parse_oxidation_state_arguments(["Co3"])
    with pytest.raises(ValueError, match="at least one"):
        parse_oxidation_state_arguments([])


# --- enumeration ----------------------------------------------------------------------------


def _co3o4_pristine() -> tuple:
    structure = read_structure(CO3O4)
    _, assignment, spins = generate_assignment(
        structure, ["Co"], "coordination-bipartite", ["Co@4=3.0", "Co@6=0"], anion_species=["O"]
    )
    rows = [(index + 1, value) for index, value in sorted(spins.items())]
    return structure, rows


def test_oxygen_vacancy_in_co3o4_enumerates_schemes_spin_states_and_signs() -> None:
    structure, rows = _co3o4_pristine()
    by_element, by_cn = parse_oxidation_state_arguments(["Co@4=+2", "Co@6=+3"])

    enumeration = enumerate_defect_spin_candidates(
        structure,
        [16],
        magnetic_species=["Co"],
        parent_spin_rows=rows,
        oxidation_by_element=by_element,
        oxidation_by_coordination=by_cn,
        spin_states=["high", "low"],
    )

    assert enumeration.carriers == 2.0
    assert [index + 1 for index in enumeration.acceptors] == [4, 12, 14, 50]
    assert enumeration.pristine_oxidation_states[3] == 3.0
    assert enumeration.pristine_oxidation_states[49] == 2.0
    assert ELECTRON_COUNT_WARNING in enumeration.notices
    schemes = {candidate.scheme for candidate in enumeration.candidates}
    assert schemes == {"localized", "split", "delocalized"}
    # Two electrons split over two octahedral Co3+ give two high-spin Co2+ (3 muB)
    # with both sign choices for the moments that did not exist before.
    split_hs = [
        c for c in enumeration.candidates
        if c.scheme == "split" and c.spin_state == "high" and set(c.acceptors) == {3, 11}
    ]
    assert {c.sign_variant for c in split_hs} == {"new-up", "new-down"}
    up = next(c for c in split_hs if c.sign_variant == "new-up")
    assert up.oxidation_states == {3: 2.0, 11: 2.0}
    assert up.moments[3] == 3.0 and up.moments[11] == 3.0
    assert up.multiplicity == 3  # three equivalent Oh acceptors -> C(3,2) pairs
    # Tetrahedral Co (one-based parent 49-56 -> child 48-55; zero-based keys shift
    # by one) keeps its inherited signed value.
    pristine = dict(rows)
    assert up.moments[47] == pristine[49] and up.moments[48] == pristine[50]
    assert abs(up.moments[47]) == 3.0 and up.moments[47] == -up.moments[48]
    low = next(
        c for c in enumeration.candidates
        if c.scheme == "split" and c.spin_state == "low" and set(c.acceptors) == {3, 11}
    )
    assert low.moments[3] == 1.0  # low-spin Co2+ (d7) on the octahedral site
    # Low spin is not applied to the tetrahedral site.
    assert any("not octahedral" in note for note in enumeration.notices)
    delocalized = [c for c in enumeration.candidates if c.scheme == "delocalized"]
    assert all(c.carriers == {3: 0.5, 11: 0.5, 13: 0.5, 49: 0.5} for c in delocalized)
    ids = [c.config_id for c in enumeration.candidates]
    assert ids == [f"{n:03d}" for n in range(1, len(ids) + 1)]
    assert "Carriers left in the lattice: +2 electrons" in format_defect_spin_summary(enumeration)


def test_cobalt_vacancy_uses_second_shell_acceptors_and_holes() -> None:
    structure, rows = _co3o4_pristine()
    by_element, by_cn = parse_oxidation_state_arguments(["Co@4=+2", "Co@6=+3"])

    enumeration = enumerate_defect_spin_candidates(
        structure,
        [0],  # octahedral Co3+
        magnetic_species=["Co"],
        parent_spin_rows=rows,
        oxidation_by_element=by_element,
        oxidation_by_coordination=by_cn,
        schemes=["localized"],
    )

    assert enumeration.carriers == -3.0
    assert enumeration.acceptors and 0 not in enumeration.acceptors
    assert any("sharing one of its first-shell ligands" in note for note in enumeration.notices)
    assert any("dropped pristine DM.InitSpin rows" in note for note in enumeration.notices)
    # Three holes on one Oh Co3+ would be Co6+ (d3): allowed; on Td Co2+ -> Co5+ (d4): allowed.
    assert all(all(state > 3 for state in c.oxidation_states.values()) for c in enumeration.candidates)


def test_cuo_oxygen_vacancy_kills_copper_moments_and_skips_impossible_localization() -> None:
    structure = read_structure(CUO_SLAB, slab=True)
    _, _, spins = generate_assignment(structure, ["Cu"], "layer", ["1.0"])
    rows = [(index + 1, value) for index, value in sorted(spins.items())]

    enumeration = enumerate_defect_spin_candidates(
        structure, [2], magnetic_species=["Cu"], parent_spin_rows=rows,
        oxidation_by_element={"cu": 2.0},
    )

    assert len(enumeration.candidates) == 1
    only = enumeration.candidates[0]
    assert only.scheme == "split" and only.oxidation_states == {0: 1.0, 1: 1.0}
    assert only.moments[0] == 0.0 and only.moments[1] == 0.0
    assert any("d^11" in note for note in enumeration.notices)
    assert any("remove a pristine moment entirely" in note for note in enumeration.notices)


def test_charge_state_and_validation_errors() -> None:
    structure, rows = _co3o4_pristine()
    by_element, by_cn = parse_oxidation_state_arguments(["Co@4=+2", "Co@6=+3"])

    neutralized = enumerate_defect_spin_candidates(
        structure, [16], magnetic_species=["Co"], parent_spin_rows=rows,
        oxidation_by_element=by_element, oxidation_by_coordination=by_cn, charge=2,
    )
    assert neutralized.carriers == 0.0
    assert len(neutralized.candidates) == 1
    assert neutralized.candidates[0].scheme == "inherited"
    # The inherited pattern is the pristine block renumbered past the removed O 17.
    expected = {
        (index - 1 if index < 17 else index - 2): value for index, value in rows
    }
    assert neutralized.candidates[0].moments == expected
    with pytest.raises(ValueError, match="no oxidation state for magnetic atom"):
        enumerate_defect_spin_candidates(
            structure, [16], magnetic_species=["Co"], parent_spin_rows=rows,
            oxidation_by_element={}, oxidation_by_coordination={("co", 4): 2.0},
        )
    with pytest.raises(ValueError, match="unknown carrier scheme"):
        enumerate_defect_spin_candidates(
            structure, [16], magnetic_species=["Co"], parent_spin_rows=rows,
            oxidation_by_element=by_element, oxidation_by_coordination=by_cn, schemes=["random"],
        )
    with pytest.raises(ValueError, match="non-integer carrier count"):
        enumerate_defect_spin_candidates(
            structure, [16], magnetic_species=["Co"], parent_spin_rows=rows,
            oxidation_by_element={**by_element, "o": -1.5}, oxidation_by_coordination=by_cn,
        )


def test_write_candidates_is_enumerate_compatible(tmp_path: Path) -> None:
    structure, rows = _co3o4_pristine()
    by_element, by_cn = parse_oxidation_state_arguments(["Co@4=+2", "Co@6=+3"])
    enumeration = enumerate_defect_spin_candidates(
        structure, [16], magnetic_species=["Co"], parent_spin_rows=rows,
        oxidation_by_element=by_element, oxidation_by_coordination=by_cn, schemes=["split"],
    )

    paths = write_defect_spin_candidates(enumeration, tmp_path / "cands")

    with paths["manifest"].open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    assert manifest[0].keys() >= {"config_id", "method", "n_up", "n_down", "net_spin", "afm_score", "file"}
    assert all(row["method"] == "defect-electron-count" for row in manifest)
    assert {row["scheme"] for row in manifest} == {"split"}
    for row in manifest:
        block = tmp_path / "cands" / row["file"]
        report = validate_spin_file(str(block), structure=read_structure(paths["structure"]), magnetic_species=["Co"])
        assert report.valid, report.errors
        text = block.read_text(encoding="utf-8")
        assert "# Method: defect-electron-count" in text
    data = json.loads(paths["report"].read_text(encoding="utf-8"))
    assert data["removed_parent_atoms"] == [17] and data["carriers"] == 2.0
    assert data["candidates"][0]["acceptors_child"] == [4, 12]
    assert (tmp_path / "cands" / "index_map.csv").is_file()


# --- CLI -------------------------------------------------------------------------------------


def test_defect_spins_cli_chains_into_prepare_cp2k_array(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pristine = tmp_path / "pristine.fdf"
    assert (
        main(
            [
                "generate", str(CO3O4), "--magnetic-species", "Co", "--method", "coordination-bipartite",
                "--anion-species", "O", "--active-coordination", "4", "--inactive-coordination", "6",
                "--moment", "Co@4=3.0", "Co@6=0", "--output", str(pristine),
            ]
        )
        == 0
    )
    cands = tmp_path / "cands"

    assert (
        main(
            [
                "defect-spins", str(CO3O4), "--remove-atoms", "17", "--magnetic-species", "Co",
                "--spin-file", str(pristine), "--oxidation-states", "Co@4=+2", "Co@6=+3",
                "--schemes", "split", "--spin-states", "high,low", "--new-moment-signs", "up",
                "--output-dir", str(cands),
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    assert "Carriers left in the lattice: +2 electrons" in captured.out
    assert "Wrote 4 candidate spin block(s)" in captured.out
    assert "formal electron counting" in captured.err
    assert parse_dm_init_spin(cands / "afm_001.fdf")
    jobs = tmp_path / "cp2k"
    assert (
        main(
            [
                "prepare-cp2k-array", str(cands / "defect.fdf"), "--magnetic-species", "Co",
                "--configs", str(cands), "--output-dir", str(jobs), "--no-job-script",
            ]
        )
        == 0
    )
    assert sorted(p.name for p in jobs.iterdir() if p.is_dir()) == [
        f"{n:03d}_defect-electron-count" for n in range(1, 5)
    ]


def test_defect_spins_cli_rejects_bad_requests(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    spin_file = tmp_path / "spins.fdf"
    spin_file.write_text("%block DM.InitSpin\n1 1.0\n%endblock DM.InitSpin\n", encoding="utf-8")
    base = [
        "defect-spins", str(CO3O4), "--magnetic-species", "Co", "--spin-file", str(spin_file),
        "--oxidation-states", "Co=+3", "--output-dir", str(tmp_path / "out"),
    ]
    assert main([*base, "--remove-atoms", "999"]) == 2
    assert "out of range: 999" in capsys.readouterr().err
    assert main([*base, "--remove-atoms", "17", "--schemes", "magic"]) == 2
    assert "unknown carrier scheme" in capsys.readouterr().err
