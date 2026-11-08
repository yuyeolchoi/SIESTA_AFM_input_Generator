from pathlib import Path

import numpy as np
import pytest

from siesta_afm.cli import main
from siesta_afm.cp2k_writer import (
    CP2K_U_WARNING,
    max_atomic_magnetization,
    parse_cp2k_coord,
    render_cp2k_input,
)
from siesta_afm.io import read_structure
from siesta_afm.structure import Structure
from siesta_afm.workflows import generate_assignment


ROOT = Path(__file__).parents[1]
CO3O4 = ROOT / "examples" / "Co3O4_spinel_COD1538531.cif"
CUO_SLAB = ROOT / "examples" / "CuO_111_slab.cif"


def _cuo_layer_afm() -> tuple[Structure, dict[int, float], object]:
    structure = read_structure(CUO_SLAB, slab=True)
    _, assignment, spins = generate_assignment(
        structure, ["Cu"], "layer", ["1.0"], axis="z"
    )
    return structure, spins, assignment


# --- magnetization limits -------------------------------------------------------


@pytest.mark.parametrize(
    "element, expected",
    [("Cu", 1.0), ("Co", 3.0), ("Ni", 2.0), ("Fe", 4.0), ("Mn", 5.0), ("Cr", 6.0), ("O", 2.0), ("Zn", 0.0)],
)
def test_max_atomic_magnetization_follows_open_shells(element: str, expected: float) -> None:
    assert max_atomic_magnetization(element) == expected


def test_max_atomic_magnetization_unknown_element_is_none() -> None:
    assert max_atomic_magnetization("Gd") is None


# --- render_cp2k_input --------------------------------------------------------------


def test_render_splits_kinds_by_initial_spin_and_preserves_atom_order() -> None:
    structure, spins, assignment = _cuo_layer_afm()

    result = render_cp2k_input(
        structure,
        spins,
        method=assignment.method,
        magnetic_species=["Cu"],
        metadata=assignment.metadata,
    )

    assert [kind.label for kind in result.kinds] == ["Cu_up", "O", "Cu_dn"]
    by_label = {kind.label: kind for kind in result.kinds}
    assert by_label["Cu_up"].magnetization == 1.0
    assert by_label["Cu_dn"].magnetization == -1.0
    assert by_label["Cu_up"].potential == "GTH-PBE-q11"
    assert by_label["O"].potential == "GTH-PBE-q6"
    assert all(kind.hubbard_u is None for kind in result.kinds)
    rows = parse_cp2k_coord(result.text)
    assert [kind for kind, _ in rows] == list(result.atom_kinds)
    np.testing.assert_allclose(
        np.asarray([position for _, position in rows]), structure.positions, atol=1e-9
    )
    assert "UKS" in result.text
    assert "MAGNETIZATION 1  # 4 atoms" in result.text
    assert "MAGNETIZATION -1  # 2 atoms" in result.text
    assert "&DFT_PLUS_U" not in result.text
    assert "PLUS_U_METHOD" not in result.text
    assert "TYPE DFTD3(BJ)" in result.text
    assert "MINIMIZER CG" in result.text
    assert "PERIODIC XYZ  # siesta-afm periodic axes: XY" in result.text
    assert any("periodic along XY only" in warning for warning in result.warnings)
    # Demo slab: 4 up and 2 down Cu with 102 valence electrons -> net 2 -> triplet.
    assert result.multiplicity == 3
    assert "MULTIPLICITY 3" in result.text


def test_render_hubbard_u_is_opt_in_and_written_per_kind() -> None:
    structure, spins, assignment = _cuo_layer_afm()

    result = render_cp2k_input(
        structure,
        spins,
        method=assignment.method,
        magnetic_species=["Cu"],
        hubbard_u=["Cu=6.0"],
        plus_u_method="LOWDIN",
    )

    assert result.text.count("U_MINUS_J [eV] 6.00") == 2
    assert "        L 2" in result.text
    assert "PLUS_U_METHOD LOWDIN" in result.text
    assert CP2K_U_WARNING in result.warnings
    assert all(
        (kind.hubbard_u == 6.0) == (kind.element == "Cu") for kind in result.kinds
    )


def test_render_rejects_magnetization_the_gth_guess_cannot_polarize() -> None:
    structure, _, _ = _cuo_layer_afm()
    spins = {index: (2.0 if index % 2 else -2.0) for index in (0, 1, 4, 5, 8, 9)}

    with pytest.raises(ValueError, match=r"GTH-PBE-q11 atomic guess.*\|M\| <= 1"):
        render_cp2k_input(structure, spins, method="manual", magnetic_species=["Cu"])

    result = render_cp2k_input(
        structure, spins, method="manual", magnetic_species=["Cu"], magnetization_check=False
    )
    assert "MAGNETIZATION 2" in result.text and "MAGNETIZATION -2" in result.text


def test_render_coordination_kinds_and_cn_specific_u_on_co3o4() -> None:
    structure = read_structure(CO3O4)
    _, assignment, spins = generate_assignment(
        structure,
        ["Co"],
        "coordination-bipartite",
        ["Co@4=3.0", "Co@6=0"],
        anion_species=["O"],
    )

    result = render_cp2k_input(
        structure,
        spins,
        method=assignment.method,
        magnetic_species=["Co"],
        metadata=assignment.metadata,
        hubbard_u=["Co@4=3.0", "Co@6=5.0"],
    )

    kinds = {kind.label: kind for kind in result.kinds}
    assert set(kinds) == {"Co_cn6_0", "O", "Co_cn4_up", "Co_cn4_dn"}
    assert kinds["Co_cn6_0"].coordination == 6 and len(kinds["Co_cn6_0"].atom_indices) == 16
    assert kinds["Co_cn4_up"].hubbard_u == 3.0 and kinds["Co_cn4_dn"].hubbard_u == 3.0
    assert kinds["Co_cn6_0"].hubbard_u == 5.0
    assert kinds["O"].hubbard_u is None
    assert result.multiplicity == 1
    assert "# MAGNETIZATION 0 (16 atoms, CN=6)" in result.text
    assert "# 49 Co CN=4" in result.text


def test_render_parity_fix_and_charge() -> None:
    structure, spins, assignment = _cuo_layer_afm()

    result = render_cp2k_input(
        structure, spins, method=assignment.method, magnetic_species=["Cu"], charge=1
    )

    # 101 electrons cannot carry 2 unpaired electrons; the writer bumps to 3.
    assert result.multiplicity == 4
    assert "CHARGE 1" in result.text
    assert any("MULTIPLICITY raised to 4" in warning for warning in result.warnings)


def test_render_rejects_non_collinear_and_cell_free_structures() -> None:
    structure, spins, assignment = _cuo_layer_afm()
    with pytest.raises(ValueError, match="non-collinear"):
        render_cp2k_input(
            structure,
            spins,
            method="graph-coloring",
            magnetic_species=["Cu"],
            angles={0: (90.0, 0.0)},
        )
    molecule = Structure(["Fe", "O"], [[0.0, 0.0, 0.0], [1.6, 0.0, 0.0]])
    with pytest.raises(ValueError, match="requires a simulation cell"):
        render_cp2k_input(molecule, {0: 4.0}, method="manual", magnetic_species=["Fe"])


def test_render_validates_hubbard_u_targets_and_potentials() -> None:
    structure, spins, assignment = _cuo_layer_afm()
    with pytest.raises(ValueError, match="not a selected magnetic species: O"):
        render_cp2k_input(
            structure, spins, method="layer", magnetic_species=["Cu"], hubbard_u=["O=2.0"]
        )
    with pytest.raises(ValueError, match="requires coordination metadata"):
        render_cp2k_input(
            structure, spins, method="layer", magnetic_species=["Cu"], hubbard_u=["Cu@4=2.0"]
        )
    result = render_cp2k_input(
        structure,
        spins,
        method="layer",
        magnetic_species=["Cu"],
        potentials=["Cu=GTH-PBE-q19"],
    )
    assert "POTENTIAL GTH-PBE-q19" in result.text
    gadolinium = Structure(["Gd", "O"], [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], cell=np.eye(3) * 6, pbc=(True,) * 3)
    with pytest.raises(ValueError, match="no default GTH-PBE potential is known for Gd"):
        render_cp2k_input(gadolinium, {0: 7.0}, method="manual", magnetic_species=["Gd"])


def test_parse_cp2k_coord_requires_a_block() -> None:
    with pytest.raises(ValueError, match="no &COORD block"):
        parse_cp2k_coord("&GLOBAL\n&END GLOBAL\n")


# --- CLI --------------------------------------------------------------------------------


def test_make_cp2k_input_cli_with_method(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output = tmp_path / "cuo.inp"

    assert (
        main(
            [
                "make-cp2k-input",
                str(CUO_SLAB),
                "--slab",
                "--magnetic-species",
                "Cu",
                "--method",
                "layer",
                "--moment",
                "1.0",
                "--hubbard-u",
                "Cu=6.0",
                "--output",
                str(output),
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    assert "CP2K kinds: Cu_up=1 (4 atoms), O=0 (6 atoms), Cu_dn=-1 (2 atoms)" in captured.err
    text = output.read_text(encoding="utf-8")
    assert "PROJECT CuO_111_slab" in text
    assert text.count("&END KIND") == 3
    assert len(parse_cp2k_coord(text)) == 12


def test_make_cp2k_input_cli_from_make_defect_spin_file(tmp_path: Path) -> None:
    pristine = tmp_path / "pristine.fdf"
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
    defect = tmp_path / "defect.fdf"
    assert (
        main(
            [
                "make-defect",
                str(CO3O4),
                "--remove-atoms",
                "17",
                "--output",
                str(defect),
                "--spin-file",
                str(pristine),
            ]
        )
        == 0
    )
    output = tmp_path / "defect.inp"

    assert (
        main(
            [
                "make-cp2k-input",
                str(defect),
                "--magnetic-species",
                "Co",
                "--spin-file",
                str(tmp_path / "defect_spin.fdf"),
                "--output",
                str(output),
            ]
        )
        == 0
    )

    text = output.read_text(encoding="utf-8")
    rows = parse_cp2k_coord(text)
    assert len(rows) == 55
    tetrahedral = [kind for kind, _ in rows[47:55]]  # parent Co 49-56 shifted to 48-55
    assert sorted(tetrahedral) == ["Co_dn"] * 4 + ["Co_up"] * 4
    assert {kind for kind, _ in rows[:16]} == {"Co_0"}
    assert "MULTIPLICITY 1" in text
    assert "Spin method: spin-file:defect_spin.fdf" in text


def test_make_cp2k_input_cli_argument_validation(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    spin_file = tmp_path / "spins.fdf"
    spin_file.write_text("%block DM.InitSpin\n1 1.0\n%endblock DM.InitSpin\n", encoding="utf-8")

    assert main(["make-cp2k-input", str(CUO_SLAB), "--magnetic-species", "Cu"]) == 2
    assert "exactly one of --method or --spin-file" in capsys.readouterr().err
    assert (
        main(
            [
                "make-cp2k-input",
                str(CUO_SLAB),
                "--magnetic-species",
                "Cu",
                "--spin-file",
                str(spin_file),
                "--moment",
                "1.0",
            ]
        )
        == 2
    )
    assert "cannot be combined with --spin-file" in capsys.readouterr().err
    bad = tmp_path / "bad.fdf"
    bad.write_text("%block DM.InitSpin\n3 1.0\n%endblock DM.InitSpin\n", encoding="utf-8")
    assert (
        main(["make-cp2k-input", str(CUO_SLAB), "--magnetic-species", "Cu", "--spin-file", str(bad)])
        == 2
    )
    assert "atom 3 (O), which is not in --magnetic-species" in capsys.readouterr().err
