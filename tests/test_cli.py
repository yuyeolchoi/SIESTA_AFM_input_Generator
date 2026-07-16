import csv
import sys
from pathlib import Path

import numpy as np
import pytest

from siesta_afm.cli import _workflow_kwargs, build_parser, main
from siesta_afm.io import parse_dm_init_spin, read_structure


ROOT = Path(__file__).parents[1]


def test_make_input_cli_writes_roundtrippable_template_and_warning(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "complete.fdf"
    source = ROOT / "tests" / "fixtures" / "inverse_spinel_coordination.cif"
    assert (
        main(
            [
                "make-input",
                str(source),
                "--slab",
                "--magnetic-species",
                "Ni",
                "Co",
                "--method",
                "by-coordination",
                "--anion-species",
                "O",
                "--kgrid",
                "4",
                "5",
                "1",
                "--hubbard-u",
                "Ni=6.0",
                "Co=3.3",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    original = read_structure(source, slab=True)
    generated = read_structure(output)
    assert generated.symbols == original.symbols
    assert np.allclose(generated.positions, original.positions)
    assert len(parse_dm_init_spin(output)) == 6
    text = output.read_text(encoding="utf-8")
    assert "# Selected k-grid 4 5 1: explicit --kgrid override." in text
    stderr = capsys.readouterr().err
    assert "using built-in default initial moments" in stderr
    assert "starting template only" in stderr


def test_make_input_cli_splits_coordination_species_and_hubbard_u(
    tmp_path: Path,
) -> None:
    source = ROOT / "tests" / "fixtures" / "inverse_spinel_coordination.cif"
    output = tmp_path / "split_complete.fdf"
    assert (
        main(
            [
                "make-input",
                str(source),
                "--slab",
                "--magnetic-species",
                "Ni",
                "Co",
                "--method",
                "by-coordination",
                "--anion-species",
                "O",
                "--split-species-by-coordination",
                "--hubbard-u",
                "Ni@6=6.0",
                "Co@4=3.0",
                "Co@6=5.0",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    original = read_structure(source, slab=True)
    generated = read_structure(output)
    assert generated.symbols == original.symbols
    assert np.allclose(generated.positions, original.positions)
    assert generated.species_ids == [1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4]
    text = output.read_text(encoding="utf-8")
    assert "Co_2  # Co CN=6 (Oh)" in text
    assert "Co_3  # Co CN=4 (Td)" in text


def test_make_input_coordination_split_rejects_invalid_combinations(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = ROOT / "tests" / "fixtures" / "inverse_spinel_coordination.cif"
    common = [
        "make-input",
        str(source),
        "--slab",
        "--magnetic-species",
        "Co",
    ]
    assert (
        main(
            [
                *common,
                "--method",
                "by-coordination",
                "--anion-species",
                "O",
                "--hubbard-u",
                "Co@4=3.0",
                "--output",
                str(tmp_path / "missing_flag.fdf"),
            ]
        )
        == 2
    )
    assert "@CN Hubbard U requires --split-species-by-coordination" in (
        capsys.readouterr().err
    )

    assert (
        main(
            [
                *common,
                "--method",
                "alternating-index",
                "--split-species-by-coordination",
                "--output",
                str(tmp_path / "wrong_method.fdf"),
            ]
        )
        == 2
    )
    assert (
        "--split-species-by-coordination requires --method by-coordination"
        in capsys.readouterr().err
    )


def test_generate_patch_in_place_rejects_different_output(
    tmp_path: Path, capsys
) -> None:
    source = tmp_path / "input.fdf"
    source.write_text(
        (ROOT / "examples" / "input.fdf").read_text(encoding="utf-8"), encoding="utf-8"
    )
    code = main(
        [
            "generate",
            str(source),
            "--magnetic-species",
            "Cu",
            "--method",
            "layer",
            "--moment",
            "0.5",
            "--patch-input",
            "--in-place",
            "--output",
            str(tmp_path / "other.fdf"),
        ]
    )
    assert code == 2
    assert "--in-place cannot be combined" in capsys.readouterr().err


def test_enumerate_graph_failure_leaves_no_partial_spin_files(tmp_path: Path) -> None:
    output = tmp_path / "configs"
    code = main(
        [
            "enumerate",
            str(ROOT / "examples" / "input.fdf"),
            "--magnetic-species",
            "Cu",
            "--moment",
            "0.5",
            "--methods",
            "layer",
            "--neighbor-shell",
            "99",
            "--n-configs",
            "1",
            "--output-dir",
            str(output),
        ]
    )
    assert code == 2
    assert not list(output.glob("afm_*.fdf"))
    assert not (output / "manifest.csv").exists()


def test_enumerate_cli_preserves_generated_and_shortfall_messages(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    structure = tmp_path / "pair.xyz"
    structure.write_text(
        "2\nCu pair\nCu 0 0 0\nCu 1 0 0\n",
        encoding="utf-8",
    )
    output = tmp_path / "configs"
    assert (
        main(
            [
                "enumerate",
                str(structure),
                "--magnetic-species",
                "Cu",
                "--moment",
                "1",
                "--methods",
                "random",
                "--cutoff",
                "1.1",
                "--n-configs",
                "4",
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert captured.out == f"Generated 2 distinct configuration(s) in {output}\n"
    assert (
        "WARNING: requested 4, but only 2 distinct patterns were found."
        in captured.err
    )


def test_generate_validate_patch_roundtrip(tmp_path: Path) -> None:
    source = ROOT / "examples" / "input.fdf"
    spin_file = tmp_path / "spin.fdf"
    patched_file = tmp_path / "input_afm.fdf"
    assert (
        main(
            [
                "generate",
                str(source),
                "--magnetic-species",
                "Cu",
                "--method",
                "layer",
                "--moment",
                "0.5",
                "--output",
                str(spin_file),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "validate",
                str(spin_file),
                "--structure",
                str(source),
                "--magnetic-species",
                "Cu",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "patch",
                str(source),
                "--spin-file",
                str(spin_file),
                "--output",
                str(patched_file),
            ]
        )
        == 0
    )
    assert parse_dm_init_spin(patched_file) == [(1, 0.5), (3, -0.5)]
    assert patched_file.read_text(encoding="utf-8").count("Spin polarized") == 1


def test_cell_free_homogeneous_catalyst_xyz_end_to_end(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("matplotlib")
    source = ROOT / "examples" / "Fe_CO5_homogeneous_catalyst.xyz"
    structure = read_structure(source)
    assert structure.pbc == (False, False, False)
    assert np.allclose(structure.cell, 0.0)

    assert (
        main(["analyze", str(source), "--magnetic-species", "Fe"])
        == 0
    )
    assert "Number of atoms: 11" in capsys.readouterr().out

    spin_file = tmp_path / "fe_co5_spin.fdf"
    assert (
        main(
            [
                "generate",
                str(source),
                "--magnetic-species",
                "Fe",
                "--method",
                "manual-spins",
                "--spin-values",
                "1=+2.0",
                "--output",
                str(spin_file),
            ]
        )
        == 0
    )
    assert parse_dm_init_spin(spin_file) == [(1, 2.0)]
    assert (
        main(
            [
                "validate",
                str(spin_file),
                "--structure",
                str(source),
                "--magnetic-species",
                "Fe",
            ]
        )
        == 0
    )
    assert "Valid: True" in capsys.readouterr().out

    image = tmp_path / "fe_co5_spin.png"
    assert (
        main(
            [
                "plot",
                str(source),
                "--spin-file",
                str(spin_file),
                "--output",
                str(image),
                "--show-bonds",
            ]
        )
        == 0
    )
    assert image.is_file() and image.stat().st_size > 0


def test_plot_value_color_mode_and_sign_color_warning(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("matplotlib")
    site_moments = tmp_path / "site_moments.csv"
    site_moments.write_text(
        "atom_index,element,moment\n1,Cu,0.7\n3,Cu,0.5\n", encoding="utf-8"
    )
    spin_file = tmp_path / "spin_values.fdf"
    assert (
        main(
            [
                "generate",
                str(ROOT / "examples" / "input.fdf"),
                "--magnetic-species",
                "Cu",
                "--method",
                "layer",
                "--moment",
                "0.6",
                "--site-moment-file",
                str(site_moments),
                "--output",
                str(spin_file),
            ]
        )
        == 0
    )
    assert {abs(value) for _, value in parse_dm_init_spin(spin_file)} == {0.5, 0.7}
    output = tmp_path / "spin_values.png"
    code = main(
        [
            "plot",
            str(ROOT / "examples" / "input.fdf"),
            "--spin-file",
            str(spin_file),
            "--output",
            str(output),
            "--color-mode",
            "value",
            "--up-color",
            "green",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert output.is_file()
    assert "ignored in value color mode" in captured.err


def test_plot_filters_spin_elements_and_draws_bonds(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    spin_file = tmp_path / "nio_spins.fdf"
    spin_file.write_text(
        "%block DM.InitSpin\n  1  1.0\n  2 -1.0\n%endblock DM.InitSpin\n",
        encoding="utf-8",
    )
    output = tmp_path / "nio_filtered_bonds.png"

    assert (
        main(
            [
                "plot",
                str(ROOT / "examples" / "NiO_111_slab.cif"),
                "--spin-file",
                str(spin_file),
                "--output",
                str(output),
                "--filter-elements",
                "Ni",
                "--show-bonds",
                "--bond-radius-scale",
                "1.1",
            ]
        )
        == 0
    )
    assert output.is_file()
    assert output.stat().st_size > 0


def test_q_vector_and_afm_type_are_mutually_exclusive() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "generate",
                "structure.cif",
                "--magnetic-species",
                "Ni",
                "--method",
                "propagation-vector",
                "--moment",
                "1",
                "--q-vector",
                "0.5",
                "0.5",
                "0.5",
                "--afm-type",
                "G",
            ]
        )


def test_layer_axis_and_direction_are_mutually_exclusive() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "generate",
                "structure.cif",
                "--magnetic-species",
                "Ni",
                "--method",
                "layer",
                "--moment",
                "1",
                "--axis",
                "z",
                "--layer-direction",
                "1",
                "1",
                "1",
            ]
        )


def test_axis_default_is_z_when_neither_axis_nor_layer_direction_is_passed() -> None:
    # Regression guard: --axis defaults to None (not "z") at the argparse
    # level so the mutually-exclusive-group conflict above is actually
    # detected on every Python version. "z" is restored downstream via
    # `args.axis or "z"`; this confirms that normalization still happens.
    parser = build_parser()
    args = parser.parse_args(
        [
            "generate",
            "structure.cif",
            "--magnetic-species",
            "Ni",
            "--method",
            "layer",
            "--moment",
            "1",
        ]
    )
    assert args.axis is None
    assert _workflow_kwargs(args)["axis"] == "z"


def test_axis_alone_without_layer_direction_still_parses() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "generate",
            "structure.cif",
            "--magnetic-species",
            "Ni",
            "--method",
            "layer",
            "--moment",
            "1",
            "--axis",
            "x",
        ]
    )
    assert _workflow_kwargs(args)["axis"] == "x"


def test_layer_per_species_cli_generates_independent_stacks(
    tmp_path: Path,
) -> None:
    structure = tmp_path / "alternating_species.xyz"
    structure.write_text(
        "4\ninterleaved Ni and Co layers\n"
        "Ni 0 0 0\n"
        "Co 0 1 0\n"
        "Ni 0 2 0\n"
        "Co 0 3 0\n",
        encoding="utf-8",
    )
    output = tmp_path / "per_species.fdf"
    assert (
        main(
            [
                "generate",
                str(structure),
                "--magnetic-species",
                "Ni",
                "Co",
                "--method",
                "layer",
                "--axis",
                "y",
                "--layer-per-species",
                "--moment",
                "1",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert parse_dm_init_spin(output) == [
        (1, 1.0),
        (2, 1.0),
        (3, -1.0),
        (4, -1.0),
    ]


def test_layer_per_species_cli_rejects_non_layer_method(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    structure = tmp_path / "two_atoms.xyz"
    structure.write_text(
        "2\ntwo Ni atoms\nNi 0 0 0\nNi 0 0 1\n",
        encoding="utf-8",
    )
    code = main(
        [
            "generate",
            str(structure),
            "--magnetic-species",
            "Ni",
            "--method",
            "alternating-index",
            "--layer-per-species",
            "--moment",
            "1",
        ]
    )
    assert code == 2
    assert (
        "--layer-per-species requires --method layer" in capsys.readouterr().err
    )


def _write_triangle_xyz(path: Path) -> None:
    root3 = 3.0**0.5
    path.write_text(
        "3\ntriangular Cu graph\n"
        "Cu 0.0 0.0 0.0\n"
        "Cu 1.0 0.0 0.0\n"
        f"Cu 0.5 {root3 / 2:.12f} 0.0\n",
        encoding="utf-8",
    )


def _write_symmetric_triplet_cif(path: Path) -> None:
    path.write_text(
        "data_symmetric_triplet\n"
        "_symmetry_space_group_name_H-M 'P 1'\n"
        "_cell_length_a 2.0\n"
        "_cell_length_b 2.0\n"
        "_cell_length_c 2.0\n"
        "_cell_angle_alpha 90\n"
        "_cell_angle_beta 90\n"
        "_cell_angle_gamma 90\n"
        "loop_\n"
        "_atom_site_label\n"
        "_atom_site_type_symbol\n"
        "_atom_site_fract_x\n"
        "_atom_site_fract_y\n"
        "_atom_site_fract_z\n"
        "Cu1 Cu 0.0 0.5 0.5\n"
        "Cu2 Cu 0.5 0.0 0.5\n"
        "Cu3 Cu 0.5 0.5 0.0\n",
        encoding="utf-8",
    )


def test_enumerate_symmetry_dedup_cli_smoke(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("spglib")
    structure = tmp_path / "symmetric_triplet.cif"
    _write_symmetric_triplet_cif(structure)
    output = tmp_path / "configs"

    code = main(
        [
            "enumerate",
            str(structure),
            "--magnetic-species",
            "Cu",
            "--moment",
            "1",
            "--methods",
            "random",
            "--cutoff",
            "1.5",
            "--n-configs",
            "8",
            "--symmetry-dedup",
            "--symprec",
            "0.0002",
            "--output-dir",
            str(output),
        ]
    )

    assert code == 0
    with (output / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert "Generated 2 distinct configuration(s)" in capsys.readouterr().out


def test_enumerate_symmetry_dedup_cli_reports_missing_spglib(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    structure = tmp_path / "symmetric_triplet.cif"
    _write_symmetric_triplet_cif(structure)
    monkeypatch.setitem(sys.modules, "spglib", None)

    code = main(
        [
            "enumerate",
            str(structure),
            "--magnetic-species",
            "Cu",
            "--moment",
            "1",
            "--methods",
            "random",
            "--cutoff",
            "1.5",
            "--n-configs",
            "2",
            "--symmetry-dedup",
            "--output-dir",
            str(tmp_path / "missing"),
        ]
    )

    assert code == 2
    assert "--symmetry-dedup requires the optional spglib dependency" in (
        capsys.readouterr().err
    )


def test_graph_coloring_cli_reports_warning_and_mapping_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    structure = tmp_path / "triangle.xyz"
    _write_triangle_xyz(structure)
    output = tmp_path / "colored.fdf"
    code = main(
        [
            "generate",
            str(structure),
            "--magnetic-species",
            "Cu",
            "--method",
            "graph-coloring",
            "--moment",
            "1",
            "--cutoff",
            "1.01",
            "--output",
            str(output),
        ]
    )
    assert code == 0
    assert {spin for _, spin in parse_dm_init_spin(output)} == {-1.0, 0.0, 1.0}
    assert "does not minimize magnetic energy" in capsys.readouterr().err

    code = main(
        [
            "generate",
            str(structure),
            "--magnetic-species",
            "Cu",
            "--method",
            "graph-coloring",
            "--moment",
            "1",
            "--cutoff",
            "1.01",
            "--color-spins",
            "+1,-1",
        ]
    )
    assert code == 2
    assert "does not match 3 graph colors" in capsys.readouterr().err


def test_noncollinear_spin_mode_requires_graph_coloring_and_no_color_map(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    structure = tmp_path / "triangle.xyz"
    _write_triangle_xyz(structure)

    code = main(
        [
            "generate",
            str(structure),
            "--magnetic-species",
            "Cu",
            "--method",
            "alternating-index",
            "--moment",
            "1",
            "--spin-mode",
            "non-collinear",
        ]
    )
    assert code == 2
    assert "only supported with --method graph-coloring" in capsys.readouterr().err

    code = main(
        [
            "generate",
            str(structure),
            "--magnetic-species",
            "Cu",
            "--method",
            "graph-coloring",
            "--moment",
            "1",
            "--spin-mode",
            "non-collinear",
            "--color-spins",
            "+1,-1,0",
        ]
    )
    assert code == 2
    assert "cannot be combined with --color-spins" in capsys.readouterr().err


def test_graph_coloring_noncollinear_cli_maps_three_colors_to_120_degrees(
    tmp_path: Path,
) -> None:
    structure = tmp_path / "triangle.xyz"
    _write_triangle_xyz(structure)
    output = tmp_path / "noncollinear.fdf"

    assert (
        main(
            [
                "generate",
                str(structure),
                "--magnetic-species",
                "Cu",
                "--method",
                "graph-coloring",
                "--moment",
                "1",
                "--cutoff",
                "1.01",
                "--max-colors",
                "3",
                "--spin-mode",
                "non-collinear",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    rows = parse_dm_init_spin(output, include_angles=True)
    assert {moment for _, moment, _, _ in rows} == {1.0}
    assert {theta for _, _, theta, _ in rows} == {90.0}
    assert sorted({phi for _, _, _, phi in rows}) == pytest.approx(
        [0.0, 120.0, 240.0]
    )
    assert "Spin non-collinear" in output.read_text(encoding="utf-8")


def test_manual_spin_cli_group_defaults_and_mutual_exclusion() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "generate",
            "molecule.xyz",
            "--magnetic-species",
            "Fe",
            "--method",
            "manual-spins",
        ]
    )
    assert args.spin_values is None
    assert args.spin_values_file is None
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "generate",
                "molecule.xyz",
                "--magnetic-species",
                "Fe",
                "--method",
                "manual-spins",
                "--spin-values",
                "1=+4.0",
                "--spin-values-file",
                "spins.csv",
            ]
        )


def test_manual_spins_is_not_an_enumeration_method(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "enumerate",
            str(ROOT / "tests" / "fixtures" / "FeN6_molecule.xyz"),
            "--magnetic-species",
            "Fe",
            "--methods",
            "manual-spins",
            "--n-configs",
            "1",
            "--output-dir",
            str(tmp_path / "configs"),
        ]
    )
    assert code == 2
    assert "unsupported enumeration method: manual-spins" in capsys.readouterr().err


def test_manual_spins_cli_signed_roundtrip_errors_fill_and_csv(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    structure = tmp_path / "fe_pair.xyz"
    structure.write_text(
        "3\nFe pair with ligand\nFe 0 0 0\nO 1.8 0 0\nFe 3.6 0 0\n",
        encoding="utf-8",
    )
    direct = tmp_path / "direct.fdf"
    assert (
        main(
            [
                "generate",
                str(structure),
                "--magnetic-species",
                "Fe",
                "--method",
                "manual-spins",
                "--spin-values",
                "1=+4.0",
                "3=-2.0",
                "--output",
                str(direct),
            ]
        )
        == 0
    )
    assert parse_dm_init_spin(direct) == [(1, 4.0), (3, -2.0)]
    assert "Method: manual-spins" in direct.read_text(encoding="utf-8")

    assert (
        main(
            [
                "generate",
                str(structure),
                "--magnetic-species",
                "Fe",
                "--method",
                "manual-spins",
                "--spin-values",
                "1=4.0",
            ]
        )
        == 2
    )
    assert "manual spins omit magnetic atom 3 (Fe)" in capsys.readouterr().err

    filled = tmp_path / "filled.fdf"
    assert (
        main(
            [
                "generate",
                str(structure),
                "--magnetic-species",
                "Fe",
                "--method",
                "manual-spins",
                "--spin-values",
                "1=4.0",
                "--fill-unspecified-zero",
                "--output",
                str(filled),
            ]
        )
        == 0
    )
    assert parse_dm_init_spin(filled) == [(1, 4.0), (3, 0.0)]

    for specification, message in (
        ("4=1.0", "manual spin atom index out of range: 4"),
        ("2=1.0", "manual spin atom 2 is not selected by --magnetic-species"),
    ):
        assert (
            main(
                [
                    "generate",
                    str(structure),
                    "--magnetic-species",
                    "Fe",
                    "--method",
                    "manual-spins",
                    "--spin-values",
                    "1=4.0",
                    "3=-2.0",
                    specification,
                ]
            )
            == 2
        )
        error = capsys.readouterr().err
        assert message in error
        if specification.startswith("2="):
            assert "actual element: O" in error

    csv_path = tmp_path / "spins.csv"
    csv_path.write_text(
        "atom_index,spin\n1,+3.5\n3,-1.25\n", encoding="utf-8"
    )
    csv_output = tmp_path / "csv.fdf"
    assert (
        main(
            [
                "generate",
                str(structure),
                "--magnetic-species",
                "Fe",
                "--method",
                "manual-spins",
                "--spin-values-file",
                str(csv_path),
                "--output",
                str(csv_output),
            ]
        )
        == 0
    )
    assert parse_dm_init_spin(csv_output) == [(1, 3.5), (3, -1.25)]


def test_site_moment_file_still_ignores_negative_sign(tmp_path: Path) -> None:
    structure = tmp_path / "fe_pair.xyz"
    structure.write_text(
        "3\nFe pair with ligand\nFe 0 0 0\nO 1.8 0 0\nFe 3.6 0 0\n",
        encoding="utf-8",
    )
    moments = tmp_path / "moments.csv"
    moments.write_text(
        "atom_index,moment\n1,-4.0\n3,-2.0\n", encoding="utf-8"
    )
    output = tmp_path / "legacy.fdf"
    assert (
        main(
            [
                "generate",
                str(structure),
                "--magnetic-species",
                "Fe",
                "--method",
                "manual-groups",
                "--up-atoms",
                "1",
                "--down-atoms",
                "3",
                "--moment",
                "4.0",
                "--site-moment-file",
                str(moments),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert parse_dm_init_spin(output) == [(1, 4.0), (3, -2.0)]


def test_explicit_collinear_spin_mode_matches_the_default_byte_for_byte(
    tmp_path: Path,
) -> None:
    structure = tmp_path / "triangle.xyz"
    _write_triangle_xyz(structure)
    implicit = tmp_path / "implicit.fdf"
    explicit = tmp_path / "explicit.fdf"
    common = [
        "generate",
        str(structure),
        "--magnetic-species",
        "Cu",
        "--method",
        "graph-coloring",
        "--moment",
        "1",
        "--cutoff",
        "1.01",
    ]

    assert main([*common, "--output", str(implicit)]) == 0
    assert main([*common, "--spin-mode", "collinear", "--output", str(explicit)]) == 0
    assert explicit.read_bytes() == implicit.read_bytes()


def test_make_input_supports_noncollinear_graph_coloring(
    tmp_path: Path,
) -> None:
    structure = tmp_path / "triangle.xyz"
    _write_triangle_xyz(structure)
    output = tmp_path / "complete_noncollinear.fdf"

    assert (
        main(
            [
                "make-input",
                str(structure),
                "--magnetic-species",
                "Cu",
                "--method",
                "graph-coloring",
                "--moment",
                "1",
                "--cutoff",
                "1.01",
                "--spin-mode",
                "non-collinear",
                "--no-lda-u",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert "Spin non-collinear" in output.read_text(encoding="utf-8")
    assert all(
        theta == 90.0
        for _, _, theta, _ in parse_dm_init_spin(output, include_angles=True)
    )


def test_enumerate_graph_coloring_varies_color_spin_permutation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    structure = tmp_path / "triangle.xyz"
    _write_triangle_xyz(structure)
    output = tmp_path / "configs"
    assert (
        main(
            [
                "enumerate",
                str(structure),
                "--magnetic-species",
                "Cu",
                "--moment",
                "1",
                "--methods",
                "graph-coloring",
                "--cutoff",
                "1.01",
                "--n-configs",
                "2",
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    patterns = {
        tuple(spin for _, spin in parse_dm_init_spin(path))
        for path in output.glob("afm_*.fdf")
    }
    assert len(patterns) >= 2
    with (output / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert all(float(row["afm_score"]) == pytest.approx(1 / 3) for row in rows)
    assert "does not minimize magnetic energy" in capsys.readouterr().err


def test_by_coordination_cli_warns_for_ambiguous_element_moment(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "inverse_spinel.fdf"
    code = main(
        [
            "generate",
            str(ROOT / "tests" / "fixtures" / "inverse_spinel_coordination.cif"),
            "--slab",
            "--magnetic-species",
            "Ni",
            "Co",
            "--method",
            "by-coordination",
            "--anion-species",
            "O",
            "--moment",
            "Ni=2.0",
            "Co=2.0",
            "--output",
            str(output),
        ]
    )
    assert code == 0
    assert output.exists()
    stderr = capsys.readouterr().err
    assert "element Co occupies both CN=4 and CN=6 sites" in stderr
    assert "Co@4=... and Co@6=..." in stderr


def test_generate_without_moment_uses_defaults_and_site_comments(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "defaults.fdf"
    code = main(
        [
            "generate",
            str(ROOT / "tests" / "fixtures" / "inverse_spinel_coordination.cif"),
            "--slab",
            "--magnetic-species",
            "Ni",
            "Co",
            "--method",
            "by-coordination",
            "--anion-species",
            "O",
            "--output",
            str(output),
        ]
    )
    assert code == 0
    assert {abs(value) for _, value in parse_dm_init_spin(output)} == {2.0, 3.0}
    text = output.read_text(encoding="utf-8")
    assert "# Ni  (Oh, CN=6)" in text
    assert "# Co  (Td, CN=4)" in text
    stderr = capsys.readouterr().err
    assert "using built-in default initial moments" in stderr
    assert "Ni=2.0, Co=3.0" in stderr
    assert "pass --moment to set them explicitly" in stderr


def test_generate_site_comments_can_be_disabled(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "legacy_rows.fdf"
    assert (
        main(
            [
                "generate",
                str(ROOT / "examples" / "input.fdf"),
                "--magnetic-species",
                "Cu",
                "--method",
                "layer",
                "--moment",
                "0.5",
                "--no-site-comments",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    block = output.read_text(encoding="utf-8").split("%block DM.InitSpin", 1)[1]
    block = block.split("%endblock DM.InitSpin", 1)[0]
    assert "#" not in block
    assert "built-in default" not in capsys.readouterr().err


def test_omitted_and_partial_moment_errors_are_actionable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ru_structure = tmp_path / "ru.xyz"
    ru_structure.write_text("1\nRu\nRu 0 0 0\n", encoding="utf-8")
    assert (
        main(
            [
                "generate",
                str(ru_structure),
                "--magnetic-species",
                "Ru",
                "--method",
                "alternating-index",
            ]
        )
        == 2
    )
    assert "no built-in default for element Ru; pass --moment Ru=..." in (
        capsys.readouterr().err
    )

    assert (
        main(
            [
                "generate",
                str(ROOT / "tests" / "fixtures" / "inverse_spinel_coordination.cif"),
                "--slab",
                "--magnetic-species",
                "Ni",
                "Co",
                "--method",
                "by-coordination",
                "--anion-species",
                "O",
                "--moment",
                "Ni=2.0",
            ]
        )
        == 2
    )
    partial_error = capsys.readouterr().err
    assert "no initial moment specified for magnetic atom 3 (Co@6)" in partial_error
    assert "the built-in default is Co=3.0" in partial_error
    assert "pass --moment Co@6=... or omit --moment" in partial_error
