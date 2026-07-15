import csv
from pathlib import Path

import pytest

from siesta_afm.cli import build_parser, main
from siesta_afm.io import parse_dm_init_spin


ROOT = Path(__file__).parents[1]


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


def _write_triangle_xyz(path: Path) -> None:
    root3 = 3.0**0.5
    path.write_text(
        "3\ntriangular Cu graph\n"
        "Cu 0.0 0.0 0.0\n"
        "Cu 1.0 0.0 0.0\n"
        f"Cu 0.5 {root3 / 2:.12f} 0.0\n",
        encoding="utf-8",
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
