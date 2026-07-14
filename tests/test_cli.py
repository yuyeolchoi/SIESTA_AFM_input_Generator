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
