from pathlib import Path

import pytest

from siesta_afm.results import (
    _find_output,
    collect_results,
    parse_eig_band_gap,
    parse_local_spins,
    prepare_array,
)


FIXTURES = Path(__file__).parent / "fixtures"


def test_prepare_array_and_collect_synthetic_result(tmp_path: Path) -> None:
    input_fdf = tmp_path / "input.fdf"
    input_fdf.write_text("SystemName test\nSpinPolarized false\n", encoding="utf-8")
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "afm_001.fdf").write_text(
        "Spin polarized\n%block DM.InitSpin\n1 0.6\n2 -0.6\n%endblock DM.InitSpin\n",
        encoding="utf-8",
    )
    (configs / "manifest.csv").write_text(
        "config_id,method,n_up,n_down,net_spin,afm_score,file\n"
        "001,layer,1,1,0.0,1.0,afm_001.fdf\n",
        encoding="utf-8",
    )
    jobs = tmp_path / "jobs"
    folders = prepare_array(input_fdf, configs, jobs)
    assert len(folders) == 1
    assert (folders[0] / "input.fdf").is_file()
    assert (folders[0] / "RUN.fdf").read_text(
        encoding="utf-8"
    ) == "%include input.fdf\n"
    assert (jobs / "folders.list").read_text(encoding="utf-8") == "001_layer\n"
    (folders[0] / "RUN.out").write_text(
        (FIXTURES / "siesta_5_atomic_populations.out").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    rows = collect_results(jobs)
    assert rows[0]["total_energy"] == -876.5
    assert rows[0]["final_net_spin"] == 0.25
    assert rows[0]["sign_retention"] == 1.0
    assert rows[0]["collapsed_atoms"] == 1
    assert rows[0]["spin_population_source"] == "hirshfeld-atomic-populations"
    assert rows[0]["status"] == "ok"


def test_prepare_array_flattens_split_base_fdf(tmp_path: Path) -> None:
    included = tmp_path / "base-settings.fdf"
    included.write_text(
        "SystemName split-array\nSpin non-polarized\n", encoding="utf-8"
    )
    root = tmp_path / "input.fdf"
    root.write_text("%include base-settings.fdf\n", encoding="utf-8")
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "afm_001.fdf").write_text(
        "%block DM.InitSpin\n1 0.5\n2 -0.5\n%endblock DM.InitSpin\n",
        encoding="utf-8",
    )
    (configs / "manifest.csv").write_text(
        "config_id,method,file\n001,layer,afm_001.fdf\n", encoding="utf-8"
    )

    folders = prepare_array(root, configs, tmp_path / "jobs")

    generated = (folders[0] / "input.fdf").read_text(encoding="utf-8")
    assert "%include" not in generated.lower()
    assert "SystemName split-array" in generated
    assert generated.count("Spin polarized") == 1
    assert included.read_text(encoding="utf-8") == (
        "SystemName split-array\nSpin non-polarized\n"
    )


def test_parse_siesta_4_1_spin_resolved_mulliken() -> None:
    text = (FIXTURES / "siesta_4_1_mulliken.out").read_text(encoding="utf-8")
    spins, source = parse_local_spins(text)
    assert source == "mulliken-spin-channels"
    assert spins == {1: 4.708, 2: -4.708, 3: 0.0, 4: 0.0}


def test_parse_siesta_4_1_hirshfeld_net_atomic_populations() -> None:
    text = (FIXTURES / "siesta_4_1_hirshfeld_net.out").read_text(encoding="utf-8")
    spins, source = parse_local_spins(text)
    assert source == "hirshfeld-atomic-populations"
    assert spins == {1: 3.65, 2: -3.55}


def test_missing_spin_table_has_explicit_diagnostic(tmp_path: Path) -> None:
    job = tmp_path / "001_layer"
    job.mkdir()
    (tmp_path / "folders.list").write_text("001_layer\n", encoding="utf-8")
    (job / "input.fdf").write_text(
        "%block DM.InitSpin\n1 0.6\n%endblock DM.InitSpin\n", encoding="utf-8"
    )
    (job / "RUN.out").write_text(
        "siesta: E_KS(eV) = -1.0\nSCF cycle converged\n", encoding="utf-8"
    )
    row = collect_results(tmp_path)[0]
    assert row["status"] == "spin-table-not-found"
    assert row["sign_retention"] is None
    assert row["collapsed_atoms"] is None


def test_find_output_uses_latest_mtime(tmp_path: Path) -> None:
    older = tmp_path / "z.out"
    newer = tmp_path / "a.log"
    older.write_text("old", encoding="utf-8")
    newer.write_text("new", encoding="utf-8")
    older.touch()
    newer.touch()
    import os

    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))
    assert _find_output(tmp_path) == newer


def test_parse_eig_band_gap_reports_direct_indirect_and_metallic() -> None:
    indirect = parse_eig_band_gap(
        "0.0\n4 1 2\n1 -3 -1 2 4\n2 -4 -2 1 3\n"
    )
    assert indirect.gap_ev == 2.0
    assert indirect.kind == "indirect"
    assert indirect.vbm_ev == -1.0
    assert indirect.cbm_ev == 1.0

    direct = parse_eig_band_gap("0.0\n2 1 2\n1 -1 1\n2 -2 2\n")
    assert direct.gap_ev == 2.0
    assert direct.kind == "direct"

    metallic = parse_eig_band_gap("0.0\n2 1 1\n1 0.0 1.0\n")
    assert metallic.gap_ev == 0.0
    assert metallic.kind == "metallic"

    sampled_crossing = parse_eig_band_gap(
        "0.0\n2 1 2\n1 -0.2 1.0\n2 0.3 1.2\n"
    )
    assert sampled_crossing.gap_ev == 0.0
    assert sampled_crossing.kind == "metallic"


def test_collect_results_joins_u_manifest_and_eig_gap(tmp_path: Path) -> None:
    job = tmp_path / "U_6.0"
    job.mkdir()
    (tmp_path / "folders.list").write_text("U_6.0\n", encoding="utf-8")
    (tmp_path / "u_sweep_manifest.csv").write_text(
        "config_id,u_eff_ev,element,species_labels,folder\n"
        "U_6.0,6.0,Cu,Cu,U_6.0\n",
        encoding="utf-8",
    )
    (job / "input.fdf").write_text(
        "%block DM.InitSpin\n1 0.7\n2 -0.7\n%endblock DM.InitSpin\n",
        encoding="utf-8",
    )
    (job / "siesta.out").write_text(
        "siesta: E_KS(eV) = -12.0\nSCF cycle converged\n"
        "Hirshfeld Atomic Populations:\n"
        "Atom # dQatom Atom pop Sz [e] Species\n"
        "1 0.0 0.0 0.6 Cu\n2 0.0 0.0 -0.6 Cu\n",
        encoding="utf-8",
    )
    (job / "cuo.EIG").write_text(
        "0.0\n2 1 2\n1 -1.0 1.0\n2 -1.2 1.2\n",
        encoding="utf-8",
    )
    row = collect_results(tmp_path)[0]
    assert row["config_id"] == "U_6.0"
    assert row["u_eff_ev"] == "6.0"
    assert row["element"] == "Cu"
    assert row["band_gap_ev"] == 2.0
    assert row["band_gap_type"] == "direct"
    assert row["band_gap_source"] == "cuo.EIG"
    assert row["band_gap_status"] == "ok"
    assert row["local_spins"] == "1:0.6 2:-0.6"
    assert row["local_spin_mean_abs"] == pytest.approx(0.6)
    assert row["local_spin_min_abs"] == pytest.approx(0.6)
    assert row["local_spin_max_abs"] == pytest.approx(0.6)
    assert row["exit_code"] == ""
    assert (tmp_path / "failed_array_indices.txt").read_text(encoding="utf-8") == ""
    assert (tmp_path / "failed_folders.list").read_text(encoding="utf-8") == ""


def test_collect_results_records_failed_u_array_indices(tmp_path: Path) -> None:
    job = tmp_path / "U_4.0"
    job.mkdir()
    (tmp_path / "folders.list").write_text("U_4.0\n", encoding="utf-8")
    (tmp_path / "u_sweep_manifest.csv").write_text(
        "config_id,u_eff_ev,element,species_labels,folder\n"
        "U_4.0,4.0,Cu,Cu,U_4.0\n",
        encoding="utf-8",
    )
    (job / "exit_code").write_text("7\n", encoding="utf-8")
    row = collect_results(tmp_path)[0]
    assert row["exit_code"] == 7
    assert row["status"] == "run-failed"
    assert (tmp_path / "failed_array_indices.txt").read_text(
        encoding="utf-8"
    ) == "1\n"
    assert (tmp_path / "failed_folders.list").read_text(
        encoding="utf-8"
    ) == "U_4.0\n"


def test_collect_results_keeps_eig_parse_diagnostic(tmp_path: Path) -> None:
    job = tmp_path / "001"
    job.mkdir()
    (job / "siesta.out").write_text("SCF cycle converged\n", encoding="utf-8")
    (job / "broken.EIG").write_text("not an EIG\n", encoding="utf-8")
    row = collect_results(tmp_path)[0]
    assert row["band_gap_source"] == "broken.EIG"
    assert row["band_gap_status"] == "parse-error"
    assert "truncated" in str(row["band_gap_error"])
