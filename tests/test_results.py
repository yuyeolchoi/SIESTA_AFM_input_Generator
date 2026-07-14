from pathlib import Path

from siesta_afm.results import (
    _find_output,
    collect_results,
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
