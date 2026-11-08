import csv
import subprocess
from pathlib import Path

import pytest
from ase.units import Hartree

from siesta_afm.cli import main
from siesta_afm.cp2k_jobs import prepare_cp2k_array, render_cp2k_slurm_script
from siesta_afm.cp2k_results import (
    last_cp2k_run,
    parse_cp2k_initial_spins,
    parse_cp2k_mulliken_spins,
    parse_cp2k_output,
)
from siesta_afm.io import read_structure
from siesta_afm.results import collect_results


ROOT = Path(__file__).parents[1]
FIXTURES = Path(__file__).parent / "fixtures"
CUO_SLAB = ROOT / "examples" / "CuO_111_slab.cif"
OUTPUT = (FIXTURES / "cp2k_uks_appended_runs.out").read_text(encoding="utf-8")


# --- output parsing -----------------------------------------------------------------


def test_last_cp2k_run_selects_the_final_appended_run() -> None:
    run, count = last_cp2k_run(OUTPUT)

    assert count == 2
    assert "PROGRAM STARTED AT               2026-11-05 10:00:00.000" in run
    assert "[ABORT]" not in run
    assert last_cp2k_run("no banner here") == ("no banner here", 1)
    assert last_cp2k_run("") == ("", 0)


def test_parse_cp2k_mulliken_spins_reads_last_uks_table() -> None:
    spins, total = parse_cp2k_mulliken_spins(last_cp2k_run(OUTPUT)[0])

    assert spins == {1: 0.275113, 2: -0.275113, 3: 0.0}
    assert total == 0.0
    restricted = (
        "Mulliken Population Analysis\n\n"
        " #  Atom  Element  Kind  Atomic population   Net charge\n"
        "       1     Cu       1          11.000000      0.000000\n"
        " # Total charge                 11.000000      0.000000\n"
    )
    assert parse_cp2k_mulliken_spins(restricted) == ({}, None)


def test_parse_cp2k_output_summary() -> None:
    summary = parse_cp2k_output(OUTPUT)

    assert summary.run_count == 2
    assert summary.ended_normally and not summary.aborted
    assert summary.energy_hartree == pytest.approx(-1234.567890123)
    assert summary.energy_ev == pytest.approx(-1234.567890123 * Hartree)
    assert summary.scf_converged and not summary.geometry_converged
    assert summary.local_spins[1] == 0.275113
    assert summary.spin_source == "cp2k-mulliken-population-analysis"
    assert summary.plus_u_negative_warnings == 2
    assert any("appended runs" in note for note in summary.notes)
    assert any("DFT+U energy contribution is negative" in note for note in summary.notes)


def test_parse_cp2k_output_flags_abort_and_unconverged_first_run() -> None:
    first_run = OUTPUT.split("PROGRAM STARTED AT", 2)
    text = "PROGRAM STARTED AT" + first_run[1]

    summary = parse_cp2k_output(text)

    assert summary.aborted and not summary.ended_normally
    assert not summary.scf_converged
    assert summary.energy_hartree == pytest.approx(-1234.1)
    assert summary.local_spins == {1: 0.8, 2: -0.8, 3: 0.0}


def test_parse_cp2k_initial_spins_from_kinds(tmp_path: Path) -> None:
    text = (
        "&SUBSYS\n  &COORD\n"
        "    Cu_up 0 0 0  # 1\n    O 1 0 0\n    Cu_dn 2 0 0\n"
        "  &END COORD\n"
        "  &KIND Cu_up\n    ELEMENT Cu\n    MAGNETIZATION 1.0\n  &END KIND\n"
        "  &KIND O\n    ELEMENT O\n  &END KIND\n"
        "  &KIND Cu_dn\n    ELEMENT Cu\n    MAGNETIZATION -1  # dn\n  &END KIND\n"
        "&END SUBSYS\n"
    )
    path = tmp_path / "input.inp"
    path.write_text(text, encoding="utf-8")

    assert parse_cp2k_initial_spins(text) == {1: 1.0, 2: 0.0, 3: -1.0}
    assert parse_cp2k_initial_spins(path) == {1: 1.0, 2: 0.0, 3: -1.0}
    with pytest.raises(ValueError, match="without a matching &KIND"):
        parse_cp2k_initial_spins("&COORD\n Zn 0 0 0\n&END COORD\n")


# --- collect_results on CP2K folders --------------------------------------------------


def _write_cp2k_job(folder: Path, *, exit_code: str | None = "0") -> None:
    folder.mkdir(parents=True)
    (folder / "input.inp").write_text(
        "&COORD\n Cu_up 0 0 0\n Cu_dn 1 0 0\n O 2 0 0\n&END COORD\n"
        "&KIND Cu_up\n MAGNETIZATION 1.0\n&END KIND\n"
        "&KIND Cu_dn\n MAGNETIZATION -1.0\n&END KIND\n"
        "&KIND O\n&END KIND\n",
        encoding="utf-8",
    )
    (folder / "cp2k_12345.out").write_text(OUTPUT, encoding="utf-8")
    if exit_code is not None:
        (folder / "exit_code").write_text(exit_code + "\n", encoding="utf-8")


def test_collect_results_reads_cp2k_job_folders(tmp_path: Path) -> None:
    _write_cp2k_job(tmp_path / "001_layer")
    (tmp_path / "folders.list").write_text("001_layer\n", encoding="utf-8")

    rows = collect_results(tmp_path)

    row = rows[0]
    assert row["code"] == "cp2k"
    assert row["total_energy"] == pytest.approx(-1234.567890123 * Hartree)
    assert row["final_net_spin"] == 0.0
    assert row["sign_retention"] == 1.0
    assert row["collapsed_atoms"] == 0  # 0.275 is below the 0.5 initial gate but above 0.1
    assert row["spin_population_source"] == "cp2k-mulliken-population-analysis"
    assert row["band_gap_status"] == "not-available-cp2k"
    assert row["scf_converged"] is True
    assert row["status"] == "ok"
    assert "appended runs" in row["notes"] and "DFT+U" in row["notes"]
    with (tmp_path / "results.csv").open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    assert header[-2:] == ["code", "notes"]


def test_collect_results_cp2k_abort_status(tmp_path: Path) -> None:
    folder = tmp_path / "002_layer"
    _write_cp2k_job(folder, exit_code=None)
    first_run = "PROGRAM STARTED AT" + OUTPUT.split("PROGRAM STARTED AT", 2)[1]
    (folder / "cp2k_12345.out").write_text(first_run, encoding="utf-8")

    row = collect_results(tmp_path)[0]

    assert row["status"] == "aborted"
    assert row["scf_converged"] is False
    assert "ABORT" in row["notes"]


def test_collect_results_mixed_codes_in_one_table(tmp_path: Path) -> None:
    _write_cp2k_job(tmp_path / "001_layer")
    siesta = tmp_path / "002_layer"
    siesta.mkdir()
    (siesta / "input.fdf").write_text(
        "%block DM.InitSpin\n1 4.0\n2 -4.0\n%endblock DM.InitSpin\n", encoding="utf-8"
    )
    (siesta / "RUN.out").write_text(
        (FIXTURES / "siesta_5_atomic_populations.out").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    rows = collect_results(tmp_path)

    assert [row["code"] for row in rows] == ["cp2k", "siesta"]
    assert rows[1]["total_energy"] == -876.5
    assert rows[1]["notes"] == ""


# --- job preparation --------------------------------------------------------------------


def test_render_cp2k_slurm_script_contains_preflight_and_is_valid_bash(tmp_path: Path) -> None:
    script = render_cp2k_slurm_script(
        3,
        partition="gpuq",
        ntasks=8,
        memory="40G",
        max_concurrent=1,
        cp2k_env="/scratch/9_CP2K/cp2k-2026.2/install/cp2k_env",
    )

    assert "#SBATCH --array=1-3%1" in script
    assert "#SBATCH --partition=gpuq" in script
    assert "#SBATCH --mem=40G" in script
    assert "export OMP_NUM_THREADS=1" in script
    assert "grep -v 'hpc_sdk'" in script
    assert 'source "/scratch/9_CP2K/cp2k-2026.2/install/cp2k_env"' in script
    assert "*/install/bin/*) ;;" in script
    assert 'grep -q "not found"' in script and 'libmpi.so.40' in script
    assert 'OUT="cp2k_${SLURM_JOB_ID}.out"' in script
    assert '--bind-to core --map-by core "$CP2K_BIN" -i input.inp -o "$OUT"' in script
    assert "> exit_code" in script
    path = tmp_path / "run.sbatch"
    path.write_text(script, encoding="utf-8")
    subprocess.run(["bash", "-n", str(path)], check=True)


def test_render_cp2k_slurm_script_module_variant_and_validation() -> None:
    script = render_cp2k_slurm_script(1, module="cp2k/2026.2", cp2k_env=None, strip_nvhpc=False)

    assert "module load cp2k/2026.2" in script
    assert "cp2k_env" not in script
    assert "strip_nvhpc()" not in script  # the ldd preflight still rejects NVHPC libraries
    assert "*/install/bin/*" not in script
    with pytest.raises(ValueError, match="--cp2k-env .* or --module"):
        render_cp2k_slurm_script(1, cp2k_env=None, module=None)
    with pytest.raises(ValueError, match="walltime"):
        render_cp2k_slurm_script(1, cp2k_env="env", walltime="4h")
    with pytest.raises(ValueError, match="plain token"):
        render_cp2k_slurm_script(1, cp2k_env="env", job_name="bad name; rm -rf")
    with pytest.raises(ValueError, match="plain path"):
        render_cp2k_slurm_script(1, cp2k_env="env", mpirun="mpirun $(evil)")


def test_prepare_cp2k_array_writes_inputs_manifest_and_script(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "afm_001.fdf").write_text(
        "%block DM.InitSpin\n1 1.0\n2 1.0\n5 -1.0\n6 -1.0\n9 1.0\n10 1.0\n%endblock DM.InitSpin\n",
        encoding="utf-8",
    )
    (configs / "afm_002.fdf").write_text(
        "%block DM.InitSpin\n1 1.0\n2 -1.0\n5 1.0\n6 -1.0\n9 1.0\n10 -1.0\n%endblock DM.InitSpin\n",
        encoding="utf-8",
    )
    (configs / "manifest.csv").write_text(
        "config_id,method,n_up,n_down,net_spin,afm_score,file\n"
        "001,layer,4,2,2.0,1.0,afm_001.fdf\n002,checkerboard,3,3,0.0,1.0,afm_002.fdf\n",
        encoding="utf-8",
    )
    structure = read_structure(CUO_SLAB, slab=True)

    folders, warnings = prepare_cp2k_array(
        structure,
        configs,
        tmp_path / "jobs",
        magnetic_species=["Cu"],
        writer_options={"hubbard_u": ["Cu=6.0"], "charge": 0},
        job_options={"cp2k_env": "/opt/cp2k_env", "partition": None},
    )

    assert [folder.name for folder in folders] == ["001_layer", "002_checkerboard"]
    assert all((folder / "input.inp").is_file() for folder in folders)
    assert (tmp_path / "jobs" / "folders.list").read_text(encoding="utf-8") == (
        "001_layer\n002_checkerboard\n"
    )
    with (tmp_path / "jobs" / "cp2k_manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["multiplicity"] for row in rows] == ["3", "1"]
    assert rows[0]["kinds"] == "Cu_up=1 O=0 Cu_dn=-1"
    assert "PROJECT afm_001" in (folders[0] / "input.inp").read_text(encoding="utf-8")
    script = (tmp_path / "jobs" / "run_cp2k_array.sbatch").read_text(encoding="utf-8")
    assert "#SBATCH --array=1-2" in script and "--partition" not in script
    assert any("do not transfer between codes" in warning for warning in warnings)
    assert parse_cp2k_initial_spins(folders[1] / "input.inp") == {
        1: 1.0, 2: -1.0, 3: 0.0, 4: 0.0, 5: 1.0, 6: -1.0, 7: 0.0, 8: 0.0, 9: 1.0, 10: -1.0, 11: 0.0, 12: 0.0
    }


def test_prepare_cp2k_array_rejects_spins_outside_magnetic_species(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "afm_001.fdf").write_text(
        "%block DM.InitSpin\n3 1.0\n%endblock DM.InitSpin\n", encoding="utf-8"
    )
    (configs / "manifest.csv").write_text(
        "config_id,method,n_up,n_down,net_spin,afm_score,file\n001,layer,1,0,1.0,0,afm_001.fdf\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="atom 3 \\(O\\) outside --magnetic-species"):
        prepare_cp2k_array(
            read_structure(CUO_SLAB, slab=True),
            configs,
            tmp_path / "jobs",
            magnetic_species=["Cu"],
            job_options={"cp2k_env": "/opt/cp2k_env"},
        )


def test_prepare_cp2k_array_cli_then_collect(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    configs = tmp_path / "cands"
    assert (
        main(
            [
                "enumerate",
                str(CUO_SLAB),
                "--slab",
                "--magnetic-species",
                "Cu",
                "--methods",
                "layer,checkerboard",
                "--moment",
                "1.0",
                "--n-configs",
                "2",
                "--output-dir",
                str(configs),
            ]
        )
        == 0
    )
    jobs = tmp_path / "cp2k_jobs"

    assert (
        main(
            [
                "prepare-cp2k-array",
                str(CUO_SLAB),
                "--slab",
                "--magnetic-species",
                "Cu",
                "--configs",
                str(configs),
                "--output-dir",
                str(jobs),
                "--ntasks",
                "4",
                "--walltime",
                "01:00:00",
            ]
        )
        == 0
    )

    out = capsys.readouterr().out
    assert "Prepared" in out and "run_cp2k_array.sbatch" in out
    folders = [line for line in (jobs / "folders.list").read_text(encoding="utf-8").splitlines() if line]
    assert folders
    script = (jobs / "run_cp2k_array.sbatch").read_text(encoding="utf-8")
    assert "#SBATCH --ntasks=4" in script and "#SBATCH --time=01:00:00" in script
    # Simulate one finished job whose Mulliken table covers every atom, then collect.
    initial = parse_cp2k_initial_spins(jobs / folders[0] / "input.inp")
    (jobs / folders[0] / "cp2k_1.out").write_text(
        _synthetic_cp2k_output(initial, energy_hartree=-100.5), encoding="utf-8"
    )
    (jobs / folders[0] / "exit_code").write_text("0\n", encoding="utf-8")
    assert main(["collect-results", str(jobs)]) == 0
    with (jobs / "results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["code"] == "cp2k" and rows[0]["status"] == "ok"
    assert float(rows[0]["total_energy"]) == pytest.approx(-100.5 * Hartree)
    assert rows[0]["sign_retention"] == "1.0"
    assert all(row["code"] == "cp2k" for row in rows)
    if len(rows) > 1:
        assert rows[1]["status"] == "missing-output"


def _synthetic_cp2k_output(initial: dict[int, float], *, energy_hartree: float) -> str:
    """A one-run UKS output whose Mulliken spins keep the initial signs at 0.3."""

    symbols = {index: ("Cu" if value != 0 else "O") for index, value in initial.items()}
    rows = [
        f"{index:8d}     {symbols[index]:<2}       1          5.5  5.5  0.0  {0.3 * (1 if value > 0 else -1 if value < 0 else 0):.6f}"
        for index, value in sorted(initial.items())
    ]
    total = sum(0.3 * (1 if v > 0 else -1 if v < 0 else 0) for v in initial.values())
    return "\n".join(
        [
            "  **** **** ******  **  PROGRAM STARTED AT               2026-11-05 10:00:00.000",
            "  *** SCF run converged in     5 steps ***",
            "                     Mulliken Population Analysis",
            "",
            " #  Atom  Element  Kind  Atomic population (alpha,beta) Net charge  Spin moment",
            *rows,
            f" # Total charge and spin                 1.0     1.0       0.0     {total:.6f}",
            f" ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]           {energy_hartree:.12f}",
            "  **** **** ******  **  PROGRAM ENDED AT                 2026-11-05 10:20:00.000",
            "",
        ]
    )
