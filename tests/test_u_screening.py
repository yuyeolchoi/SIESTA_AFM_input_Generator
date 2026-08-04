import csv
import hashlib
from pathlib import Path

import pytest

from siesta_afm.io import parse_dm_init_spin
from siesta_afm.u_screening import (
    patch_hubbard_u,
    prepare_u_sweep,
    render_scheduler_script,
)


def _base_input(*, j_value: str = "0.00", two_cu_species: bool = False) -> str:
    species = (
        " 1 29 CuA\n 2 29 CuB\n 3 8 O"
        if two_cu_species
        else " 1 29 Cu\n 2 8 O"
    )
    second_cu_species = 2 if two_cu_species else 1
    oxygen_species = 3 if two_cu_species else 2
    projectors = (
        f"CuA 1\n n=3 2\n 6.00 {j_value}\n 0.00 0.00\n"
        f"CuB 1\n n=3 2\n 6.00 {j_value}\n 0.00 0.00"
        if two_cu_species
        else f"Cu 1\n n=3 2\n 6.00 {j_value}\n 0.00 0.00"
    )
    return f"""SystemName CuO U screening
SystemLabel cuo
NumberOfAtoms 4
NumberOfSpecies {3 if two_cu_species else 2}
LatticeConstant 1.0 Ang
MeshCutoff 6.00 Ry
%block ChemicalSpeciesLabel
{species}
%endblock ChemicalSpeciesLabel
%block LatticeVectors
4 0 0
0 4 0
0 0 4
%endblock LatticeVectors
AtomicCoordinatesFormat Fractional
%block AtomicCoordinatesAndAtomicSpecies
0 0 0 1
0.5 0.5 0.5 {oxygen_species}
0.5 0 0.5 {second_cu_species}
0 0.5 0 {oxygen_species}
%endblock AtomicCoordinatesAndAtomicSpecies
%block LDAU.proj
{projectors}
%endblock LDAU.proj
Spin polarized
%block DM.InitSpin
1 0.7
3 -0.7
%endblock DM.InitSpin
"""


def test_patch_hubbard_u_changes_only_selected_d_shell_tokens() -> None:
    original = _base_input(two_cu_species=True)
    patched, labels = patch_hubbard_u(original, "Cu", 4.5)
    assert labels == ("CuA", "CuB")
    assert patched.count("4.50000000 0.00") == 2
    assert "MeshCutoff 6.00 Ry" in patched
    assert parse_dm_init_spin(patched) == [(1, 0.7), (3, -0.7)]

    only_a, labels = patch_hubbard_u(
        original, "Cu", 2.0, species_labels=["CuA"]
    )
    assert labels == ("CuA",)
    assert only_a.count("2.00000000 0.00") == 1
    assert only_a.count("6.00 0.00") == 1


def test_patch_hubbard_u_preserves_block_layout_and_accepts_documented_comments(
) -> None:
    original = _base_input().replace(
        "%block ChemicalSpeciesLabel",
        "%block ChemicalSpeciesLabel  # atom labels",
    ).replace(
        "%block LDAU.proj\nCu 1\n n=3 2",
        "%block LDAU.proj  # correlated shells\nCu 1\n 2 E 50.0 2.5",
    ).replace(
        "%endblock LDAU.proj",
        "%endblock LDAU.proj  # end correlated shells",
    )
    patched, labels = patch_hubbard_u(original, "Cu", 3.5)
    assert labels == ("Cu",)
    assert "%block LDAU.proj  # correlated shells\nCu 1\n" in patched
    assert "%endblock LDAU.proj  # end correlated shells" in patched
    assert patched == original.replace("6.00 0.00", "3.50000000 0.00", 1)


def test_patch_hubbard_u_rejects_nonzero_j_and_wrong_species() -> None:
    with pytest.raises(ValueError, match="requires J=0"):
        patch_hubbard_u(_base_input(j_value="0.5"), "Cu", 4.0)
    with pytest.raises(ValueError, match="not a Cu species"):
        patch_hubbard_u(_base_input(), "Cu", 4.0, species_labels=["O"])


def test_prepare_u_sweep_writes_manifest_inputs_and_nurion_array(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base.fdf"
    base.write_text(_base_input(), encoding="utf-8")
    reference = tmp_path / "reference_spin.fdf"
    reference.write_text(
        "%block DM.InitSpin\n1 0.71\n3 -0.49\n%endblock DM.InitSpin\n",
        encoding="utf-8",
    )
    pseudopotential = tmp_path / "Cu.psml"
    pseudopotential.write_bytes(b"validated Cu pseudopotential\n")
    output = tmp_path / "sweep"
    result = prepare_u_sweep(
        base,
        reference,
        output,
        element="Cu",
        u_values=[0, 6],
        support_files=[pseudopotential],
        scheduler="nurion-pbs",
        project="project123",
        queue="normal",
        tasks_per_node=68,
        max_concurrent=2,
        module="siesta/5.4",
        run_command="mpirun siesta",
    )
    assert [folder.name for folder in result.folders] == ["U_0.0", "U_6.0"]
    assert parse_dm_init_spin(result.folders[0] / "input.fdf") == [
        (1, 0.71),
        (3, -0.49),
    ]
    assert "0.00000000 0.00" in (result.folders[0] / "input.fdf").read_text(
        encoding="utf-8"
    )
    with result.manifest_path.open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    assert [row["u_eff_ev"] for row in manifest] == ["0.0", "6.0"]
    assert all(len(row["input_sha256"]) == 64 for row in manifest)
    assert all(
        row["input_sha256"]
        == hashlib.sha256(
            (output / row["input_file"]).read_bytes()
        ).hexdigest()
        for row in manifest
    )
    assert all(row["support_files"] == '["Cu.psml"]' for row in manifest)
    assert all('"Cu.psml":' in row["support_files_sha256"] for row in manifest)
    assert (result.folders[0] / "Cu.psml").read_bytes() == (
        pseudopotential.read_bytes()
    )
    assert all(row["git_commit"] for row in manifest)
    assert all(row["package_version"] == "0.3.3" for row in manifest)
    assert all(row["git_dirty"] in {"true", "false", "unknown"} for row in manifest)
    assert result.scheduler_path is not None
    script = result.scheduler_path.read_text(encoding="utf-8")
    assert "#PBS -A siesta" in script
    assert "#PBS -P project123" in script
    assert "#PBS -J 1-2%2" in script
    assert "module load siesta/5.4" in script
    assert "mpirun siesta < RUN.fdf > siesta.out" in script

    with pytest.raises(ValueError, match="is not empty"):
        prepare_u_sweep(
            base,
            reference,
            output,
            element="Cu",
            u_values=[4],
        )


def test_prepare_u_sweep_rejects_incomplete_or_non_afm_reference(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base.fdf"
    base.write_text(_base_input(), encoding="utf-8")
    reference = tmp_path / "reference.fdf"
    reference.write_text(
        "%block DM.InitSpin\n1 0.7\n%endblock DM.InitSpin\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="omits Cu atom 3"):
        prepare_u_sweep(
            base,
            reference,
            tmp_path / "missing",
            element="Cu",
            u_values=[6],
        )

    reference.write_text(
        "%block DM.InitSpin\n1 0.7\n3 0.7\n%endblock DM.InitSpin\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="is not antiferromagnetic"):
        prepare_u_sweep(
            base,
            reference,
            tmp_path / "ferromagnetic",
            element="Cu",
            u_values=[6],
        )

    reference.write_text(
        "%block DM.InitSpin\n1 0.7 0 0\n3 -0.7 0 0\n"
        "%endblock DM.InitSpin\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="requires a collinear reference"):
        prepare_u_sweep(
            base,
            reference,
            tmp_path / "noncollinear",
            element="Cu",
            u_values=[6],
        )


def test_prepare_u_sweep_validates_scheduler_before_writing(tmp_path: Path) -> None:
    base = tmp_path / "base.fdf"
    base.write_text(_base_input(), encoding="utf-8")
    reference = tmp_path / "reference.fdf"
    reference.write_text(
        "%block DM.InitSpin\n1 0.7\n3 -0.7\n%endblock DM.InitSpin\n",
        encoding="utf-8",
    )
    output = tmp_path / "not_created"
    with pytest.raises(ValueError, match="single line"):
        prepare_u_sweep(
            base,
            reference,
            output,
            element="Cu",
            u_values=[4, 6],
            scheduler="nurion-pbs",
            pbs_application="siesta\ninvalid",
        )
    assert not output.exists()


def test_prepare_u_sweep_rejects_ambiguous_support_file_names(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base.fdf"
    base.write_text(_base_input(), encoding="utf-8")
    reference = tmp_path / "reference.fdf"
    reference.write_text(
        "%block DM.InitSpin\n1 0.7\n3 -0.7\n%endblock DM.InitSpin\n",
        encoding="utf-8",
    )
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    (left / "Cu.psml").write_text("left", encoding="utf-8")
    (right / "Cu.psml").write_text("right", encoding="utf-8")
    output = tmp_path / "not_created"
    with pytest.raises(ValueError, match="duplicate U-sweep support-file name"):
        prepare_u_sweep(
            base,
            reference,
            output,
            element="Cu",
            u_values=[6],
            support_files=[left / "Cu.psml", right / "Cu.psml"],
        )
    assert not output.exists()


def test_prepare_u_sweep_rejects_u_values_with_same_folder_name(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base.fdf"
    base.write_text(_base_input(), encoding="utf-8")
    reference = tmp_path / "reference.fdf"
    reference.write_text(
        "%block DM.InitSpin\n1 0.7\n3 -0.7\n%endblock DM.InitSpin\n",
        encoding="utf-8",
    )
    output = tmp_path / "not_created"
    with pytest.raises(ValueError, match="duplicate output folder names"):
        prepare_u_sweep(
            base,
            reference,
            output,
            element="Cu",
            u_values=[1.000000001, 1.000000002],
        )
    assert not output.exists()


def test_render_neuron_scheduler_uses_slurm_array_and_siesta_comment() -> None:
    script = render_scheduler_script(
        "neuron-slurm",
        5,
        partition="skl",
        tasks_per_node=20,
        max_concurrent=3,
        run_command="srun siesta",
    )
    assert "#SBATCH --partition=skl" in script
    assert "#SBATCH --array=1-5%3" in script
    assert "#SBATCH --comment=siesta" in script
    assert "SLURM_ARRAY_TASK_ID" in script
    assert "srun siesta < RUN.fdf > siesta.out" in script

    with pytest.raises(ValueError, match="minutes and seconds"):
        render_scheduler_script("neuron-slurm", 1, walltime="01:60:00")
