"""Prepare reproducible Hubbard-U sweeps from a validated SIESTA input."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Sequence

from ase.data import atomic_numbers

from . import __version__
from .fdf_writer import patch_fdf_text
from .io import parse_dm_init_spin, read_fdf_document, read_structure


_FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][-+]?\d+)?"
_UJ_LINE_RE = re.compile(
    rf"^(?P<indent>\s*)(?P<u>{_FLOAT_RE})(?P<separator>\s+)"
    rf"(?P<j>{_FLOAT_RE})(?P<tail>.*)$"
)


@dataclass(frozen=True, slots=True)
class USweepPreparation:
    """Paths and rows created by :func:`prepare_u_sweep`."""

    folders: tuple[Path, ...]
    manifest: tuple[dict[str, str], ...]
    manifest_path: Path
    scheduler_path: Path | None


def _fdf_block(text: str, name: str) -> tuple[re.Match[str], list[str]]:
    normalized = re.sub(r"[._-]", "", name)
    label = r"[._-]*".join(re.escape(character) for character in normalized)
    pattern = re.compile(
        rf"(?ims)^(?P<start>[ \t]*%block[ \t]+{label}"
        rf"[ \t]*(?:[#!][^\r\n]*)?\r?$)"
        rf"(?P<body>.*?)"
        rf"^(?P<end>[ \t]*%endblock(?:[ \t]+{label})?"
        rf"[ \t]*(?:[#!][^\r\n]*)?\r?$)"
    )
    matches = list(pattern.finditer(text))
    if not matches:
        raise ValueError(f"FDF block {name} was not found in the top-level input")
    if len(matches) > 1:
        raise ValueError(f"FDF block {name} occurs more than once")
    match = matches[0]
    return match, match.group("body").splitlines()


def _code_fields(line: str) -> list[str]:
    return re.split(r"[#!]", line, maxsplit=1)[0].split()


def _chemical_species_labels(text: str, element: str) -> list[str]:
    _, lines = _fdf_block(text, "ChemicalSpeciesLabel")
    expected_number = atomic_numbers[element]
    labels: list[str] = []
    for line in lines:
        fields = _code_fields(line)
        if len(fields) < 3:
            continue
        try:
            number = abs(int(fields[1]))
        except ValueError:
            continue
        if number == expected_number and fields[2] not in labels:
            labels.append(fields[2])
    if not labels:
        raise ValueError(
            f"ChemicalSpeciesLabel contains no species for element {element}"
        )
    return labels


def _projector_block(text: str) -> tuple[re.Match[str], list[str], str]:
    found: list[tuple[re.Match[str], list[str], str]] = []
    for name in ("DFTU.Proj", "LDAU.Proj"):
        try:
            match, lines = _fdf_block(text, name)
        except ValueError as exc:
            if "was not found" not in str(exc):
                raise
            continue
        found.append((match, lines, name))
    if not found:
        raise ValueError(
            "no DFTU.Proj or LDAU.Proj block was found in the top-level input"
        )
    if len(found) > 1:
        raise ValueError("input contains both DFTU.Proj and LDAU.Proj blocks")
    return found[0]


def _species_header(fields: Sequence[str]) -> str | None:
    if (
        len(fields) < 2
        or re.fullmatch(_FLOAT_RE, fields[0])
        or re.fullmatch(r"(?i)n=\d+", fields[0])
    ):
        return None
    try:
        shells = int(fields[1])
    except ValueError:
        return None
    return fields[0] if shells > 0 else None


def _projector_l(fields: Sequence[str]) -> int | None:
    if not fields:
        return None
    if re.fullmatch(r"(?i)n=\d+", fields[0]) and len(fields) >= 2:
        try:
            return int(fields[1])
        except ValueError:
            return None
    if len(fields) == 1 or (len(fields) > 1 and fields[1].upper() == "E"):
        try:
            return int(fields[0])
        except ValueError:
            return None
    return None


def _replace_u_line(line: str, u_eff: float, *, require_j_zero: bool) -> str:
    match = _UJ_LINE_RE.match(line)
    if not match:
        raise ValueError(f"invalid DFT+U U/J line: {line.strip()!r}")
    j_value = float(match.group("j").replace("D", "E").replace("d", "e"))
    if require_j_zero and abs(j_value) > 1e-12:
        raise ValueError(
            "U_eff sweep requires J=0, but the selected projector has "
            f"J={j_value:g} eV"
        )
    return (
        f"{match.group('indent')}{u_eff:.8f}{match.group('separator')}"
        f"{match.group('j')}{match.group('tail')}"
    )


def patch_hubbard_u(
    input_text: str,
    element: str,
    u_eff: float,
    *,
    species_labels: Sequence[str] | None = None,
    require_j_zero: bool = True,
) -> tuple[str, tuple[str, ...]]:
    """Patch the d-shell U value for every selected species of one element.

    The function changes only the U token that follows an explicit d-shell
    projector descriptor.  Ambiguous or unsupported blocks fail closed rather
    than risking a broad textual substitution.
    """

    canonical = element.strip().capitalize()
    if canonical not in atomic_numbers:
        raise ValueError(f"unknown element for Hubbard U sweep: {element}")
    value = float(u_eff)
    if not math.isfinite(value) or value < 0:
        raise ValueError("Hubbard U values must be finite and nonnegative")

    element_labels = _chemical_species_labels(input_text, canonical)
    selected = list(species_labels) if species_labels else element_labels
    if not selected:
        raise ValueError("at least one Hubbard projector species label is required")
    by_lower = {label.lower(): label for label in element_labels}
    unknown = [label for label in selected if label.lower() not in by_lower]
    if unknown:
        raise ValueError(
            f"species label {unknown[0]!r} is not a {canonical} species in "
            "ChemicalSpeciesLabel"
        )
    selected = [by_lower[label.lower()] for label in selected]
    if len({label.lower() for label in selected}) != len(selected):
        raise ValueError("duplicate Hubbard projector species label")

    block_match, lines, block_name = _projector_block(input_text)
    headers: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        label = _species_header(_code_fields(line))
        if label is not None:
            headers.append((index, label))

    patched = list(lines)
    patched_labels: list[str] = []
    for wanted in selected:
        matching = [item for item in headers if item[1].lower() == wanted.lower()]
        if not matching:
            raise ValueError(f"{block_name} has no projector for species {wanted}")
        if len(matching) > 1:
            raise ValueError(
                f"{block_name} has duplicate projectors for species {wanted}"
            )
        header_index = matching[0][0]
        next_headers = [index for index, _ in headers if index > header_index]
        end_index = min(next_headers) if next_headers else len(lines)
        u_line_indices: list[int] = []
        for descriptor_index in range(header_index + 1, end_index):
            if _projector_l(_code_fields(lines[descriptor_index])) != 2:
                continue
            u_line_index = next(
                (
                    index
                    for index in range(descriptor_index + 1, end_index)
                    if _code_fields(lines[index])
                ),
                None,
            )
            if u_line_index is None:
                raise ValueError(
                    f"{block_name} d-shell projector for {wanted} has no U/J line"
                )
            u_line_indices.append(u_line_index)
        if len(u_line_indices) != 1:
            raise ValueError(
                f"{block_name} must contain exactly one explicit d-shell projector "
                f"for {wanted}; found {len(u_line_indices)}"
            )
        line_index = u_line_indices[0]
        patched[line_index] = _replace_u_line(
            lines[line_index], value, require_j_zero=require_j_zero
        )
        patched_labels.append(wanted)

    body_lines = block_match.group("body").splitlines(keepends=True)
    if len(body_lines) != len(patched):  # pragma: no cover - splitlines invariant
        raise RuntimeError("internal DFT+U block line-count mismatch")
    patched_body = "".join(
        new_line + old_line[len(old_line.rstrip("\r\n")) :]
        for old_line, new_line in zip(body_lines, patched, strict=True)
    )
    replacement = (
        block_match.group("start") + patched_body + block_match.group("end")
    )
    output = input_text[: block_match.start()] + replacement + input_text[block_match.end() :]
    return output, tuple(patched_labels)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _source_file_provenance(
    root: Path, source_files: Sequence[Path]
) -> tuple[str, str]:
    """Serialize portable dependency names and byte hashes for an FDF root."""

    names: list[str] = []
    hashes: dict[str, str] = {}
    parent = root.resolve().parent
    for source in source_files:
        resolved = source.resolve()
        try:
            name = resolved.relative_to(parent).as_posix()
        except ValueError:
            name = resolved.as_posix()
        names.append(name)
        hashes[name] = _sha256_file(resolved)
    return (
        json.dumps(names, ensure_ascii=False),
        json.dumps(hashes, ensure_ascii=False, sort_keys=True),
    )


def _git_provenance() -> tuple[str, str]:
    repository = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown", "unknown"
    dirty = str(bool(status.stdout.strip())).lower()
    return revision.stdout.strip() or "unknown", dirty


def _validate_reference_spin(
    base_input: Path,
    reference_spin: Path,
    element: str,
) -> None:
    structure = read_structure(base_input)
    spin_warnings: list[str] = []
    rows = parse_dm_init_spin(reference_spin, warnings=spin_warnings)
    if spin_warnings:
        raise ValueError(
            "U_eff screening requires a collinear reference DM.InitSpin; "
            "theta/phi fields are not supported"
        )
    values: dict[int, float] = {}
    for one_based, spin in rows:
        if one_based in values:
            raise ValueError(f"reference spin has duplicate atom {one_based}")
        if not 1 <= one_based <= len(structure):
            raise ValueError(f"reference spin atom index out of range: {one_based}")
        values[one_based] = spin
    expected = {
        index + 1
        for index, symbol in enumerate(structure.symbols)
        if symbol.lower() == element.lower()
    }
    if not expected:
        raise ValueError(f"base input contains no atoms of element {element}")
    nonmagnetic = [
        index for index, spin in values.items() if index not in expected and spin != 0
    ]
    if nonmagnetic:
        raise ValueError(
            f"reference spin has nonzero value on non-{element} atom "
            f"{min(nonmagnetic)}"
        )
    missing = expected - values.keys()
    if missing:
        raise ValueError(f"reference spin omits {element} atom {min(missing)}")
    selected_values = [values[index] for index in expected]
    if not any(value > 0 for value in selected_values) or not any(
        value < 0 for value in selected_values
    ):
        raise ValueError(
            f"reference spin is not antiferromagnetic: {element} requires both "
            "positive and negative initial spins"
        )


def _format_u(value: float) -> str:
    text = f"{value:.8f}".rstrip("0").rstrip(".")
    return text + ".0" if "." not in text else text


def _validated_u_values(values: Sequence[float]) -> list[float]:
    if not values:
        raise ValueError("at least one Hubbard U value is required")
    result = [float(value) for value in values]
    if any(not math.isfinite(value) or value < 0 for value in result):
        raise ValueError("Hubbard U values must be finite and nonnegative")
    if len(set(result)) != len(result):
        raise ValueError("duplicate Hubbard U value")
    return result


def _validate_script_value(value: str, label: str) -> str:
    stripped = value.strip()
    if not stripped or "\n" in stripped or "\r" in stripped:
        raise ValueError(f"{label} must be a nonempty single line")
    return stripped


def _validate_scheduler_token(value: str, label: str) -> str:
    token = _validate_script_value(value, label)
    if not re.fullmatch(r"[A-Za-z0-9_.+/@:-]+", token):
        raise ValueError(f"{label} contains unsupported scheduler characters")
    return token


def render_scheduler_script(
    scheduler: str,
    job_count: int,
    *,
    job_name: str = "cuo_u_sweep",
    pbs_application: str = "siesta",
    project: str | None = None,
    queue: str | None = None,
    partition: str | None = None,
    walltime: str = "04:00:00",
    nodes: int = 1,
    tasks_per_node: int = 1,
    max_concurrent: int | None = None,
    module: str | None = None,
    run_command: str = "siesta",
) -> str:
    """Render a Nurion PBS or Neuron SLURM array script."""

    scheduler = scheduler.lower()
    if scheduler not in {"nurion-pbs", "neuron-slurm"}:
        raise ValueError("scheduler must be nurion-pbs or neuron-slurm")
    if job_count < 1 or nodes < 1 or tasks_per_node < 1:
        raise ValueError("job count, nodes, and tasks per node must be positive")
    if max_concurrent is not None and max_concurrent < 1:
        raise ValueError("maximum concurrent jobs must be positive")
    if not re.fullmatch(r"\d{1,3}:\d{2}:\d{2}", walltime):
        raise ValueError("walltime must use HH:MM:SS")
    _, minutes, seconds = (int(field) for field in walltime.split(":"))
    if minutes >= 60 or seconds >= 60:
        raise ValueError("walltime minutes and seconds must be below 60")
    name = _validate_scheduler_token(job_name, "job name")
    command = _validate_script_value(run_command, "run command")
    array = f"1-{job_count}" + (
        f"%{max_concurrent}" if max_concurrent is not None else ""
    )
    lines = ["#!/bin/bash"]
    if scheduler == "nurion-pbs":
        lines.extend(
            [
                f"#PBS -N {name}",
                "#PBS -A "
                + _validate_scheduler_token(
                    pbs_application, "PBS application accounting code"
                ),
                f"#PBS -l select={nodes}:ncpus={tasks_per_node}:mpiprocs={tasks_per_node}",
                f"#PBS -l walltime={walltime}",
                f"#PBS -J {array}",
                "#PBS -j oe",
            ]
        )
        if queue:
            lines.insert(3, f"#PBS -q {_validate_scheduler_token(queue, 'queue')}")
        if project:
            lines.insert(
                3,
                f"#PBS -P {_validate_scheduler_token(project, 'project')}",
            )
        index_variable = "PBS_ARRAY_INDEX"
    else:
        lines.extend(
            [
                f"#SBATCH --job-name={name}",
                f"#SBATCH --nodes={nodes}",
                f"#SBATCH --ntasks-per-node={tasks_per_node}",
                f"#SBATCH --time={walltime}",
                f"#SBATCH --array={array}",
                "#SBATCH --comment=siesta",
                "#SBATCH --output=%x_%A_%a.out",
                "#SBATCH --error=%x_%A_%a.err",
            ]
        )
        if partition:
            lines.insert(
                3,
                f"#SBATCH --partition={_validate_scheduler_token(partition, 'partition')}",
            )
        index_variable = "SLURM_ARRAY_TASK_ID"
    lines.extend(
        [
            "",
            "set -euo pipefail",
            'ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
            f'TASK_INDEX="${{{index_variable}}}"',
            'FOLDER="$(sed -n "${TASK_INDEX}p" "$ROOT/folders.list")"',
            'if [ -z "$FOLDER" ] || [ ! -d "$ROOT/$FOLDER" ]; then',
            '  echo "Invalid array index $TASK_INDEX: $FOLDER" >&2',
            "  exit 2",
            "fi",
            'cd "$ROOT/$FOLDER"',
        ]
    )
    if module:
        lines.extend(
            [
                "module purge",
                f"module load {_validate_scheduler_token(module, 'module')}",
            ]
        )
    lines.extend(
        [
            "set +e",
            f"{command} < RUN.fdf > siesta.out 2> siesta.err",
            "status=$?",
            "set -e",
            "printf '%s\\n' \"$status\" > exit_code",
            'exit "$status"',
            "",
        ]
    )
    return "\n".join(lines)


def prepare_u_sweep(
    base_input: str | Path,
    reference_spin: str | Path,
    output_dir: str | Path,
    *,
    element: str,
    u_values: Sequence[float],
    species_labels: Sequence[str] | None = None,
    support_files: Sequence[str | Path] = (),
    scheduler: str | None = None,
    job_name: str = "cuo_u_sweep",
    pbs_application: str = "siesta",
    project: str | None = None,
    queue: str | None = None,
    partition: str | None = None,
    walltime: str = "04:00:00",
    nodes: int = 1,
    tasks_per_node: int = 1,
    max_concurrent: int | None = None,
    module: str | None = None,
    run_command: str = "siesta",
) -> USweepPreparation:
    """Create one validated SIESTA folder per U value and an optional job array."""

    source = Path(base_input)
    spin_source = Path(reference_spin)
    destination = Path(output_dir)
    if not source.is_file():
        raise FileNotFoundError(source)
    if not spin_source.is_file():
        raise FileNotFoundError(spin_source)
    if destination.exists():
        if not destination.is_dir():
            raise ValueError(f"U-sweep output path is not a directory: {destination}")
        if any(destination.iterdir()):
            raise ValueError(f"U-sweep output directory is not empty: {destination}")

    prepared_support: list[tuple[str, bytes, str]] = []
    seen_support_names: set[str] = set()
    for item in support_files:
        support_path = Path(item)
        if not support_path.is_file():
            raise FileNotFoundError(support_path)
        name = support_path.name
        if name.lower() in {"input.fdf", "run.fdf"}:
            raise ValueError(f"reserved U-sweep support-file name: {name}")
        normalized_name = name.casefold()
        if normalized_name in seen_support_names:
            raise ValueError(f"duplicate U-sweep support-file name: {name}")
        seen_support_names.add(normalized_name)
        content = support_path.read_bytes()
        prepared_support.append((name, content, _sha256_bytes(content)))

    canonical = element.strip().capitalize()
    if canonical not in atomic_numbers:
        raise ValueError(f"unknown element for Hubbard U sweep: {element}")
    values = _validated_u_values(u_values)
    formatted_values = [_format_u(value) for value in values]
    if len(set(formatted_values)) != len(formatted_values):
        raise ValueError("U values resolve to duplicate output folder names")
    _validate_reference_spin(source, spin_source, canonical)
    base_document = read_fdf_document(source)
    base_text = base_document.text
    if spin_source.suffix.lower() == ".fdf":
        spin_document = read_fdf_document(spin_source)
        spin_text = spin_document.text
        spin_sources = spin_document.source_files
    else:
        spin_text = spin_source.read_text(encoding="utf-8-sig")
        spin_sources = (spin_source.resolve(),)
    spin_patched = patch_fdf_text(base_text, spin_text)

    prepared: list[tuple[str, str, tuple[str, ...], str]] = []
    for value in values:
        patched, labels = patch_hubbard_u(
            spin_patched,
            canonical,
            value,
            species_labels=species_labels,
        )
        u_text = _format_u(value)
        folder_name = f"U_{u_text}"
        prepared.append((folder_name, patched, labels, u_text))

    scheduler_script: str | None = None
    scheduler_suffix: str | None = None
    if scheduler:
        scheduler_script = render_scheduler_script(
            scheduler,
            len(prepared),
            job_name=job_name,
            pbs_application=pbs_application,
            project=project,
            queue=queue,
            partition=partition,
            walltime=walltime,
            nodes=nodes,
            tasks_per_node=tasks_per_node,
            max_concurrent=max_concurrent,
            module=module,
            run_command=run_command,
        )
        scheduler_suffix = "pbs" if scheduler.lower() == "nurion-pbs" else "slurm"

    base_hash = _sha256_file(source)
    spin_hash = _sha256_file(spin_source)
    base_source_files, base_source_hashes = _source_file_provenance(
        source, base_document.source_files
    )
    spin_source_files, spin_source_hashes = _source_file_provenance(
        spin_source, spin_sources
    )
    base_effective_hash = _sha256_bytes(base_text.encode("utf-8"))
    spin_effective_hash = _sha256_bytes(spin_text.encode("utf-8"))
    revision, git_dirty = _git_provenance()
    destination.mkdir(parents=True, exist_ok=True)
    folders: list[Path] = []
    manifest: list[dict[str, str]] = []
    for folder_name, patched, labels, u_text in prepared:
        folder = destination / folder_name
        folder.mkdir()
        input_path = folder / "input.fdf"
        input_path.write_text(patched, encoding="utf-8", newline="")
        (folder / "RUN.fdf").write_text(
            "%include input.fdf\n", encoding="utf-8", newline="\n"
        )
        for name, content, _ in prepared_support:
            (folder / name).write_bytes(content)
        folders.append(folder)
        manifest.append(
            {
                "config_id": folder_name,
                "u_eff_ev": u_text,
                "element": canonical,
                "species_labels": " ".join(labels),
                "folder": folder_name,
                "input_file": f"{folder_name}/input.fdf",
                "base_input_sha256": base_hash,
                "spin_file_sha256": spin_hash,
                "base_input_effective_sha256": base_effective_hash,
                "spin_file_effective_sha256": spin_effective_hash,
                "base_input_source_files": base_source_files,
                "base_input_source_files_sha256": base_source_hashes,
                "spin_file_source_files": spin_source_files,
                "spin_file_source_files_sha256": spin_source_hashes,
                "input_sha256": _sha256_bytes(patched.encode("utf-8")),
                "support_files": json.dumps(
                    [name for name, _, _ in prepared_support], ensure_ascii=False
                ),
                "support_files_sha256": json.dumps(
                    {name: digest for name, _, digest in prepared_support},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "package_version": __version__,
                "git_commit": revision,
                "git_dirty": git_dirty,
            }
        )

    manifest_path = destination / "u_sweep_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    (destination / "folders.list").write_text(
        "\n".join(folder.name for folder in folders) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    scheduler_path: Path | None = None
    if scheduler_script is not None and scheduler_suffix is not None:
        scheduler_path = destination / f"run_u_array.{scheduler_suffix}"
        scheduler_path.write_text(
            scheduler_script, encoding="utf-8", newline="\n"
        )

    return USweepPreparation(
        tuple(folders), tuple(manifest), manifest_path, scheduler_path
    )
