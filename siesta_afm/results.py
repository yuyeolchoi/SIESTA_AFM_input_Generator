"""Prepare SIESTA job arrays and collect common magnetic result fields.

``collect_results`` also understands CP2K job folders (an ``input.inp`` beside
the output) and reports both codes in one table; energies are in eV for both.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import math
import re
import shutil
from pathlib import Path

from .cp2k_results import find_cp2k_input, parse_cp2k_initial_spins, parse_cp2k_output
from .fdf_writer import patch_fdf_text
from .io import parse_dm_init_spin, read_fdf_with_includes, read_structure


@dataclass(frozen=True, slots=True)
class BandGapResult:
    """A gap estimated from the SCF k-point eigenvalues in a SIESTA EIG file."""

    gap_ev: float
    kind: str
    vbm_ev: float
    cbm_ev: float
    fermi_ev: float


def prepare_array(
    input_fdf: str | Path,
    configs_dir: str | Path,
    output_dir: str | Path,
    *,
    template: str | Path | None = None,
) -> list[Path]:
    source = Path(input_fdf)
    configs = Path(configs_dir)
    destination = Path(output_dir)
    manifest_path = configs / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"configuration manifest not found: {manifest_path}")
    destination.mkdir(parents=True, exist_ok=True)
    input_text = read_fdf_with_includes(source)
    folders: list[Path] = []
    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            config_id = row["config_id"]
            method = row["method"]
            spin_path = configs / row["file"]
            folder = destination / f"{config_id}_{method}"
            folder.mkdir(parents=True, exist_ok=True)
            patched = patch_fdf_text(
                input_text, spin_path.read_text(encoding="utf-8-sig")
            )
            (folder / "input.fdf").write_text(patched, encoding="utf-8")
            if template:
                template_path = Path(template)
                shutil.copy2(template_path, folder / template_path.name)
                run_text = f"%include {template_path.name}\n%include input.fdf\n"
            else:
                run_text = "%include input.fdf\n"
            (folder / "RUN.fdf").write_text(run_text, encoding="utf-8")
            folders.append(folder)
    (destination / "folders.list").write_text(
        "\n".join(folder.name for folder in folders) + "\n", encoding="utf-8"
    )
    return folders


_ENERGY_PATTERNS = [
    re.compile(r"(?im)^\s*siesta:\s+E_KS\(eV\)\s*=\s*([-+0-9.eEdD]+)"),
    re.compile(r"(?im)Total\s+energy\s*[:=]\s*([-+0-9.eEdD]+)"),
]
_NET_SPIN_PATTERNS = [
    re.compile(
        r"(?im)^\s*siesta:\s+Total\s+spin\s+polarization\s*"
        r"\(\s*Qup\s*-\s*Qdown\s*\)\s*=\s*([-+0-9.eEdD]+)"
    ),
    re.compile(
        r"(?im)Total\s+(?:spin|magnetic)\s+(?:moment)?\s*[:=]\s*([-+0-9.eEdD]+)"
    ),
    re.compile(r"(?im)spin\s+moment\s*[:=]\s*([-+0-9.eEdD]+)"),
]
_ATOMIC_POPULATION_HEADING_RE = re.compile(
    r"(?im)^\s*(?:mulliken:\s*)?"
    r"(?P<method>Mulliken|Hirshfeld|Voronoi)\s+(?:Net\s+)?"
    r"Atomic\s+Populations:\s*$"
)
_OLD_MULLIKEN_HEADING_RE = re.compile(
    r"(?im)^\s*mulliken:\s+Atomic\s+and\s+Orbital\s+Populations:\s*$"
)
_FLOAT_TOKEN_RE = re.compile(r"^[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][-+]?\d+)?$")


def _last_float(patterns: list[re.Pattern[str]], text: str) -> float | None:
    matches = [match for pattern in patterns for match in pattern.finditer(text)]
    if not matches:
        return None
    last = max(matches, key=lambda item: item.start())
    return float(last.group(1).replace("D", "E").replace("d", "e"))


def _find_output(folder: Path) -> Path | None:
    candidates = {
        path.resolve(): path for path in [*folder.glob("*.out"), *folder.glob("*.log")]
    }
    return (
        max(candidates.values(), key=lambda path: path.stat().st_mtime_ns)
        if candidates
        else None
    )


def _find_eig(folder: Path) -> Path | None:
    candidates = [
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() == ".eig"
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime_ns) if candidates else None


def parse_eig_band_gap(
    text_or_path: str | Path, *, fermi_tolerance: float = 1e-8
) -> BandGapResult:
    """Estimate the sampled fundamental gap from a standard SIESTA EIG file.

    The EIG format stores the Fermi level, ``nbands nspin nkpoints``, and one
    indexed eigenvalue record per k point.  This estimate is limited to the SCF
    sampling mesh; a publication-quality direct/indirect gap still requires an
    adequately converged mesh or a dedicated band path.
    """

    if not math.isfinite(fermi_tolerance) or fermi_tolerance < 0:
        raise ValueError("Fermi tolerance must be finite and nonnegative")
    if isinstance(text_or_path, Path):
        text = text_or_path.read_text(encoding="utf-8", errors="replace")
    else:
        possible_path = Path(text_or_path)
        try:
            is_file = "\n" not in text_or_path and possible_path.is_file()
        except OSError:
            is_file = False
        if is_file:
            text = possible_path.read_text(encoding="utf-8", errors="replace")
        else:
            text = text_or_path
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 3:
        raise ValueError("truncated SIESTA EIG file")
    try:
        fermi = float(lines[0].split()[0].replace("D", "E").replace("d", "e"))
        dimensions = [int(value) for value in lines[1].split()[:3]]
    except (ValueError, IndexError) as exc:
        raise ValueError("invalid SIESTA EIG header") from exc
    if len(dimensions) != 3:
        raise ValueError("SIESTA EIG header requires nbands nspin nkpoints")
    n_bands, n_spin, n_kpoints = dimensions
    if n_bands < 1 or n_spin not in {1, 2} or n_kpoints < 1:
        raise ValueError("invalid SIESTA EIG dimensions")
    if not math.isfinite(fermi):
        raise ValueError("nonfinite Fermi level in SIESTA EIG file")

    tokens = " ".join(lines[2:]).split()
    per_kpoint = n_bands * n_spin
    expected = n_kpoints * (per_kpoint + 1)
    if len(tokens) != expected:
        raise ValueError(
            f"SIESTA EIG data has {len(tokens)} tokens; expected {expected}"
        )
    entries: list[tuple[int, int, int, float]] = []
    offset = 0
    for expected_index in range(1, n_kpoints + 1):
        try:
            index = int(tokens[offset])
        except ValueError as exc:
            raise ValueError("invalid SIESTA EIG k-point index") from exc
        if index != expected_index:
            raise ValueError(
                f"unexpected SIESTA EIG k-point index {index}; expected {expected_index}"
            )
        offset += 1
        for spin in range(n_spin):
            for band in range(n_bands):
                try:
                    energy = float(
                        tokens[offset].replace("D", "E").replace("d", "e")
                    )
                except ValueError as exc:
                    raise ValueError("invalid eigenvalue in SIESTA EIG file") from exc
                if not math.isfinite(energy):
                    raise ValueError("nonfinite eigenvalue in SIESTA EIG file")
                entries.append((index, spin, band, energy))
                offset += 1

    bands = {
        (spin, band): [
            energy
            for _, entry_spin, entry_band, energy in entries
            if entry_spin == spin and entry_band == band
        ]
        for spin in range(n_spin)
        for band in range(n_bands)
    }
    touches_fermi = any(
        abs(energy - fermi) <= fermi_tolerance
        for _, _, _, energy in entries
    )
    crosses_fermi = any(
        min(energies) < fermi - fermi_tolerance
        and max(energies) > fermi + fermi_tolerance
        for energies in bands.values()
    )
    if touches_fermi or crosses_fermi:
        return BandGapResult(0.0, "metallic", fermi, fermi, fermi)
    occupied = [entry for entry in entries if entry[3] < fermi]
    empty = [entry for entry in entries if entry[3] > fermi]
    if not occupied or not empty:
        raise ValueError("SIESTA EIG spectrum does not bracket the Fermi level")
    vbm = max(entry[3] for entry in occupied)
    cbm = min(entry[3] for entry in empty)
    gap = max(0.0, cbm - vbm)
    vbm_k = {
        index for index, _, _, energy in occupied if abs(energy - vbm) <= 1e-10
    }
    cbm_k = {
        index for index, _, _, energy in empty if abs(energy - cbm) <= 1e-10
    }
    kind = "direct" if vbm_k & cbm_k else "indirect"
    return BandGapResult(gap, kind, vbm, cbm, fermi)


def _u_sweep_metadata(root: Path) -> dict[str, dict[str, str]]:
    manifest = root / "u_sweep_manifest.csv"
    if not manifest.is_file():
        return {}
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"folder", "config_id", "u_eff_ev", "element"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                "U-sweep manifest is missing required column(s): "
                + ", ".join(sorted(missing))
            )
        rows = [dict(row) for row in reader]
    metadata: dict[str, dict[str, str]] = {}
    for row in rows:
        folder = row["folder"].strip()
        if not folder:
            raise ValueError("U-sweep manifest contains an empty folder")
        if folder in metadata:
            raise ValueError(f"duplicate U-sweep manifest folder: {folder}")
        metadata[folder] = row
    return metadata


def _read_exit_code(folder: Path) -> tuple[int | None, bool]:
    """Return a scheduler wrapper exit code and whether its file is invalid."""

    path = folder / "exit_code"
    if not path.is_file():
        return None, False
    try:
        fields = path.read_text(encoding="utf-8-sig").split()
        if len(fields) != 1:
            raise ValueError
        return int(fields[0]), False
    except (OSError, ValueError):
        return None, True


def _float_token(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


def _numeric_atom_row(line: str) -> tuple[int, list[float]] | None:
    """Parse an atomic-population row up to its first nonnumeric field."""

    fields = line.split()
    if not fields or not fields[0].isdigit():
        return None
    values: list[float] = []
    for token in fields[1:]:
        if not _FLOAT_TOKEN_RE.match(token):
            break
        values.append(_float_token(token))
    return (int(fields[0]), values) if values else None


def _parse_common_atomic_spin_tables(
    text: str,
) -> list[tuple[int, dict[int, float], str]]:
    """Parse SIESTA 5.x common Mulliken/Hirshfeld/Voronoi summaries.

    Collinear tables expose a signed ``Sz`` (older variants may call it
    ``S`` or ``Spin``) as the final numeric column before the species label.
    """

    headings = list(_ATOMIC_POPULATION_HEADING_RE.finditer(text))
    tables: list[tuple[int, dict[int, float], str]] = []
    for number, heading in enumerate(headings):
        end = headings[number + 1].start() if number + 1 < len(headings) else len(text)
        segment = text[heading.end() : end]
        lines = segment.splitlines()
        header_index: int | None = None
        for index, line in enumerate(lines):
            normalized = line.lower()
            has_spin_column = bool(re.search(r"(?i)(?:\bSz\b|\bSpin\b|\bS\b)", line))
            if "atom" in normalized and "species" in normalized and has_spin_column:
                header_index = index
                break
        if header_index is None:
            continue
        spins: dict[int, float] = {}
        for line in lines[header_index + 1 :]:
            if re.match(r"^\s*(?:-+|Total\b)", line, re.IGNORECASE):
                if spins:
                    break
                continue
            row = _numeric_atom_row(line)
            if row is None:
                continue
            atom_index, values = row
            # The common SIESTA format places signed Sz/S/Spin last.
            spins[atom_index] = values[-1]
        if spins:
            method = heading.group("method").lower()
            tables.append((heading.start(), spins, f"{method}-atomic-populations"))
    return tables


def _parse_old_spin_resolved_mulliken(
    text: str,
) -> list[tuple[int, dict[int, float], str]]:
    """Parse SIESTA 4.1-style Qatom tables and return Qup-Qdown per atom."""

    headings = list(_OLD_MULLIKEN_HEADING_RE.finditer(text))
    tables: list[tuple[int, dict[int, float], str]] = []
    for number, heading in enumerate(headings):
        end = headings[number + 1].start() if number + 1 < len(headings) else len(text)
        lines = text[heading.end() : end].splitlines()
        channels: dict[str, dict[int, float]] = {"up": {}, "down": {}}
        channel: str | None = None
        reading_rows = False
        for line in lines:
            spin_match = re.search(r"(?i)mulliken:\s*Spin\s+(UP|DOWN)\b", line)
            if spin_match:
                channel = spin_match.group(1).lower()
                reading_rows = False
                continue
            if re.search(r"(?i)^\s*Atom\s+Qatom\s+Qorb\b", line):
                reading_rows = channel is not None
                continue
            if re.search(r"(?i)mulliken:\s*Qtot\b|Atomic\s+Populations:", line):
                reading_rows = False
                continue
            if not reading_rows or channel is None:
                continue
            row = _numeric_atom_row(line)
            if row is not None:
                atom_index, values = row
                channels[channel][atom_index] = values[0]
        common = channels["up"].keys() & channels["down"].keys()
        if common:
            spins = {
                index: channels["up"][index] - channels["down"][index]
                for index in sorted(common)
            }
            tables.append((heading.start(), spins, "mulliken-spin-channels"))
    return tables


def parse_local_spins(text: str) -> tuple[dict[int, float], str | None]:
    """Return the last SIESTA atomic spin table and a diagnostic source."""

    tables = [
        *_parse_old_spin_resolved_mulliken(text),
        *_parse_common_atomic_spin_tables(text),
    ]
    if not tables:
        return {}, None
    _, spins, source = max(tables, key=lambda item: item[0])
    return spins, source


def collect_results(
    jobs_dir: str | Path,
    *,
    output_csv: str | Path | None = None,
    collapse_initial: float = 0.5,
    collapse_final: float = 0.1,
) -> list[dict[str, object]]:
    root = Path(jobs_dir)
    folders_list = root / "folders.list"
    if folders_list.is_file():
        folders = [
            root / name.strip()
            for name in folders_list.read_text(encoding="utf-8-sig").splitlines()
            if name.strip()
        ]
    else:
        folders = sorted(path for path in root.iterdir() if path.is_dir())
    sweep_metadata = _u_sweep_metadata(root)
    results: list[dict[str, object]] = []
    failed_indices: list[int] = []
    failed_folders: list[str] = []
    for array_index, folder in enumerate(folders, start=1):
        metadata = sweep_metadata.get(folder.name, {})
        if sweep_metadata and not metadata:
            raise ValueError(
                f"jobs folder is missing from U-sweep manifest: {folder.name}"
            )
        config_id = metadata.get("config_id", folder.name.split("_", 1)[0])
        shared = {
            "config_id": config_id,
            "u_eff_ev": metadata.get("u_eff_ev", ""),
            "element": metadata.get("element", ""),
            "species_labels": metadata.get("species_labels", ""),
        }
        initial_path = folder / "input.fdf"
        cp2k_input = None if initial_path.is_file() else find_cp2k_input(folder)
        code = "cp2k" if cp2k_input is not None else "siesta"
        if cp2k_input is not None:
            initial = parse_cp2k_initial_spins(cp2k_input)
        else:
            initial_rows = (
                parse_dm_init_spin(initial_path) if initial_path.is_file() else []
            )
            initial = {index: spin for index, spin in initial_rows}
        exit_code, invalid_exit_code = _read_exit_code(folder)
        if exit_code is not None and exit_code != 0:
            failed_indices.append(array_index)
            failed_folders.append(folder.name)
        output = _find_output(folder)
        if output is None:
            results.append(
                {
                    **shared,
                    "total_energy": "",
                    "band_gap_ev": "",
                    "band_gap_type": "",
                    "band_gap_source": "",
                    "band_gap_status": (
                        "not-available-cp2k" if code == "cp2k" else "missing-eig"
                    ),
                    "band_gap_error": "",
                    "final_net_spin": "",
                    "local_spins": "",
                    "local_spin_mean_abs": "",
                    "local_spin_min_abs": "",
                    "local_spin_max_abs": "",
                    "sign_retention": "",
                    "collapsed_atoms": "",
                    "spin_population_source": "",
                    "exit_code": exit_code if exit_code is not None else "",
                    "scf_converged": False,
                    "geometry_converged": False,
                    "status": (
                        "invalid-exit-code"
                        if invalid_exit_code
                        else "run-failed"
                        if exit_code not in {None, 0}
                        else "missing-output"
                    ),
                    "code": code,
                    "notes": "",
                }
            )
            continue
        text = output.read_text(encoding="utf-8", errors="replace")
        notes: list[str] = []
        cp2k_aborted = False
        if code == "cp2k":
            summary = parse_cp2k_output(text)
            local, spin_source = summary.local_spins, summary.spin_source
            total_energy: float | None = summary.energy_ev
            net_spin: float | None = summary.net_spin
            scf = summary.scf_converged
            geometry = summary.geometry_converged
            notes = list(summary.notes)
            cp2k_aborted = summary.aborted
        else:
            local, spin_source = parse_local_spins(text)
            total_energy = _last_float(_ENERGY_PATTERNS, text)
            net_spin = _last_float(_NET_SPIN_PATTERNS, text)
            lower = text.lower()
            scf = any(
                marker in lower
                for marker in (
                    "scf cycle converged",
                    "scf converged",
                    "scf convergence achieved",
                )
            )
            geometry = any(
                marker in lower
                for marker in (
                    "geometry converged",
                    "geom. converged",
                    "optimization converged",
                )
            )
        comparable = [
            index for index in initial if index in local and initial[index] != 0
        ]
        retained = (
            sum(initial[index] * local[index] > 0 for index in comparable)
            / len(comparable)
            if comparable
            else None
        )
        collapsed = (
            sum(
                abs(initial[index]) > collapse_initial
                and index in local
                and abs(local[index]) < collapse_final
                for index in initial
            )
            if local
            else None
        )
        if metadata.get("element") and initial_path.is_file():
            try:
                structure = read_structure(initial_path)
                target_indices = [
                    index + 1
                    for index, symbol in enumerate(structure.symbols)
                    if symbol.lower() == metadata["element"].lower()
                ]
            except (OSError, ValueError):
                target_indices = [
                    index for index, value in initial.items() if value != 0
                ]
        else:
            target_indices = [
                index for index, value in initial.items() if value != 0
            ]
        target_local = {
            index: local[index] for index in target_indices if index in local
        }
        target_absolute = [abs(value) for value in target_local.values()]
        expected_local = {index for index, value in initial.items() if value != 0}
        matched_local = expected_local & local.keys()
        if not local:
            status = "spin-table-not-found"
        elif expected_local and not matched_local:
            status = "spin-table-no-matching-atoms"
        elif matched_local != expected_local:
            status = "spin-table-partial"
        else:
            status = "ok" if scf else "not-converged"
        if invalid_exit_code:
            status = "invalid-exit-code"
        elif exit_code not in {None, 0}:
            status = "run-failed"
        elif cp2k_aborted and status in {"ok", "not-converged"}:
            status = "aborted"
        eig_path = None if code == "cp2k" else _find_eig(folder)
        band_gap: BandGapResult | None = None
        band_gap_status = "not-available-cp2k" if code == "cp2k" else "missing-eig"
        band_gap_error = ""
        if eig_path is not None:
            try:
                band_gap = parse_eig_band_gap(eig_path)
                band_gap_status = "ok"
            except ValueError as exc:
                band_gap_status = "parse-error"
                band_gap_error = str(exc)
        results.append(
            {
                **shared,
                "total_energy": total_energy,
                "band_gap_ev": band_gap.gap_ev if band_gap else "",
                "band_gap_type": band_gap.kind if band_gap else "",
                "band_gap_source": eig_path.name if eig_path else "",
                "band_gap_status": band_gap_status,
                "band_gap_error": band_gap_error,
                "final_net_spin": net_spin,
                "local_spins": " ".join(
                    f"{index}:{value:.12g}"
                    for index, value in target_local.items()
                ),
                "local_spin_mean_abs": (
                    sum(target_absolute) / len(target_absolute)
                    if target_absolute
                    else ""
                ),
                "local_spin_min_abs": min(target_absolute) if target_absolute else "",
                "local_spin_max_abs": max(target_absolute) if target_absolute else "",
                "sign_retention": retained,
                "collapsed_atoms": collapsed,
                "spin_population_source": spin_source or "",
                "exit_code": exit_code if exit_code is not None else "",
                "scf_converged": scf,
                "geometry_converged": geometry,
                "status": status,
                "code": code,
                "notes": "; ".join(notes),
            }
        )
    destination = Path(output_csv) if output_csv else root / "results.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "config_id",
        "u_eff_ev",
        "element",
        "species_labels",
        "total_energy",
        "band_gap_ev",
        "band_gap_type",
        "band_gap_source",
        "band_gap_status",
        "band_gap_error",
        "final_net_spin",
        "local_spins",
        "local_spin_mean_abs",
        "local_spin_min_abs",
        "local_spin_max_abs",
        "sign_retention",
        "collapsed_atoms",
        "spin_population_source",
        "exit_code",
        "scf_converged",
        "geometry_converged",
        "status",
        "code",
        "notes",
    ]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    if sweep_metadata:
        (root / "failed_array_indices.txt").write_text(
            ",".join(str(index) for index in failed_indices) + (
                "\n" if failed_indices else ""
            ),
            encoding="utf-8",
        )
        (root / "failed_folders.list").write_text(
            "\n".join(failed_folders) + ("\n" if failed_folders else ""),
            encoding="utf-8",
        )
    return results
