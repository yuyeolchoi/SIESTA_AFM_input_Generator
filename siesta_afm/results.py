"""Prepare SIESTA job arrays and collect common magnetic result fields."""

from __future__ import annotations

import csv
import re
import shutil
from pathlib import Path

from .fdf_writer import patch_fdf_text
from .io import parse_dm_init_spin


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
    input_text = source.read_text(encoding="utf-8-sig")
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
    results: list[dict[str, object]] = []
    for folder in folders:
        initial_path = folder / "input.fdf"
        initial_rows = (
            parse_dm_init_spin(initial_path) if initial_path.is_file() else []
        )
        initial = {index: spin for index, spin in initial_rows}
        output = _find_output(folder)
        if output is None:
            results.append(
                {
                    "config_id": folder.name.split("_", 1)[0],
                    "total_energy": "",
                    "final_net_spin": "",
                    "sign_retention": "",
                    "collapsed_atoms": "",
                    "spin_population_source": "",
                    "scf_converged": False,
                    "geometry_converged": False,
                    "status": "missing-output",
                }
            )
            continue
        text = output.read_text(encoding="utf-8", errors="replace")
        local, spin_source = parse_local_spins(text)
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
        results.append(
            {
                "config_id": folder.name.split("_", 1)[0],
                "total_energy": _last_float(_ENERGY_PATTERNS, text),
                "final_net_spin": _last_float(_NET_SPIN_PATTERNS, text),
                "sign_retention": retained,
                "collapsed_atoms": collapsed,
                "spin_population_source": spin_source or "",
                "scf_converged": scf,
                "geometry_converged": geometry,
                "status": status,
            }
        )
    destination = Path(output_csv) if output_csv else root / "results.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "config_id",
        "total_energy",
        "final_net_spin",
        "sign_retention",
        "collapsed_atoms",
        "spin_population_source",
        "scf_converged",
        "geometry_converged",
        "status",
    ]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    return results
