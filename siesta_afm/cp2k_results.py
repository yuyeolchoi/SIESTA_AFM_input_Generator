"""Parse CP2K Quickstep output for the common result fields.

CP2K appends to an existing output file instead of overwriting it, so every
parser here looks only at the text after the last ``PROGRAM STARTED AT``
banner and reports how many runs the file holds.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from ase.units import Hartree

from .cp2k_writer import parse_cp2k_coord


_PROGRAM_STARTED_RE = re.compile(r"PROGRAM STARTED AT")
_PROGRAM_ENDED_RE = re.compile(r"PROGRAM ENDED AT")
_ABORT_RE = re.compile(r"\[ABORT\]|CP2K\|\s*Abnormal program termination", re.IGNORECASE)
_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][-+]?\d+)?"
_ENERGY_RE = re.compile(
    r"ENERGY\|\s*Total FORCE_EVAL\s*\(\s*\w+\s*\)\s*energy(?:\s*\[hartree\]|\s*\(a\.u\.\))?:?\s*(" + _FLOAT + r")"
)
_MULLIKEN_HEADING_RE = re.compile(r"Mulliken Population Analysis")
_MULLIKEN_ROW_RE = re.compile(
    r"^\s*(\d+)\s+([A-Z][a-z]?)\s+(\d+)\s+((?:" + _FLOAT + r"\s+)*" + _FLOAT + r")\s*$"
)
_MULLIKEN_TOTAL_RE = re.compile(r"^\s*#\s*Total charge(?: and spin)?\s+(.*)$", re.IGNORECASE)
_SCF_CONVERGED_RE = re.compile(r"SCF run converged in|outer SCF loop converged in")
_SCF_FAILED_RE = re.compile(r"SCF run NOT converged|outer SCF loop FAILED", re.IGNORECASE)
_GEOMETRY_RE = re.compile(r"GEOMETRY OPTIMIZATION COMPLETED")
_PLUS_U_NEGATIVE_RE = re.compile(r"DFT\+U energy contribution is negative")
_MAGNETIZATION_RE = re.compile(r"^\s*MAGNETIZATION\s+(" + _FLOAT + r")", re.IGNORECASE)


@dataclass(slots=True)
class CP2KRunSummary:
    """Result fields of the last CP2K run in an output file."""

    run_count: int
    ended_normally: bool
    aborted: bool
    energy_hartree: float | None
    energy_ev: float | None
    local_spins: dict[int, float]
    net_spin: float | None
    spin_source: str | None
    scf_converged: bool
    geometry_converged: bool
    plus_u_negative_warnings: int
    notes: list[str] = field(default_factory=list)


def last_cp2k_run(text: str) -> tuple[str, int]:
    """Return the text of the last run and the number of runs in the file."""

    starts = [match.start() for match in _PROGRAM_STARTED_RE.finditer(text)]
    if not starts:
        return text, 1 if text.strip() else 0
    return text[starts[-1] :], len(starts)


def _float(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


def parse_cp2k_energy(text: str) -> float | None:
    """Return the last FORCE_EVAL total energy in hartree, or None."""

    matches = list(_ENERGY_RE.finditer(text))
    return _float(matches[-1].group(1)) if matches else None


def parse_cp2k_mulliken_spins(text: str) -> tuple[dict[int, float], float | None]:
    """Return per-atom spin moments and the total spin from the last Mulliken table.

    Only unrestricted (UKS) tables carry a spin column: rows then hold alpha
    and beta populations, the net charge, and the spin moment.  Restricted
    tables yield an empty mapping.
    """

    headings = list(_MULLIKEN_HEADING_RE.finditer(text))
    if not headings:
        return {}, None
    segment = text[headings[-1].end() :]
    spins: dict[int, float] = {}
    total_spin: float | None = None
    for line in segment.splitlines()[1:]:
        total = _MULLIKEN_TOTAL_RE.match(line)
        if total:
            numbers = re.findall(_FLOAT, total.group(1))
            if len(numbers) >= 4:
                total_spin = _float(numbers[-1])
            break
        row = _MULLIKEN_ROW_RE.match(line)
        if row:
            values = [_float(item) for item in row.group(4).split()]
            if len(values) >= 4:
                spins[int(row.group(1))] = values[3]
            continue
        if spins and line.strip().startswith("!"):
            break
    return spins, total_spin


def parse_cp2k_output(text: str) -> CP2KRunSummary:
    """Summarize the last run of a CP2K output text."""

    run, run_count = last_cp2k_run(text)
    notes: list[str] = []
    if run_count > 1:
        notes.append(
            f"output holds {run_count} appended runs; only the last one was read"
        )
    energy = parse_cp2k_energy(run)
    spins, total_spin = parse_cp2k_mulliken_spins(run)
    converged_positions = [match.start() for match in _SCF_CONVERGED_RE.finditer(run)]
    failed_positions = [match.start() for match in _SCF_FAILED_RE.finditer(run)]
    scf = bool(converged_positions) and (
        not failed_positions or converged_positions[-1] > failed_positions[-1]
    )
    aborted = bool(_ABORT_RE.search(run))
    ended = bool(_PROGRAM_ENDED_RE.search(run))
    plus_u_negative = len(_PLUS_U_NEGATIVE_RE.findall(run))
    if plus_u_negative:
        notes.append(
            f"'DFT+U energy contribution is negative' appeared {plus_u_negative} "
            "times; treat the +U projection setup as suspect"
        )
    if aborted:
        notes.append("CP2K reported an ABORT")
    elif not ended:
        notes.append("no PROGRAM ENDED banner; the run may still be running or was killed")
    return CP2KRunSummary(
        run_count=run_count,
        ended_normally=ended and not aborted,
        aborted=aborted,
        energy_hartree=energy,
        energy_ev=energy * Hartree if energy is not None else None,
        local_spins=spins,
        net_spin=total_spin,
        spin_source="cp2k-mulliken-population-analysis" if spins else None,
        scf_converged=scf,
        geometry_converged=bool(_GEOMETRY_RE.search(run)),
        plus_u_negative_warnings=plus_u_negative,
        notes=notes,
    )


def parse_cp2k_initial_spins(text_or_path: str | Path) -> dict[int, float]:
    """Return one-based initial spins implied by ``&KIND`` MAGNETIZATION values."""

    candidate = Path(text_or_path) if not isinstance(text_or_path, Path) else text_or_path
    try:
        is_file = candidate.is_file()
    except OSError:
        is_file = False
    text = candidate.read_text(encoding="utf-8-sig") if is_file else str(text_or_path)
    magnetization: dict[str, float] = {}
    current: str | None = None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].split("!", 1)[0].strip()
        if not line:
            continue
        upper = line.upper()
        if upper.startswith("&KIND"):
            fields = line.split()
            current = fields[1] if len(fields) > 1 else None
            if current is not None:
                magnetization.setdefault(current, 0.0)
            continue
        if upper.startswith("&END") and current is not None and "KIND" in upper:
            current = None
            continue
        if current is not None:
            match = _MAGNETIZATION_RE.match(line)
            if match:
                magnetization[current] = _float(match.group(1))
    rows = parse_cp2k_coord(text)
    result: dict[int, float] = {}
    for number, (kind, _) in enumerate(rows, start=1):
        if kind not in magnetization:
            raise ValueError(f"&COORD uses kind {kind!r} without a matching &KIND block")
        result[number] = magnetization[kind]
    return result


def find_cp2k_input(folder: Path) -> Path | None:
    """Return the CP2K input of a job folder, preferring ``input.inp``."""

    preferred = folder / "input.inp"
    if preferred.is_file():
        return preferred
    candidates = sorted(folder.glob("*.inp"))
    return candidates[0] if len(candidates) == 1 else None
