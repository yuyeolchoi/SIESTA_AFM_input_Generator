"""Magnetic-site selection and moment handling."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Sequence

from .structure import Structure


def parse_atom_indices(values: str | Iterable[str] | None) -> set[int]:
    """Parse one-based indices and inclusive ranges such as ``2,5,8-11``."""

    if values is None:
        return set()
    tokens: list[str] = []
    source = [values] if isinstance(values, str) else list(values)
    for item in source:
        tokens.extend(part.strip() for part in str(item).split(",") if part.strip())
    result: set[int] = set()
    for token in tokens:
        if "-" in token[1:]:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start <= 0 or end < start:
                raise ValueError(f"invalid atom range: {token}")
            result.update(range(start, end + 1))
        else:
            index = int(token)
            if index <= 0:
                raise ValueError("atom indices are one-based positive integers")
            result.add(index)
    return result


def select_magnetic_sites(
    structure: Structure,
    species: Sequence[str],
    *,
    exclude_atoms: str | Iterable[str] | None = None,
    adsorbate_indices: str | Iterable[str] | None = None,
) -> list[int]:
    """Select magnetic atoms and return zero-based indices in input order."""

    wanted = {item.strip().lower() for item in species if item.strip()}
    if not wanted:
        raise ValueError("at least one magnetic species must be specified")
    excluded = parse_atom_indices(exclude_atoms) | parse_atom_indices(adsorbate_indices)
    invalid = sorted(index for index in excluded if index > len(structure))
    if invalid:
        raise ValueError(f"excluded atom index out of range: {invalid[0]}")
    selected = [
        i
        for i, symbol in enumerate(structure.symbols)
        if symbol.lower() in wanted and i + 1 not in excluded
    ]
    if not selected:
        raise ValueError(
            "no magnetic atoms matched --magnetic-species after exclusions"
        )
    return selected


def parse_moment_arguments(
    values: Sequence[str] | str,
) -> tuple[float | None, dict[str, float]]:
    """Parse either a global moment or element-specific ``Element=value`` pairs."""

    items = [values] if isinstance(values, str) else list(values)
    if not items:
        raise ValueError("--moment requires a value")
    global_moment: float | None = None
    by_element: dict[str, float] = {}
    for item in items:
        if "=" in item:
            element, value = item.split("=", 1)
            if not element.strip():
                raise ValueError(f"invalid moment specification: {item}")
            by_element[element.strip().lower()] = abs(float(value))
        else:
            if global_moment is not None or by_element:
                raise ValueError("a global --moment cannot be mixed with element=value")
            global_moment = abs(float(item))
    return global_moment, by_element


def load_moment_config(path: str | Path) -> list[str]:
    """Load the documented ``moments: {Element: value}`` YAML schema."""

    text = Path(path).read_text(encoding="utf-8-sig")
    try:
        import yaml

        data = yaml.safe_load(text) or {}
        moments = data.get("moments", {})
        if not isinstance(moments, dict):
            raise ValueError("moment config 'moments' must be a mapping")
        result = [f"{element}={float(value)}" for element, value in moments.items()]
    except ImportError:
        result = []
        in_moments = False
        for raw in text.splitlines():
            line = raw.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            if line.strip() == "moments:":
                in_moments = True
                continue
            if in_moments and ":" in line and line[0].isspace():
                element, value = line.strip().split(":", 1)
                result.append(f"{element.strip()}={float(value.strip())}")
            elif in_moments and not line[0].isspace():
                break
    if not result:
        raise ValueError("moment config contains no moments")
    return result


def load_site_moments(path: str | Path, structure: Structure) -> dict[int, float]:
    """Read site-specific moments from CSV, keyed by zero-based atom index."""

    result: dict[int, float] = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"atom_index", "moment"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("site moment CSV requires atom_index and moment columns")
        for row in reader:
            one_based = int(row["atom_index"])
            if not 1 <= one_based <= len(structure):
                raise ValueError(f"site moment atom index out of range: {one_based}")
            if (
                row.get("element")
                and row["element"].strip().lower()
                != structure.symbols[one_based - 1].lower()
            ):
                raise ValueError(
                    f"site moment element mismatch at atom {one_based}: "
                    f"{row['element']} != {structure.symbols[one_based - 1]}"
                )
            if one_based - 1 in result:
                raise ValueError(f"duplicate site moment atom index: {one_based}")
            result[one_based - 1] = abs(float(row["moment"]))
    return result


def resolve_moments(
    structure: Structure,
    magnetic_indices: Sequence[int],
    moment_values: Sequence[str] | str,
    site_moment_file: str | Path | None = None,
) -> dict[int, float]:
    global_moment, by_element = parse_moment_arguments(moment_values)
    site_moments = (
        load_site_moments(site_moment_file, structure) if site_moment_file else {}
    )
    result: dict[int, float] = {}
    for index in magnetic_indices:
        if index in site_moments:
            result[index] = site_moments[index]
        elif global_moment is not None:
            result[index] = global_moment
        else:
            symbol = structure.symbols[index].lower()
            if symbol not in by_element:
                raise ValueError(
                    f"no initial moment specified for magnetic species "
                    f"{structure.symbols[index]}"
                )
            result[index] = by_element[symbol]
    return result


def guess_oxidation_states(structure: Structure) -> list[float]:
    """Opt-in pymatgen oxidation-state guess, deliberately never automatic."""

    try:
        from pymatgen.core import Lattice, Structure as PmgStructure
    except ImportError as exc:
        raise RuntimeError(
            "--guess-oxidation-states requires the optional pymatgen dependency"
        ) from exc
    pmg = PmgStructure(
        Lattice(structure.cell), structure.symbols, structure.fractional_positions
    )
    guessed = pmg.copy()
    guessed.add_oxidation_state_by_guess()
    return [float(site.specie.oxi_state) for site in guessed]
