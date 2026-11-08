"""Magnetic-site selection and moment handling."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Sequence

from .structure import Structure


DEFAULT_ELEMENT_MOMENTS: dict[str, float] = {
    "Ti": 2.0,
    "V": 3.0,
    "Cr": 3.0,
    "Mn": 5.0,
    "Fe": 4.0,
    "Co": 3.0,
    "Ni": 2.0,
    "Cu": 1.0,
    "Gd": 7.0,
}


def built_in_element_moments(
    structure: Structure, magnetic_indices: Sequence[int]
) -> dict[str, float]:
    """Return generic high-spin guesses in first-appearance element order."""

    defaults_by_lower = {
        element.lower(): (element, value)
        for element, value in DEFAULT_ELEMENT_MOMENTS.items()
    }
    result: dict[str, float] = {}
    seen: set[str] = set()
    for index in magnetic_indices:
        symbol = structure.symbols[index]
        normalized = symbol.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        if normalized not in defaults_by_lower:
            raise ValueError(
                f"no built-in default for element {symbol}; "
                f"pass --moment {symbol}=..."
            )
        canonical, value = defaults_by_lower[normalized]
        result[canonical] = value
    return result


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
) -> tuple[float | None, dict[str, float], dict[tuple[str, int], float]]:
    """Parse global, ``Element=value``, and ``Element@CN=value`` moments."""

    items = (
        [item for item in values.replace(",", " ").split() if item]
        if isinstance(values, str)
        else list(values)
    )
    if not items:
        raise ValueError("--moment requires a value")
    global_moment: float | None = None
    by_element: dict[str, float] = {}
    by_coordination: dict[tuple[str, int], float] = {}
    for item in items:
        if "=" in item:
            key, value = item.split("=", 1)
            key = key.strip()
            if not key:
                raise ValueError(f"invalid moment specification: {item}")
            if "@" in key:
                element, coordination_text = key.split("@", 1)
                if not element.strip():
                    raise ValueError(f"invalid moment specification: {item}")
                try:
                    coordination = int(coordination_text)
                except ValueError as exc:
                    raise ValueError(
                        f"coordination in moment specification must be an integer: {item}"
                    ) from exc
                if coordination < 0:
                    raise ValueError(
                        f"coordination in moment specification must be nonnegative: {item}"
                    )
                by_coordination[(element.strip().lower(), coordination)] = abs(
                    float(value)
                )
            else:
                by_element[key.lower()] = abs(float(value))
        else:
            if global_moment is not None:
                raise ValueError("only one global --moment may be specified")
            global_moment = abs(float(item))
    return global_moment, by_element, by_coordination


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
    coordinations: dict[int, int] | None = None,
) -> dict[int, float]:
    """Resolve moment magnitudes while preserving the established return type."""

    moments, _ = resolve_moments_with_sources(
        structure,
        magnetic_indices,
        moment_values,
        site_moment_file=site_moment_file,
        coordinations=coordinations,
    )
    return moments


def resolve_moments_with_sources(
    structure: Structure,
    magnetic_indices: Sequence[int],
    moment_values: Sequence[str] | str,
    site_moment_file: str | Path | None = None,
    coordinations: dict[int, int] | None = None,
) -> tuple[dict[int, float], dict[int, str]]:
    """Resolve moments and report each atom's matching precedence source."""

    global_moment, by_element, by_coordination = parse_moment_arguments(moment_values)
    site_moments = (
        load_site_moments(site_moment_file, structure) if site_moment_file else {}
    )
    result: dict[int, float] = {}
    sources: dict[int, str] = {}
    for index in magnetic_indices:
        if index in site_moments:
            result[index] = site_moments[index]
            sources[index] = "site"
        else:
            symbol = structure.symbols[index].lower()
            coordination = coordinations.get(index) if coordinations is not None else None
            if coordination is not None and (symbol, coordination) in by_coordination:
                result[index] = by_coordination[(symbol, coordination)]
                sources[index] = "coordination"
            elif symbol in by_element:
                result[index] = by_element[symbol]
                sources[index] = "element"
            elif global_moment is not None:
                result[index] = global_moment
                sources[index] = "global"
            else:
                default = next(
                    (
                        value
                        for element, value in DEFAULT_ELEMENT_MOMENTS.items()
                        if element.lower() == symbol
                    ),
                    None,
                )
                hint = ""
                if default is not None:
                    site_specification = (
                        f"{structure.symbols[index]}@{coordination}=..."
                        if coordination is not None
                        else f"{structure.symbols[index]}=..."
                    )
                    hint = (
                        f"; the built-in default is "
                        f"{structure.symbols[index]}={default:.1f} -- pass --moment "
                        f"{site_specification} or omit --moment to use defaults for "
                        "all species"
                    )
                raise ValueError(
                    f"no initial moment specified for magnetic atom {index + 1} "
                    f"({structure.symbols[index]}"
                    + (f"@{coordination}" if coordination is not None else "")
                    + ")"
                    + hint
                )
    return result, sources


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


# Valence electron counts of the d-block (group number) used to convert a
# nominal oxidation state into a d-electron count: d^n with n = group - state.
D_BLOCK_GROUP: dict[str, int] = {
    "Sc": 3, "Ti": 4, "V": 5, "Cr": 6, "Mn": 7, "Fe": 8, "Co": 9, "Ni": 10, "Cu": 11, "Zn": 12,
    "Y": 3, "Zr": 4, "Nb": 5, "Mo": 6, "Tc": 7, "Ru": 8, "Rh": 9, "Pd": 10, "Ag": 11, "Cd": 12,
    "Hf": 4, "Ta": 5, "W": 6, "Re": 7, "Os": 8, "Ir": 9, "Pt": 10, "Au": 11, "Hg": 12,
}

# Unpaired electrons of d^n in an octahedral field: high-spin (weak field) and
# low-spin (strong field).  The low-spin values are the Oh ones; tetrahedral
# sites are almost always high-spin.
_HIGH_SPIN_UNPAIRED = (0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0)
_LOW_SPIN_UNPAIRED = (0, 1, 2, 3, 2, 1, 0, 1, 2, 1, 0)

# Nominal oxidation states of the common anions handled by the defect tools.
DEFAULT_ANION_OXIDATION_STATES: dict[str, int] = {
    "O": -2, "S": -2, "Se": -2, "Te": -2, "N": -3, "F": -1, "Cl": -1,
}


def d_electron_count(element: str, oxidation_state: float) -> float:
    """Return the formal d-electron count for a d-block element and oxidation state."""

    canonical = element.strip().capitalize()
    if canonical not in D_BLOCK_GROUP:
        raise ValueError(f"{element} is not a d-block element with a known group number")
    count = D_BLOCK_GROUP[canonical] - float(oxidation_state)
    if count < -1e-9 or count > 10 + 1e-9:
        raise ValueError(
            f"{canonical}{oxidation_state:+g} implies d^{count:g}, outside d^0..d^10"
        )
    return min(max(count, 0.0), 10.0)


def spin_only_moment(
    element: str, oxidation_state: float, spin_state: str = "high"
) -> float:
    """Return the spin-only moment (unpaired electrons, in μB) of a d^n ion.

    ``spin_state`` is ``"high"`` or ``"low"``.  A non-integer oxidation state,
    as produced by a delocalized-electron model, is interpolated linearly
    between the two neighboring integer configurations; that is a bookkeeping
    convenience, not a physical state.
    """

    state = spin_state.strip().lower()
    if state not in {"high", "low"}:
        raise ValueError("spin state must be 'high' or 'low'")
    table = _HIGH_SPIN_UNPAIRED if state == "high" else _LOW_SPIN_UNPAIRED
    count = d_electron_count(element, oxidation_state)
    lower = int(count // 1)
    fraction = count - lower
    if fraction < 1e-9 or lower >= 10:
        return float(table[min(lower, 10)])
    return float(table[lower] * (1.0 - fraction) + table[lower + 1] * fraction)


def parse_oxidation_state_arguments(
    values: Sequence[str] | str,
) -> tuple[dict[str, float], dict[tuple[str, int], float]]:
    """Parse signed ``Element=+3`` and ``Element@CN=+2`` oxidation states."""

    items = (
        [item for item in values.replace(",", " ").split() if item]
        if isinstance(values, str)
        else [token for raw in values for token in str(raw).replace(",", " ").split()]
    )
    if not items:
        raise ValueError("--oxidation-states requires at least one Element=value")
    by_element: dict[str, float] = {}
    by_coordination: dict[tuple[str, int], float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(
                f"invalid oxidation state specification {item!r}; use Element=+3 or Element@CN=+2"
            )
        key, value = item.split("=", 1)
        key = key.strip()
        try:
            state = float(value)
        except ValueError as exc:
            raise ValueError(f"invalid oxidation state value in {item!r}") from exc
        if "@" in key:
            element, cn_text = key.split("@", 1)
            if not element.strip() or not cn_text.isdigit():
                raise ValueError(f"invalid oxidation state specification {item!r}")
            by_coordination[(element.strip().lower(), int(cn_text))] = state
        elif key:
            by_element[key.lower()] = state
        else:
            raise ValueError(f"invalid oxidation state specification {item!r}")
    return by_element, by_coordination
