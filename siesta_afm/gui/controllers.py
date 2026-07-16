"""Tk-free workflow controllers for the desktop interface."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..fdf_writer import patch_fdf_text, render_dm_init_spin
from ..input_template import InputTemplateResult, render_complete_input
from ..io import parse_dm_init_spin, read_structure
from ..magnetic_sites import (
    DEFAULT_ELEMENT_MOMENTS,
    select_magnetic_sites,
)
from ..neighbors import classify_coordination_geometry
from ..ordering import SpinAssignment, analyze_coordination_sites
from ..results import collect_results, prepare_array
from ..structure import Structure
from ..validation import (
    ValidationReport,
    analyze_structure,
    format_analysis,
    format_validation,
    validate_spins,
)
from ..visualize import create_spin_figure, plot_spin_pattern
from ..workflows import (
    ENUMERATION_METHODS,
    EnumerationResult,
    apply_coordination_labels,
    enumerate_candidates,
    generate_assignment,
)


_GUI_INSTALL_HINT = 'python -m pip install -e ".[gui]"'
_METHODS = [
    "layer",
    "neighbor-bipartite",
    "alternating-index",
    "random",
    "checkerboard",
    "graph-coloring",
    "propagation-vector",
    "by-species",
    "by-coordination",
    "manual-spins",
]
_BATCH_METHODS = list(ENUMERATION_METHODS)
_CANDIDATE_COLUMNS = (
    "config_id",
    "method",
    "n_up",
    "n_down",
    "net_spin",
    "afm_score",
    "file",
)
_RESULT_COLUMNS = (
    "config_id",
    "total_energy",
    "final_net_spin",
    "sign_retention",
    "collapsed_atoms",
    "spin_population_source",
    "scf_converged",
    "geometry_converged",
    "status",
)
_NEAR_GROUND_ENERGY_WINDOW_EV = 0.01
_COLOR_MODES = {"spin sign": "sign", "spin value": "value"}
_LEFT_PANEL_MIN_WIDTH = 600
_RESULTS_NOTEBOOK_MIN_HEIGHT = 250
_PANE_SASH_MARGIN = 8
_AUTO_SHOW_INDICES_MAX_ATOMS = 60
_AUTO_SHOW_BONDS_MAX_ATOMS = 60
_MANUAL_SPINS_MAX_ATOMS = 60


@dataclass(frozen=True, slots=True)
class GenerationParams:
    """Widget-independent parameters for one AFM generation run."""

    structure_path: str | Path
    magnetic_species: tuple[str, ...]
    method: str = "layer"
    moment: str | None = None
    site_moment_file: str | Path | None = None
    site_comments: bool = True
    axis: str = "z"
    layer_direction: tuple[float, float, float] | None = None
    fractional_layers: bool = False
    layer_per_species: bool = False
    cutoff: str | float = "auto"
    layer_tolerance: float = 0.25
    slab: bool = False
    q_vector: tuple[float, float, float] | None = None
    afm_type: str | None = None
    allow_frustrated: bool = False
    up_species: tuple[str, ...] = ()
    down_species: tuple[str, ...] = ()
    anion_species: tuple[str, ...] = ()
    anion_cutoff: str | float = "auto"
    up_coordination: tuple[int, ...] = (6,)
    down_coordination: tuple[int, ...] = (4,)
    coordination_tolerance: int = 0
    coordination_labels: tuple[tuple[str, int, str], ...] = ()
    max_colors: int = 4
    color_spins: str | None = None
    balance_colors: bool = False
    spin_values: Mapping[int, float] | None = None
    spin_values_file: str | Path | None = None
    fill_unspecified_zero: bool = False
    seed: int = 0
    color_mode: str = "sign"
    spin_mode: str = "collinear"


@dataclass(slots=True)
class GenerationResult:
    """Controller result consumed by both the desktop UI and tests."""

    structure: Structure
    magnetic_indices: list[int]
    assignment: SpinAssignment
    spins: dict[int, float]
    block: str
    report: dict[str, object]
    warnings: tuple[str, ...]


@dataclass(slots=True)
class SpinFileResult:
    """Validated data loaded from an existing DM.InitSpin file."""

    structure: Structure
    spins: dict[int, float]
    block: str
    validation: ValidationReport
    warnings: tuple[str, ...]
    angles: dict[int, tuple[float, float]] | None = None


@dataclass(slots=True)
class MagnetizationRow:
    """One editable row in the structure-derived magnetization table."""

    use: bool
    element: str
    label: str
    coordination: int | None
    value: str
    count: int
    role: str = "-"
    atom_indices: tuple[int, ...] = ()


@dataclass(slots=True)
class ManualSpinRow:
    """One atom row in the small-system direct-spin editor."""

    atom_index: int
    element: str
    spin: str = ""


@dataclass(frozen=True, slots=True)
class ResultTableRow:
    """One sorted result row plus visual tags for a Tk Treeview."""

    values: tuple[str, ...]
    tags: tuple[str, ...]


def angles_from_result(
    result: GenerationResult | SpinFileResult | SpinAssignment,
) -> dict[int, tuple[float, float]] | None:
    """Map graph-coloring metadata to in-plane non-collinear directions."""

    if isinstance(result, SpinFileResult):
        return result.angles
    assignment = result.assignment if isinstance(result, GenerationResult) else result
    if assignment.metadata.get("spin_mode") != "non-collinear":
        return None
    colors = assignment.metadata.get("colors")
    n_colors = assignment.metadata.get("n_colors")
    if not isinstance(colors, Mapping) or not isinstance(n_colors, int) or n_colors < 1:
        raise ValueError(
            "non-collinear graph-coloring result is missing colors/n_colors metadata"
        )
    try:
        return {
            int(index): (90.0, 360.0 * int(color) / n_colors)
            for index, color in colors.items()
        }
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "non-collinear graph-coloring result has invalid color metadata"
        ) from exc


def _split_words(text: str) -> tuple[str, ...]:
    return tuple(value for value in text.replace(",", " ").split() if value)


def _three_numbers(text: str, label: str) -> tuple[float, float, float]:
    values = [float(value) for value in text.replace(",", " ").split()]
    if len(values) != 3:
        raise ValueError(f"{label} must contain exactly three numbers")
    return values[0], values[1], values[2]


def parse_spin_values(values: Sequence[str] | str | None) -> dict[int, float]:
    """Parse signed ``index=value`` specifications with one-based indices."""

    if values is None:
        raise ValueError("--spin-values requires at least one index=value")
    source = [values] if isinstance(values, str) else list(values)
    items: list[str] = []
    for item in source:
        items.extend(part for part in str(item).replace(",", " ").split() if part)
    if not items:
        raise ValueError("--spin-values requires at least one index=value")

    result: dict[int, float] = {}
    for item in items:
        if item.count("=") != 1:
            raise ValueError(
                f"invalid spin specification {item!r}; expected index=value"
            )
        index_text, value_text = item.split("=", 1)
        try:
            atom_index = int(index_text)
        except ValueError as exc:
            raise ValueError(
                f"manual spin atom index must be a one-based integer: {index_text}"
            ) from exc
        if atom_index <= 0:
            raise ValueError("manual spin atom indices are one-based positive integers")
        if atom_index in result:
            raise ValueError(f"duplicate manual spin atom index: {atom_index}")
        try:
            spin = float(value_text)
        except ValueError as exc:
            raise ValueError(
                f"manual spin for atom {atom_index} must be a number: {value_text}"
            ) from exc
        if not isfinite(spin):
            raise ValueError(
                f"manual spin for atom {atom_index} must be a finite number"
            )
        result[atom_index] = spin
    return result


def load_spin_values_file(path: str | Path) -> dict[int, float]:
    """Read signed direct moments from an ``atom_index,spin`` CSV file."""

    result: dict[int, float] = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"atom_index", "spin"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("spin values CSV requires atom_index and spin columns")
        for row in reader:
            parsed = parse_spin_values([f"{row['atom_index']}={row['spin']}"])
            atom_index, spin = next(iter(parsed.items()))
            if atom_index in result:
                raise ValueError(f"duplicate manual spin atom index: {atom_index}")
            result[atom_index] = spin
    return result


def spin_values_from_inputs(
    values: Mapping[int, float] | Sequence[str] | str | None,
    file_path: str | Path | None,
) -> dict[int, float] | None:
    """Resolve the two mutually exclusive direct-spin input forms."""

    if values is not None and file_path is not None:
        raise ValueError("--spin-values and --spin-values-file are mutually exclusive")
    if values is not None:
        if isinstance(values, Mapping):
            if not values:
                return {}
            return parse_spin_values(
                [f"{atom_index}={spin}" for atom_index, spin in values.items()]
            )
        return parse_spin_values(values)
    if file_path is not None:
        return load_spin_values_file(file_path)
    return None


def manual_spin_rows_from_structure(
    structure: Structure,
    magnetic_species: Sequence[str],
    element_spin_defaults: Mapping[str, str] | None = None,
    *,
    existing_rows: Sequence[ManualSpinRow] = (),
) -> list[ManualSpinRow]:
    """Build atom-order rows while preserving edits for selected species."""

    selected = {item.strip().lower() for item in magnetic_species if item.strip()}
    defaults = {
        str(element).lower(): str(value)
        for element, value in (element_spin_defaults or {}).items()
    }
    previous = {
        (row.atom_index, row.element.lower()): row.spin for row in existing_rows
    }
    rows: list[ManualSpinRow] = []
    for atom_index, element in enumerate(structure.symbols, start=1):
        normalized = element.lower()
        spin = ""
        if normalized in selected:
            spin = previous.get(
                (atom_index, normalized), defaults.get(normalized, "")
            )
        rows.append(ManualSpinRow(atom_index, element, spin))
    return rows


def spin_values_from_rows(rows: Sequence[ManualSpinRow]) -> dict[int, float]:
    """Parse every nonblank direct-spin table cell."""

    values = [f"{row.atom_index}={row.spin}" for row in rows if row.spin.strip()]
    return parse_spin_values(values) if values else {}


def _element_counts(structure: Structure) -> dict[str, int]:
    counts: dict[str, int] = {}
    for symbol in structure.symbols:
        counts[symbol] = counts.get(symbol, 0) + 1
    return counts


def magnetic_species_from_rows(rows: Sequence[MagnetizationRow]) -> tuple[str, ...]:
    """Return checked elements once each, preserving table order."""

    result: list[str] = []
    seen: set[str] = set()
    for row in rows:
        normalized = row.element.lower()
        if row.use and normalized not in seen:
            result.append(row.element)
            seen.add(normalized)
    return tuple(result)


def toggle_magnetization_use(
    rows: Sequence[MagnetizationRow], row_index: int, method: str
) -> MagnetizationRow:
    """Toggle one table selection, independently per CN when applicable."""

    row = rows[row_index]
    new_value = not row.use
    if method == "by-coordination" and row.coordination is not None:
        row.use = new_value
        return row
    for candidate in rows:
        if candidate.element.lower() == row.element.lower():
            candidate.use = new_value
            if not new_value:
                candidate.label = ""
                candidate.value = ""
                candidate.role = "-"
    return row


def moment_text_from_rows(
    rows: Sequence[MagnetizationRow], method: str
) -> str | None:
    """Derive the existing CLI moment syntax from checked table rows."""

    values: list[str] = []
    seen: set[tuple[str, int | None]] = set()
    for row in rows:
        value = row.value.strip()
        if not row.use or not value:
            continue
        coordination = row.coordination if method == "by-coordination" else None
        key = (row.element.lower(), coordination)
        if key in seen:
            continue
        seen.add(key)
        name = (
            f"{row.element}@{coordination}"
            if coordination is not None
            else row.element
        )
        values.append(f"{name}={value}")
    return " ".join(values) or None


def batch_moment_text_from_rows(
    rows: Sequence[MagnetizationRow],
    methods: Sequence[str],
    *,
    moment_sweep: Sequence[str] | str | None = None,
) -> str | None:
    """Return moment specifications usable by every selected batch method.

    Coordination rows can carry different values for the same element.  A mixed
    batch therefore needs both an element fallback for ordinary methods and the
    more specific ``Element@CN`` values used by ``by-coordination``.
    """

    sweep_items = (
        [moment_sweep]
        if isinstance(moment_sweep, str)
        else list(moment_sweep or ())
    )
    swept_elements: set[str] = set()
    global_sweep = False
    for sweep_item in sweep_items:
        for specification in str(sweep_item).split():
            if "=" not in specification:
                global_sweep = True
                continue
            target = specification.split("=", 1)[0].split("@", 1)[0].strip()
            if target:
                swept_elements.add(target.lower())
    baseline_rows = [
        row
        for row in rows
        if not global_sweep and row.element.lower() not in swept_elements
    ]

    element_moments = moment_text_from_rows(baseline_rows, "layer")
    if "by-coordination" not in methods:
        return element_moments
    coordination_moments = moment_text_from_rows(
        baseline_rows, "by-coordination"
    )
    if all(method == "by-coordination" for method in methods):
        return coordination_moments
    if not coordination_moments or coordination_moments == element_moments:
        return element_moments
    return " ".join(
        value for value in (element_moments, coordination_moments) if value
    )


def species_roles_from_rows(
    rows: Sequence[MagnetizationRow],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return checked by-species elements grouped by their editable role."""

    up: list[str] = []
    down: list[str] = []
    for element in magnetic_species_from_rows(rows):
        role = next(
            (
                row.role.lower()
                for row in rows
                if row.use and row.element.lower() == element.lower()
            ),
            "-",
        )
        if role == "up":
            up.append(element)
        elif role == "down":
            down.append(element)
    return tuple(up), tuple(down)


def coordination_labels_from_rows(
    rows: Sequence[MagnetizationRow],
) -> tuple[tuple[str, int, str], ...]:
    """Return editable coordination labels for checked site rows."""

    return tuple(
        (row.element, row.coordination, row.label)
        for row in rows
        if row.use
        and row.coordination is not None
        and row.label not in {"", "-"}
    )


def workflow_kwargs_from_inputs(
    methods: Sequence[str],
    magnetization_rows: Sequence[MagnetizationRow],
    *,
    site_moment_file: str = "",
    axis: str = "z",
    layer_direction: str = "",
    layer_tolerance: str | float = 0.25,
    fractional_layers: bool = False,
    layer_per_species: bool = False,
    auto_cutoff: bool = True,
    cutoff: str | float = 3.2,
    allow_frustrated: bool = False,
    q_vector: str = "",
    afm_type: str = "custom",
    anion_species: str = "",
    anion_cutoff: str = "auto",
    up_coordination: str = "6",
    down_coordination: str = "4",
    coordination_tolerance: str | int = 0,
    max_colors: str | int = 4,
    color_spins: str = "",
    balance_colors: bool = False,
    group_file: str = "",
    seed_offset: str | int = 0,
    symmetry_dedup: bool = False,
    symprec: str | float = 1e-3,
) -> dict[str, object]:
    """Collect workflow options from widget-independent input values."""

    selected_methods = set(methods)
    parsed_cutoff: str | float = "auto" if auto_cutoff else float(cutoff)
    parsed_layer_direction = None
    direction_text = layer_direction.strip()
    if "layer" in selected_methods and direction_text:
        parsed_layer_direction = _three_numbers(direction_text, "layer direction")

    parsed_q_vector = None
    selected_afm_type = None
    if "propagation-vector" in selected_methods:
        if afm_type == "custom":
            parsed_q_vector = _three_numbers(q_vector, "q-vector")
        else:
            selected_afm_type = afm_type

    up_species, down_species = species_roles_from_rows(magnetization_rows)
    if "by-coordination" in selected_methods:
        parsed_up_coordination = tuple(
            int(value) for value in _split_words(up_coordination)
        )
        parsed_down_coordination = tuple(
            int(value) for value in _split_words(down_coordination)
        )
        parsed_coordination_tolerance = int(coordination_tolerance)
    else:
        parsed_up_coordination = (6,)
        parsed_down_coordination = (4,)
        parsed_coordination_tolerance = 0

    if "graph-coloring" in selected_methods:
        parsed_max_colors = int(max_colors)
        parsed_color_spins = color_spins.strip() or None
        parsed_balance_colors = balance_colors
    else:
        parsed_max_colors = 4
        parsed_color_spins = None
        parsed_balance_colors = False

    return {
        "site_moment_file": site_moment_file.strip() or None,
        "axis": axis,
        "layer_direction": parsed_layer_direction,
        "layer_tolerance": float(layer_tolerance),
        "fractional_layers": fractional_layers,
        "layer_per_species": layer_per_species and "layer" in selected_methods,
        "cutoff": parsed_cutoff,
        "neighbor_shell": 1,
        "allow_frustrated": allow_frustrated,
        "q_vector": parsed_q_vector,
        "afm_type": selected_afm_type,
        "up_species": up_species,
        "down_species": down_species,
        "anion_species": _split_words(anion_species),
        "anion_cutoff": anion_cutoff.strip() or "auto",
        "up_coordination": parsed_up_coordination,
        "down_coordination": parsed_down_coordination,
        "coordination_tolerance": parsed_coordination_tolerance,
        "max_colors": parsed_max_colors,
        "color_spins": parsed_color_spins,
        "balance_colors": parsed_balance_colors,
        "group_file": (
            group_file.strip() if "manual-groups" in selected_methods else None
        ),
        "seed_offset": int(seed_offset),
        "symmetry_dedup": symmetry_dedup,
        "symprec": float(symprec),
    }


def equivalent_cli_options(rows: Sequence[MagnetizationRow], method: str) -> str:
    """Render the table's equivalent reusable CLI options."""

    species = magnetic_species_from_rows(rows)
    parts = ["--magnetic-species", *species]
    moment = moment_text_from_rows(rows, method)
    if moment:
        parts.extend(("--moment", moment))
    if method == "by-species":
        up, down = species_roles_from_rows(rows)
        if up:
            parts.extend(("--up-species", *up))
        if down:
            parts.extend(("--down-species", *down))
    return " ".join(parts)


def magnetization_rows_from_structure(
    structure: Structure,
    method: str,
    *,
    existing_rows: Sequence[MagnetizationRow] = (),
    anion_species: Sequence[str] | None = None,
    anion_cutoff: str | float | None = "auto",
) -> list[MagnetizationRow]:
    """Build editable rows from a structure, preserving prior values where possible."""

    counts = _element_counts(structure)
    defaults = {
        element.lower(): value for element, value in DEFAULT_ELEMENT_MOMENTS.items()
    }
    if existing_rows:
        use_by_element = {
            element.lower(): any(
                row.use for row in existing_rows if row.element.lower() == element.lower()
            )
            for element in counts
        }
    else:
        use_by_element = {element.lower(): element.lower() in defaults for element in counts}
    use_by_site = {
        (row.element.lower(), row.coordination): row.use
        for row in existing_rows
        if row.coordination is not None
    }
    values_by_site = {
        (row.element.lower(), row.coordination): row.value
        for row in existing_rows
        if row.value.strip()
    }
    values_by_element: dict[str, str] = {}
    roles_by_element: dict[str, str] = {}
    labels_by_site: dict[tuple[str, int], str] = {}
    for row in existing_rows:
        normalized = row.element.lower()
        if row.value.strip():
            values_by_element.setdefault(normalized, row.value)
        if row.role in {"up", "down"}:
            roles_by_element.setdefault(normalized, row.role)
        if row.coordination is not None and row.label not in {"", "-"}:
            labels_by_site[(normalized, row.coordination)] = row.label

    if method != "by-coordination":
        used_order = [
            element for element in counts if use_by_element.get(element.lower(), False)
        ]
        rows: list[MagnetizationRow] = []
        for element, count in counts.items():
            normalized = element.lower()
            use = use_by_element.get(normalized, False)
            value = values_by_element.get(normalized)
            if value is None and use and normalized in defaults:
                value = f"{defaults[normalized]:.1f}"
            role = "-"
            if method == "by-species" and use:
                role = roles_by_element.get(
                    normalized, "up" if element == used_order[0] else "down"
                )
            rows.append(
                MagnetizationRow(
                    use=use,
                    element=element,
                    label="-" if use else "",
                    coordination=None,
                    value=(value or "") if use else "",
                    count=count,
                    role=role,
                    atom_indices=tuple(
                        index + 1
                        for index, symbol in enumerate(structure.symbols)
                        if symbol.lower() == normalized
                    ),
                )
            )
        return rows

    # Keep coordination rows available even when every site row for an element
    # is currently unchecked. Otherwise a refresh would discard the rows and
    # make it impossible to re-enable one CN group independently.
    analyzed_elements = {
        row.element.lower()
        for row in existing_rows
        if row.coordination is not None
    }
    selected = [
        index
        for index, symbol in enumerate(structure.symbols)
        if use_by_element.get(symbol.lower(), False)
        or symbol.lower() in analyzed_elements
    ]
    if not selected:
        return [
            MagnetizationRow(False, element, "", None, "", count, "-")
            for element, count in counts.items()
        ]
    analysis = analyze_coordination_sites(
        structure,
        selected,
        anion_species=anion_species,
        anion_cutoff=anion_cutoff,
    )
    selected_species = [
        element
        for element in counts
        if use_by_element.get(element.lower(), False)
        or element.lower() in analyzed_elements
    ]
    combination_counts = _coordination_combination_counts(
        structure,
        selected,
        analysis.coordination_numbers,
        selected_species,
    )
    rows = []
    for (element, coordination), count in combination_counts.items():
        indices = [
            index
            for index in selected
            if structure.symbols[index].lower() == element.lower()
            and analysis.coordination_numbers[index] == coordination
        ]
        normalized = element.lower()
        labels = {
            classify_coordination_geometry(analysis.ligand_vectors[index])
            for index in indices
        }
        inferred_label = next(iter(labels)) if len(labels) == 1 else "mixed"
        value = values_by_site.get((normalized, coordination))
        if value is None:
            value = values_by_element.get(normalized)
        if value is None and normalized in defaults:
            value = f"{defaults[normalized]:.1f}"
        rows.append(
            MagnetizationRow(
                use_by_site.get((normalized, coordination), True),
                element,
                labels_by_site.get((normalized, coordination), inferred_label),
                coordination,
                value or "",
                count,
                "-",
                tuple(index + 1 for index in indices),
            )
        )
    for element, count in counts.items():
        if element.lower() not in {item.lower() for item in selected_species}:
            rows.append(MagnetizationRow(False, element, "", None, "", count, "-"))
    return rows


def safe_magnetization_rows_from_structure(
    structure: Structure,
    method: str,
    *,
    existing_rows: Sequence[MagnetizationRow] = (),
    anion_species: Sequence[str] | None = None,
    anion_cutoff: str | float | None = "auto",
) -> tuple[list[MagnetizationRow], str | None]:
    """Build table rows and fall back safely when coordination analysis fails."""

    try:
        return (
            magnetization_rows_from_structure(
                structure,
                method,
                existing_rows=existing_rows,
                anion_species=anion_species,
                anion_cutoff=anion_cutoff,
            ),
            None,
        )
    except ValueError as exc:
        if method != "by-coordination":
            raise
        rows = magnetization_rows_from_structure(
            structure,
            "layer",
            existing_rows=existing_rows,
        )
        return rows, f"Coordination analysis unavailable; using element rows: {exc}"


def _coordination_combination_counts(
    structure: Structure,
    indices: Sequence[int],
    coordinations: Mapping[int, object],
    magnetic_species: Sequence[str],
) -> dict[tuple[str, int], int]:
    raw_counts: dict[tuple[str, int], int] = {}
    for index in indices:
        key = (structure.symbols[index], int(coordinations[index]))
        raw_counts[key] = raw_counts.get(key, 0) + 1
    species_order = {
        symbol.lower(): order for order, symbol in enumerate(magnetic_species)
    }
    return dict(
        sorted(
            raw_counts.items(),
            key=lambda item: (
                species_order.get(item[0][0].lower(), len(species_order)),
                item[0][1],
                item[0][0].lower(),
            ),
        )
    )


def detect_coordination_combinations(
    structure: Structure,
    magnetic_species: Sequence[str],
    *,
    anion_species: Sequence[str] | None = None,
    anion_cutoff: str | float | None = "auto",
    up_coordination: Sequence[int] = (6,),
    down_coordination: Sequence[int] = (4,),
    coordination_tolerance: int = 0,
) -> dict[tuple[str, int], int]:
    """Count detected magnetic ``(element, CN)`` site types in display order."""

    indices = select_magnetic_sites(structure, magnetic_species)
    analysis = analyze_coordination_sites(
        structure,
        indices,
        anion_species=anion_species,
        anion_cutoff=anion_cutoff,
    )
    return _coordination_combination_counts(
        structure, indices, analysis.coordination_numbers, magnetic_species
    )


def coordination_numbers_from_result(
    result: GenerationResult | SpinFileResult,
) -> dict[int, int]:
    """Return stored per-atom CN values for a by-coordination result."""

    if (
        not isinstance(result, GenerationResult)
        or result.assignment.method != "by-coordination"
    ):
        return {}
    values = result.assignment.metadata.get("coordination_numbers")
    if not isinstance(values, Mapping):
        return {}
    coordinations: dict[int, int] = {}
    for index, coordination in values.items():
        try:
            coordinations[int(index)] = int(coordination)
        except (TypeError, ValueError):
            continue
    return coordinations


def site_assignment_rows(
    result: GenerationResult | SpinFileResult,
) -> list[dict[str, object]]:
    """Return input-order site rows for the GUI without requiring Tk."""

    assignment = result.assignment if isinstance(result, GenerationResult) else None
    indices = (
        result.magnetic_indices
        if isinstance(result, GenerationResult)
        else sorted(result.spins)
    )
    coordinations = coordination_numbers_from_result(result)
    sublattices = (
        assignment.metadata.get("sublattice_classification", {}) if assignment else {}
    )
    if not isinstance(sublattices, dict):
        sublattices = {}
    rows: list[dict[str, object]] = []
    for index in sorted(indices):
        spin = float(result.spins.get(index, 0.0))
        rows.append(
            {
                "atom": index + 1,
                "element": result.structure.symbols[index],
                "CN": coordinations.get(index, "-"),
                "sublattice": sublattices.get(index, "-"),
                "sign": "+" if spin > 0 else "-" if spin < 0 else "0",
                "moment": abs(spin),
            }
        )
    return rows


def site_assignment_summary(result: GenerationResult | SpinFileResult) -> str:
    values = [float(value) for value in result.spins.values()]
    net_moment = sum(values)
    if abs(net_moment) < 1e-12:
        net_moment = 0.0
    return (
        f"n_up = {sum(value > 0 for value in values)} / "
        f"n_down = {sum(value < 0 for value in values)} / "
        f"n_zero = {sum(value == 0 for value in values)}, "
        f"net moment = {net_moment:g} μB"
    )




def run_generation(params: GenerationParams) -> GenerationResult:
    """Generate and analyze an AFM state without importing or creating Tk."""

    if not params.magnetic_species:
        raise ValueError("at least one magnetic species is required")
    if params.color_mode not in {"sign", "value"}:
        raise ValueError("color mode must be 'sign' or 'value'")
    structure = read_structure(params.structure_path, slab=params.slab)
    spin_values = spin_values_from_inputs(
        params.spin_values,
        params.spin_values_file,
    )
    indices, assignment, spins = generate_assignment(
        structure,
        params.magnetic_species,
        params.method,
        params.moment,
        site_moment_file=(
            str(params.site_moment_file) if params.site_moment_file else None
        ),
        axis=params.axis,
        layer_direction=params.layer_direction,
        layer_tolerance=params.layer_tolerance,
        fractional_layers=params.fractional_layers,
        layer_per_species=params.layer_per_species,
        cutoff=params.cutoff,
        allow_frustrated=params.allow_frustrated,
        q_vector=params.q_vector,
        afm_type=params.afm_type,
        up_species=params.up_species,
        down_species=params.down_species,
        anion_species=params.anion_species or None,
        anion_cutoff=params.anion_cutoff,
        up_coordination=params.up_coordination,
        down_coordination=params.down_coordination,
        coordination_tolerance=params.coordination_tolerance,
        max_colors=params.max_colors,
        color_spins=params.color_spins,
        balance_colors=params.balance_colors,
        spin_values=spin_values,
        fill_unspecified_zero=params.fill_unspecified_zero,
        spin_mode=params.spin_mode,
        seed=params.seed,
    )
    apply_coordination_labels(
        structure, indices, assignment, params.coordination_labels
    )
    block = render_dm_init_spin(
        spins,
        method=assignment.method,
        magnetic_species=params.magnetic_species,
        metadata=assignment.metadata,
        angles=angles_from_result(assignment),
        structure=structure,
        site_comments=params.site_comments,
    )
    report = analyze_structure(
        structure,
        indices,
        magnetic_species=params.magnetic_species,
        cutoff=params.cutoff,
        axis=params.axis,
        layer_tolerance=params.layer_tolerance,
        fractional_layers=params.fractional_layers,
        layer_direction=params.layer_direction,
    )
    return GenerationResult(
        structure=structure,
        magnetic_indices=indices,
        assignment=assignment,
        spins=spins,
        block=block,
        report=report,
        warnings=tuple(assignment.warnings),
    )


def run_candidate_generation(
    structure: Structure,
    magnetic_species: Sequence[str],
    methods: Sequence[str],
    moment: Sequence[str] | str | None,
    n_configs: int,
    output_dir: str | Path,
    *,
    moment_sweep: Sequence[str] | str | None = None,
    keep_global_spin_inversion: bool = False,
    symmetry_dedup: bool = False,
    symprec: float = 1e-3,
    site_comments: bool = True,
    **workflow_kwargs: object,
) -> EnumerationResult:
    """Generate a candidate batch without importing or creating Tk."""

    if not magnetic_species:
        raise ValueError("select at least one magnetic element in the table")
    return enumerate_candidates(
        structure,
        magnetic_species,
        methods,
        moment,
        n_configs,
        output_dir,
        moment_sweep=moment_sweep,
        keep_global_spin_inversion=keep_global_spin_inversion,
        symmetry_dedup=symmetry_dedup,
        symprec=symprec,
        site_comments=site_comments,
        **workflow_kwargs,
    )


def candidate_table_rows(
    result: EnumerationResult,
) -> list[tuple[str, ...]]:
    """Convert an enumeration result to deterministic Treeview values."""

    return [
        tuple(_display_value(row.get(column, "")) for column in _CANDIDATE_COLUMNS)
        for row in result.manifest
    ]


def prepare_job_folders(
    base_input: str | Path,
    candidates_dir: str | Path,
    output_dir: str | Path,
) -> list[Path]:
    """Prepare job folders through the same pure function used by the CLI."""

    return prepare_array(base_input, candidates_dir, output_dir)


def load_results_csv(path: str | Path) -> list[dict[str, object]]:
    """Load an existing collection without rerunning result extraction."""

    source = Path(path)
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        required = {"config_id", "total_energy", "scf_converged"}
        missing = sorted(required - fields)
        if missing:
            raise ValueError(
                "results CSV is missing required column(s): " + ", ".join(missing)
            )
        return [dict(row) for row in reader]


def collect_or_load_results(
    *,
    jobs_dir: str | Path | None = None,
    results_csv: str | Path | None = None,
) -> list[dict[str, object]]:
    """Collect a jobs directory, or load a previously collected CSV."""

    if results_csv:
        return load_results_csv(results_csv)
    if not jobs_dir:
        raise ValueError("select a jobs directory or an existing results.csv")
    return collect_results(jobs_dir)


def _display_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _energy_value(row: Mapping[str, object]) -> float | None:
    value = row.get("total_energy")
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(str(value))
    except ValueError:
        return None


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def results_table_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    energy_window_ev: float = _NEAR_GROUND_ENERGY_WINDOW_EV,
) -> list[ResultTableRow]:
    """Sort by energy and tag converged candidates near the lowest energy.

    The function deliberately marks a group only; it never chooses a winner.
    """

    if energy_window_ev < 0:
        raise ValueError("energy window must be nonnegative")
    ordered = sorted(
        rows,
        key=lambda row: (
            _energy_value(row) is None,
            _energy_value(row) if _energy_value(row) is not None else float("inf"),
            str(row.get("config_id", "")),
        ),
    )
    converged_energies = [
        energy
        for row in ordered
        if _truthy(row.get("scf_converged"))
        for energy in [_energy_value(row)]
        if energy is not None
    ]
    minimum = min(converged_energies) if converged_energies else None
    table_rows: list[ResultTableRow] = []
    for row in ordered:
        energy = _energy_value(row)
        converged = _truthy(row.get("scf_converged"))
        tags: list[str] = []
        if not converged:
            tags.append("unconverged")
        if (
            converged
            and energy is not None
            and minimum is not None
            and energy - minimum <= energy_window_ev
        ):
            tags.append("near_ground")
        table_rows.append(
            ResultTableRow(
                tuple(_display_value(row.get(column, "")) for column in _RESULT_COLUMNS),
                tuple(tags),
            )
        )
    return table_rows


def load_spin_file(
    spin_path: str | Path,
    structure: Structure,
    *,
    cutoff: str | float = "auto",
    axis: str = "z",
    layer_tolerance: float = 0.25,
    site_comments: bool = True,
) -> SpinFileResult:
    """Load and validate an existing spin file against a structure."""

    parse_warnings: list[str] = []
    rows = parse_dm_init_spin(spin_path, warnings=parse_warnings)
    angle_rows = parse_dm_init_spin(spin_path, include_angles=True)
    angles = (
        {
            index - 1: (theta, phi)
            for index, _moment, theta, phi in angle_rows
        }
        if parse_warnings
        else None
    )
    validation = validate_spins(
        rows,
        structure=structure,
        cutoff=cutoff,
        axis=axis,
        layer_tolerance=layer_tolerance,
    )
    if not validation.valid:
        raise ValueError("invalid spin file:\n" + "\n".join(validation.errors))
    spins = {index - 1: value for index, value in rows}
    species = sorted(
        {
            structure.symbols[index]
            for index, value in spins.items()
            if 0 <= index < len(structure) and value != 0
        }
    )
    block = render_dm_init_spin(
        spins,
        method="loaded-spin-file",
        magnetic_species=species,
        angles=angles,
        structure=structure,
        site_comments=site_comments,
    )
    return SpinFileResult(
        structure=structure,
        spins=spins,
        angles=angles,
        block=block,
        validation=validation,
        warnings=tuple(parse_warnings),
    )


def export_spin_block(block: str, destination: str | Path) -> Path:
    """Write the current generated DM.InitSpin document."""

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(block, encoding="utf-8")
    return path


def export_patched_input(
    base_input: str | Path,
    block: str,
    destination: str | Path,
) -> Path:
    """Patch a copy of an FDF input while guaranteeing the base is untouched."""

    source = Path(base_input)
    output = Path(destination)
    if output.resolve() == source.resolve():
        raise ValueError("patched export must not overwrite the base input file")
    patched = patch_fdf_text(source.read_text(encoding="utf-8-sig"), block)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(patched, encoding="utf-8")
    return output


def export_structure_with_moments(
    structure: Structure,
    spins: dict[int, float],
    destination: str | Path,
) -> Path:
    """Export XYZ/CIF through the existing magmom-preserving core path."""

    return plot_spin_pattern(structure, spins, destination)


def complete_input_document(
    result: GenerationResult | SpinFileResult,
    *,
    split_species_by_coordination: bool = False,
    **options: object,
) -> InputTemplateResult:
    """Build the same complete starting input used by the CLI."""

    if isinstance(result, GenerationResult):
        indices = result.magnetic_indices
        method = result.assignment.method
        metadata = result.assignment.metadata
    else:
        indices = sorted(index for index, value in result.spins.items() if value != 0)
        method = "loaded-spin-file"
        metadata = {}
    magnetic_species: list[str] = []
    for index in indices:
        symbol = result.structure.symbols[index]
        if symbol not in magnetic_species:
            magnetic_species.append(symbol)
    return render_complete_input(
        result.structure,
        result.spins,
        method=method,
        magnetic_species=magnetic_species,
        metadata=metadata,
        angles=angles_from_result(result),
        split_species_by_coordination=split_species_by_coordination,
        **options,
    )


def export_complete_input(
    result: GenerationResult | SpinFileResult,
    destination: str | Path,
    *,
    split_species_by_coordination: bool = False,
    **options: object,
) -> Path:
    """Write a complete SIESTA starting input through the shared renderer."""

    document = complete_input_document(
        result,
        split_species_by_coordination=split_species_by_coordination,
        **options,
    )
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document.text, encoding="utf-8")
    return path
