"""Tkinter desktop interface and testable GUI workflow controllers."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .fdf_writer import patch_fdf_text, render_dm_init_spin
from .input_template import InputTemplateResult, render_complete_input
from .io import parse_dm_init_spin, read_structure
from .magnetic_sites import (
    DEFAULT_ELEMENT_MOMENTS,
    select_magnetic_sites,
)
from .neighbors import classify_coordination_geometry
from .ordering import SpinAssignment, analyze_coordination_sites
from .results import collect_results, prepare_array
from .structure import Structure
from .validation import (
    ValidationReport,
    analyze_structure,
    format_analysis,
    format_validation,
    validate_spins,
)
from .visualize import create_spin_figure, plot_spin_pattern
from .workflows import (
    ENUMERATION_METHODS,
    EnumerationResult,
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
    seed: int = 0
    color_mode: str = "sign"


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


@dataclass(frozen=True, slots=True)
class ResultTableRow:
    """One sorted result row plus visual tags for a Tk Treeview."""

    values: tuple[str, ...]
    tags: tuple[str, ...]


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
    rows: Sequence[MagnetizationRow], methods: Sequence[str]
) -> str | None:
    """Return moment specifications usable by every selected batch method.

    Coordination rows can carry different values for the same element.  A mixed
    batch therefore needs both an element fallback for ordinary methods and the
    more specific ``Element@CN`` values used by ``by-coordination``.
    """

    element_moments = moment_text_from_rows(rows, "layer")
    if "by-coordination" not in methods:
        return element_moments
    coordination_moments = moment_text_from_rows(rows, "by-coordination")
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
    coordinations = (
        assignment.metadata.get("coordination_numbers", {}) if assignment else {}
    )
    sublattices = (
        assignment.metadata.get("sublattice_classification", {}) if assignment else {}
    )
    if not isinstance(coordinations, dict):
        coordinations = {}
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


@dataclass(frozen=True, slots=True)
class _CameraState:
    elev: float
    azim: float
    xlim: tuple[float, float]
    ylim: tuple[float, float]
    zlim: tuple[float, float]


@dataclass(frozen=True, slots=True)
class _GuiDependencies:
    tk: Any
    ttk: Any
    filedialog: Any
    messagebox: Any
    FigureCanvasTkAgg: Any
    NavigationToolbar2Tk: Any


def run_generation(params: GenerationParams) -> GenerationResult:
    """Generate and analyze an AFM state without importing or creating Tk."""

    if not params.magnetic_species:
        raise ValueError("at least one magnetic species is required")
    if params.color_mode not in {"sign", "value"}:
        raise ValueError("color mode must be 'sign' or 'value'")
    structure = read_structure(params.structure_path, slab=params.slab)
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
        seed=params.seed,
    )
    if params.method == "by-coordination" and params.coordination_labels:
        coordinations = assignment.metadata.get("coordination_numbers")
        if isinstance(coordinations, dict):
            labels = {
                (element.lower(), coordination): label
                for element, coordination, label in params.coordination_labels
                if label
            }
            geometry = dict(assignment.metadata.get("coordination_geometry", {}))
            for index in indices:
                key = (structure.symbols[index].lower(), int(coordinations[index]))
                if key in labels:
                    geometry[index] = labels[key]
            assignment.metadata["coordination_geometry"] = geometry
    block = render_dm_init_spin(
        spins,
        method=assignment.method,
        magnetic_species=params.magnetic_species,
        metadata=assignment.metadata,
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
    keep_global_spin_inversion: bool = False,
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
        keep_global_spin_inversion=keep_global_spin_inversion,
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
        structure=structure,
        site_comments=site_comments,
    )
    return SpinFileResult(
        structure=structure,
        spins=spins,
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
        **options,
    )


def export_complete_input(
    result: GenerationResult | SpinFileResult,
    destination: str | Path,
    **options: object,
) -> Path:
    """Write a complete SIESTA starting input through the shared renderer."""

    document = complete_input_document(result, **options)
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document.text, encoding="utf-8")
    return path


def _load_gui_dependencies() -> _GuiDependencies:
    """Import Tk and its matplotlib bridge only when the desktop UI is launched."""

    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except ImportError as exc:
        raise RuntimeError(
            "the desktop GUI requires Tkinter and matplotlib; install Python with "
            f"Tk support, then run '{_GUI_INSTALL_HINT}'"
        ) from exc
    FigureCanvasTkAgg, NavigationToolbar2Tk = _load_matplotlib_tk_backend()
    return _GuiDependencies(
        tk=tk,
        ttk=ttk,
        filedialog=filedialog,
        messagebox=messagebox,
        FigureCanvasTkAgg=FigureCanvasTkAgg,
        NavigationToolbar2Tk=NavigationToolbar2Tk,
    )


def _load_matplotlib_tk_backend() -> tuple[Any, Any]:
    try:
        from matplotlib.backends.backend_tkagg import (
            FigureCanvasTkAgg,
            NavigationToolbar2Tk,
        )
    except ImportError as exc:
        raise RuntimeError(
            "the GUI requires optional dependencies; install them with "
            f"'{_GUI_INSTALL_HINT}'"
        ) from exc
    return FigureCanvasTkAgg, NavigationToolbar2Tk


class DesktopApp:
    """Tkinter desktop shell; scientific work remains in controller functions."""

    def __init__(self, root: Any, dependencies: _GuiDependencies) -> None:
        self.root = root
        self.deps = dependencies
        self.tk = dependencies.tk
        self.ttk = dependencies.ttk
        self.structure_path: Path | None = None
        self.current_structure: Structure | None = None
        self.current_spins: dict[int, float] = {}
        self.current_block = ""
        self.current_result: GenerationResult | SpinFileResult | None = None
        self.viewing_spin_path: Path | None = None
        self.figure: Any | None = None
        self.canvas: Any | None = None
        self.toolbar: Any | None = None
        self._live_after_id: str | None = None
        self._table_after_id: str | None = None
        self._reset_camera = True
        self._traces_ready = False
        self.magnetization_rows: list[MagnetizationRow] = []
        self._cell_editor: Any | None = None
        self.control_inputs: list[Any] = []
        self.method_option_frames: dict[str, Any] = {}
        self._coordination_fallback: str | None = None
        self._coordination_use_note: str | None = None
        self._pane_width_initialized = False
        self._results_pane_height_initialized = False
        self._active_scroll_canvas: Any | None = None
        self._mousewheel_leave_after_id: str | None = None

        root.title("SIESTA AFM initial-spin generator")
        root.geometry("1450x900")
        root.minsize(1050, 700)
        root.protocol("WM_DELETE_WINDOW", self._close)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        self._create_variables()
        self._build_controls()
        self._build_results()
        self._install_traces()
        self._traces_ready = True
        self._sync_cutoff_state()
        self._show_method_options()
        self._set_initial_pane_width()
        self._set_initial_results_pane_height()

    def _create_variables(self) -> None:
        tk = self.tk
        self.file_var = tk.StringVar(value="No structure selected")
        self.method_var = tk.StringVar(value="layer")
        self.site_moment_file_var = tk.StringVar(value="")
        self.site_comments_var = tk.BooleanVar(value=True)
        self.cli_options_var = tk.StringVar(value="--magnetic-species")
        self.axis_var = tk.StringVar(value="z")
        self.layer_direction_var = tk.StringVar(value="")
        self.fractional_layers_var = tk.BooleanVar(value=False)
        self.auto_cutoff_var = tk.BooleanVar(value=True)
        self.cutoff_var = tk.StringVar(value="3.2")
        self.tolerance_var = tk.StringVar(value="0.25")
        self.slab_var = tk.BooleanVar(value=False)
        self.q_vector_var = tk.StringVar(value="0.5 0.5 0.5")
        self.afm_type_var = tk.StringVar(value="custom")
        self.allow_frustrated_var = tk.BooleanVar(value=False)
        self.anion_species_var = tk.StringVar(value="")
        self.anion_cutoff_var = tk.StringVar(value="auto")
        self.up_coordination_var = tk.StringVar(value="6")
        self.down_coordination_var = tk.StringVar(value="4")
        self.coordination_tolerance_var = tk.StringVar(value="0")
        self.max_colors_var = tk.StringVar(value="4")
        self.color_spins_var = tk.StringVar(value="")
        self.balance_colors_var = tk.BooleanVar(value=False)
        self.seed_var = tk.StringVar(value="0")
        self.color_mode_var = tk.StringVar(value="spin sign")
        self.show_atom_indices_var = tk.BooleanVar(value=True)
        self.live_update_var = tk.BooleanVar(value=True)
        self.mode_var = tk.StringVar(value="generation mode")
        self.status_var = tk.StringVar(value="Select a structure file to begin.")
        self.batch_method_vars = {
            method: tk.BooleanVar(value=method == "layer")
            for method in _BATCH_METHODS
        }
        self.batch_n_configs_var = tk.StringVar(value="8")
        self.batch_keep_inversion_var = tk.BooleanVar(value=False)
        self.candidate_output_var = tk.StringVar(value="")
        self.batch_group_file_var = tk.StringVar(value="")
        self.base_input_var = tk.StringVar(value="")
        self.candidates_dir_var = tk.StringVar(value="")
        self.jobs_output_var = tk.StringVar(value="")
        self.jobs_dir_var = tk.StringVar(value="")
        self.results_csv_var = tk.StringVar(value="")

    def _build_controls(self) -> None:
        ttk = self.ttk
        self.main_pane = ttk.PanedWindow(self.root, orient="horizontal")
        self.main_pane.grid(row=0, column=0, sticky="nsew")
        self.main_pane.bind(
            "<Map>", lambda _event: self.root.after_idle(self._set_initial_pane_width)
        )
        container = ttk.Frame(self.main_pane)
        self.main_pane.add(container, weight=0)
        self.controls_canvas, panel = self._make_scrollable_frame(
            container, padding=10
        )
        panel.columnconfigure(0, minsize=115, weight=0)
        panel.columnconfigure(1, minsize=160, weight=1)

        ttk.Label(panel, text="Structure").grid(row=0, column=0, sticky="w")
        ttk.Button(
            panel, text="Open structure...", command=self._choose_structure
        ).grid(row=0, column=1, sticky="ew", padx=(6, 0))
        ttk.Label(panel, textvariable=self.file_var, wraplength=300).grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=(2, 10)
        )

        row = 2
        row = self._combo_row(panel, row, "Method", self.method_var, _METHODS)

        table_frame = ttk.LabelFrame(panel, text="Magnetic species and moments", padding=5)
        table_frame.grid(row=row, column=0, columnspan=2, sticky="nsew", pady=(5, 4))
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        columns = ("use", "element", "label", "CN", "value", "count", "role")
        self.magnetization_tree = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings",
            height=7,
            selectmode="browse",
        )
        headings = {
            "use": "use",
            "element": "element",
            "label": "label",
            "CN": "CN",
            "value": "value (μB)",
            "count": "count",
            "role": "role",
        }
        widths = {
            "use": 42,
            "element": 62,
            "label": 125,
            "CN": 42,
            "value": 80,
            "count": 50,
            "role": 58,
        }
        for column in columns:
            self.magnetization_tree.heading(column, text=headings[column])
            self.magnetization_tree.column(
                column,
                width=widths[column],
                minwidth=widths[column],
                stretch=column == "label",
                anchor="center",
            )
        table_y_scroll = ttk.Scrollbar(
            table_frame, orient="vertical", command=self.magnetization_tree.yview
        )
        table_x_scroll = ttk.Scrollbar(
            table_frame, orient="horizontal", command=self.magnetization_tree.xview
        )
        self.magnetization_tree.configure(
            yscrollcommand=table_y_scroll.set,
            xscrollcommand=table_x_scroll.set,
        )
        self.magnetization_tree.grid(row=0, column=0, sticky="nsew")
        table_y_scroll.grid(row=0, column=1, sticky="ns")
        table_x_scroll.grid(row=1, column=0, sticky="ew")
        self.magnetization_tree.bind("<Double-1>", self._edit_magnetization_cell)
        ttk.Label(
            table_frame,
            text=(
                "Double-click use, label, value, or role to edit. CN and count "
                "are read-only. In by-coordination mode, unchecking one "
                "coordination site keeps the element selected; set moment 0 or "
                "use --exclude-atoms to fully exclude those atoms."
            ),
            foreground="#666666",
            wraplength=530,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(3, 0))
        row += 1
        ttk.Label(panel, text="Equivalent CLI").grid(row=row, column=0, sticky="nw")
        ttk.Label(
            panel,
            textvariable=self.cli_options_var,
            wraplength=460,
            justify="left",
        ).grid(row=row, column=1, sticky="ew", padx=(6, 0), pady=3)
        row += 1

        ttk.Checkbutton(
            panel,
            text="Include element/CN comments in DM.InitSpin",
            variable=self.site_comments_var,
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=3)
        row += 1
        ttk.Label(panel, text="Site moment file").grid(row=row, column=0, sticky="w")
        site_file_controls = ttk.Frame(panel)
        site_file_controls.grid(row=row, column=1, sticky="ew", padx=(6, 0), pady=3)
        site_file_controls.columnconfigure(0, weight=1)
        site_file_entry = ttk.Entry(
            site_file_controls, textvariable=self.site_moment_file_var
        )
        site_file_entry.grid(row=0, column=0, sticky="ew")
        self.control_inputs.append(site_file_entry)
        ttk.Button(
            site_file_controls,
            text="Browse...",
            command=self._choose_site_moment_file,
        ).grid(row=0, column=1, padx=(4, 0))
        row += 1

        ttk.Checkbutton(panel, text="Slab (periodic xy)", variable=self.slab_var).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=3
        )
        row += 1

        layer_frame = self._option_frame(panel, row, "Layer options")
        option_row = self._combo_row(
            layer_frame, 0, "Axis", self.axis_var, ["z", "x", "y"]
        )
        option_row = self._entry_row(
            layer_frame, option_row, "Direction", self.layer_direction_var
        )
        self._help_row(layer_frame, option_row, "Optional dx dy dz vector.")
        option_row += 1
        ttk.Checkbutton(
            layer_frame,
            text="Fractional layer coordinates",
            variable=self.fractional_layers_var,
        ).grid(row=option_row, column=0, columnspan=2, sticky="w", pady=3)
        option_row += 1
        option_row = self._entry_row(
            layer_frame, option_row, "Tolerance (Å)", self.tolerance_var
        )
        self._help_row(
            layer_frame,
            option_row,
            "Fractional units when fractional layers are enabled.",
        )
        self.method_option_frames["layer"] = layer_frame
        row += 1

        neighbor_frame = self._option_frame(panel, row, "Neighbor options")
        ttk.Checkbutton(
            neighbor_frame,
            text="Automatic first-shell cutoff",
            variable=self.auto_cutoff_var,
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=3)
        ttk.Label(neighbor_frame, text="Cutoff (Å)").grid(
            row=1, column=0, sticky="w"
        )
        self.cutoff_entry = ttk.Entry(neighbor_frame, textvariable=self.cutoff_var)
        self.cutoff_entry.grid(row=1, column=1, sticky="ew", padx=(6, 0), pady=3)
        self.control_inputs.append(self.cutoff_entry)
        self.frustrated_check = ttk.Checkbutton(
            neighbor_frame,
            text="Allow frustrated heuristic",
            variable=self.allow_frustrated_var,
        )
        self.frustrated_check.grid(
            row=2, column=0, columnspan=2, sticky="w", pady=3
        )
        self.checker_tolerance_label = ttk.Label(
            neighbor_frame, text="Plane tolerance (Å)", wraplength=105
        )
        self.checker_tolerance_label.grid(row=3, column=0, sticky="w")
        self.checker_tolerance_entry = ttk.Entry(
            neighbor_frame, textvariable=self.tolerance_var
        )
        self.checker_tolerance_entry.grid(
            row=3, column=1, sticky="ew", padx=(6, 0), pady=3
        )
        self.control_inputs.append(self.checker_tolerance_entry)
        self.method_option_frames["neighbor"] = neighbor_frame
        row += 1

        propagation_frame = self._option_frame(panel, row, "Propagation-vector options")
        option_row = self._entry_row(
            propagation_frame, 0, "q-vector", self.q_vector_var
        )
        self._combo_row(
            propagation_frame,
            option_row,
            "AFM preset",
            self.afm_type_var,
            ["custom", "A", "C", "G"],
        )
        self.method_option_frames["propagation-vector"] = propagation_frame
        row += 1

        coordination_frame = self._option_frame(panel, row, "Coordination options")
        option_row = self._entry_row(
            coordination_frame, 0, "Anion species", self.anion_species_var
        )
        self._help_row(coordination_frame, option_row, "Blank enables safe auto-detection.")
        option_row += 1
        option_row = self._entry_row(
            coordination_frame, option_row, "Anion cutoff", self.anion_cutoff_var
        )
        self._help_row(coordination_frame, option_row, "Use auto or a distance in Å.")
        option_row += 1
        option_row = self._entry_row(
            coordination_frame, option_row, "Up CN", self.up_coordination_var
        )
        option_row = self._entry_row(
            coordination_frame, option_row, "Down CN", self.down_coordination_var
        )
        self._entry_row(
            coordination_frame,
            option_row,
            "CN tolerance",
            self.coordination_tolerance_var,
        )
        self.method_option_frames["by-coordination"] = coordination_frame
        row += 1

        graph_frame = self._option_frame(panel, row, "Graph-coloring options")
        option_row = self._entry_row(
            graph_frame, 0, "Maximum colors", self.max_colors_var
        )
        option_row = self._entry_row(
            graph_frame, option_row, "Color spins", self.color_spins_var
        )
        self._help_row(graph_frame, option_row, "+1,-1,0; blank uses the default map.")
        option_row += 1
        ttk.Checkbutton(
            graph_frame,
            text="Balance graph colors",
            variable=self.balance_colors_var,
        ).grid(row=option_row, column=0, columnspan=2, sticky="w", pady=3)
        option_row += 1
        self._entry_row(graph_frame, option_row, "Seed", self.seed_var)
        self.method_option_frames["graph-coloring"] = graph_frame
        row += 1

        random_frame = self._option_frame(panel, row, "Random options")
        self._entry_row(random_frame, 0, "Seed", self.seed_var)
        self.method_option_frames["random"] = random_frame
        row += 1

        row = self._combo_row(
            panel,
            row,
            "Plot color mode",
            self.color_mode_var,
            list(_COLOR_MODES),
        )
        ttk.Checkbutton(
            panel,
            text="Live update",
            variable=self.live_update_var,
            command=self._toggle_live_update,
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=(6, 3))
        row += 1

        secondary_actions = ttk.LabelFrame(panel, text="Open / Export", padding=6)
        secondary_actions.grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=(8, 4)
        )
        secondary_actions.columnconfigure((0, 1), weight=1)
        ttk.Button(
            secondary_actions,
            text="Open spin file...",
            command=self._open_spin_file,
        ).grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        export_group_buttons = [
            ttk.Button(
                secondary_actions,
                text="DM.InitSpin block...",
                command=self._save_spin,
            ),
            ttk.Button(
                secondary_actions,
                text="Complete SIESTA input...",
                command=self._export_complete_input,
            ),
            ttk.Button(
                secondary_actions,
                text="Patched SIESTA input...",
                command=self._export_patched,
            ),
            ttk.Button(
                secondary_actions,
                text="Structure with moments...",
                command=self._export_structure,
            ),
        ]
        for number, button in enumerate(export_group_buttons):
            button.grid(
                row=1 + number // 2,
                column=number % 2,
                sticky="ew",
                padx=(0, 3) if number % 2 == 0 else (3, 0),
                pady=2,
            )
            button.configure(state="disabled")

        self.primary_actions = ttk.LabelFrame(
            container, text="Primary actions", padding=6
        )
        self.primary_actions.grid(
            row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=(4, 10)
        )
        self.primary_actions.columnconfigure(0, weight=1)
        self.primary_actions.columnconfigure(1, weight=3)
        self.generate_button = ttk.Button(
            self.primary_actions, text="Generate", command=self._generate_explicit
        )
        self.generate_button.grid(row=0, column=0, sticky="ew", padx=(0, 3))
        self.complete_input_action = ttk.Button(
            self.primary_actions,
            text="Build complete SIESTA input (make-input)...",
            command=self._export_complete_input,
            state="disabled",
        )
        self.complete_input_action.grid(row=0, column=1, sticky="ew", padx=(3, 0))
        ttk.Label(
            self.primary_actions,
            text="Creates a runnable starting FDF; same as `siesta-afm make-input`.",
            foreground="#555555",
            wraplength=520,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))
        self.export_buttons = [self.complete_input_action, *export_group_buttons]

        self._configure_canvas_mousewheel(
            self.controls_canvas,
            blocked=(self.magnetization_tree,),
        )

    def _make_scrollable_frame(
        self,
        parent: Any,
        *,
        padding: int | tuple[int, ...] = 0,
    ) -> tuple[Any, Any]:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        canvas = self.tk.Canvas(parent, highlightthickness=0)
        scrollbar = self.ttk.Scrollbar(
            parent, orient="vertical", command=canvas.yview
        )
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        content = self.ttk.Frame(canvas, padding=padding)
        content_window = canvas.create_window((0, 0), window=content, anchor="nw")
        content.bind(
            "<Configure>",
            lambda _event: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.bind(
            "<Configure>",
            lambda event: canvas.itemconfigure(content_window, width=event.width),
        )
        return canvas, content

    def _configure_canvas_mousewheel(
        self,
        canvas: Any,
        *,
        blocked: Sequence[Any] = (),
    ) -> None:
        blocked_widgets = set(blocked)

        def bind_widget(widget: Any) -> None:
            if widget in blocked_widgets:
                widget.bind(
                    "<Enter>",
                    lambda _event, target=canvas: self._suspend_canvas_mousewheel(
                        target
                    ),
                    add="+",
                )
                return
            widget.bind(
                "<Enter>",
                lambda _event, target=canvas: self._activate_canvas_mousewheel(
                    target
                ),
                add="+",
            )
            widget.bind(
                "<Leave>",
                lambda _event, target=canvas: self._schedule_canvas_mousewheel_unbind(
                    target
                ),
                add="+",
            )
            for child in widget.winfo_children():
                bind_widget(child)

        bind_widget(canvas)

    def _cancel_mousewheel_leave(self) -> None:
        if self._mousewheel_leave_after_id is None:
            return
        try:
            self.root.after_cancel(self._mousewheel_leave_after_id)
        except self.tk.TclError:
            pass
        self._mousewheel_leave_after_id = None

    def _activate_canvas_mousewheel(self, canvas: Any) -> None:
        self._cancel_mousewheel_leave()
        self._active_scroll_canvas = canvas
        canvas.bind_all("<MouseWheel>", self._scroll_active_canvas)

    def _schedule_canvas_mousewheel_unbind(self, canvas: Any) -> None:
        if self._active_scroll_canvas is not canvas:
            return
        self._cancel_mousewheel_leave()
        self._mousewheel_leave_after_id = self.root.after_idle(
            lambda target=canvas: self._deactivate_canvas_mousewheel(target)
        )

    def _deactivate_canvas_mousewheel(self, canvas: Any | None = None) -> None:
        self._cancel_mousewheel_leave()
        if canvas is not None and self._active_scroll_canvas is not canvas:
            return
        if self._active_scroll_canvas is not None:
            self._active_scroll_canvas.unbind_all("<MouseWheel>")
        self._active_scroll_canvas = None

    def _suspend_canvas_mousewheel(self, canvas: Any) -> None:
        if self._active_scroll_canvas is canvas:
            self._deactivate_canvas_mousewheel(canvas)

    def _scroll_active_canvas(self, event: Any) -> str | None:
        if self._active_scroll_canvas is None:
            return None
        return self._scroll_canvas(self._active_scroll_canvas, event)

    @staticmethod
    def _scroll_canvas(canvas: Any, event: Any) -> str:
        steps = int(-1 * (event.delta / 120))
        if steps:
            canvas.yview_scroll(steps, "units")
        return "break"

    def _bind_controls_mousewheel(self, _event: object | None = None) -> None:
        self._activate_canvas_mousewheel(self.controls_canvas)

    def _unbind_controls_mousewheel(self, _event: object | None = None) -> None:
        self._deactivate_canvas_mousewheel()

    def _scroll_controls(self, event: Any) -> str:
        return self._scroll_canvas(self.controls_canvas, event)

    def _build_results(self) -> None:
        ttk = self.ttk
        panel = ttk.Frame(self.main_pane, padding=(0, 10, 10, 10))
        self.main_pane.add(panel, weight=1)
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(1, weight=1)

        ttk.Label(panel, textvariable=self.mode_var).grid(row=0, column=0, sticky="w")
        self.results_pane = ttk.PanedWindow(panel, orient="vertical")
        self.results_pane.grid(row=1, column=0, sticky="nsew", pady=(4, 0))
        self.results_pane.bind(
            "<Map>",
            lambda _event: self.root.after_idle(
                self._set_initial_results_pane_height
            ),
        )

        preview = ttk.LabelFrame(
            self.results_pane, text="Interactive 3D preview", padding=4
        )
        self.results_pane.add(preview, weight=3)
        preview.columnconfigure(0, weight=1)
        preview.rowconfigure(1, weight=1)
        ttk.Checkbutton(
            preview,
            text="Show atom indices",
            variable=self.show_atom_indices_var,
            command=self._show_atom_indices_changed,
        ).grid(row=0, column=0, sticky="w", padx=2, pady=(0, 3))
        self.canvas_host = ttk.Frame(preview)
        self.canvas_host.grid(row=1, column=0, sticky="nsew")
        self.toolbar_host = ttk.Frame(preview)
        self.toolbar_host.grid(row=2, column=0, sticky="ew")

        notebook = ttk.Notebook(self.results_pane)
        self.results_pane.add(notebook, weight=2)
        analysis_frame = ttk.Frame(notebook)
        sites_frame = ttk.Frame(notebook)
        spin_frame = ttk.Frame(notebook)
        batch_frame = ttk.Frame(notebook)
        notebook.add(analysis_frame, text="Analysis")
        notebook.add(sites_frame, text="Sites")
        notebook.add(spin_frame, text="DM.InitSpin")
        notebook.add(batch_frame, text="Batch workflow")
        self.results_notebook = notebook
        self.analysis_text = self._readonly_text(analysis_frame)
        self.spin_text = self._readonly_text(spin_frame)
        sites_frame.columnconfigure(0, weight=1)
        sites_frame.rowconfigure(0, weight=1)
        columns = ("atom", "element", "CN", "sublattice", "sign", "moment")
        self.sites_tree = ttk.Treeview(
            sites_frame, columns=columns, show="headings", selectmode="browse"
        )
        headings = {
            "atom": "atom (1-based)",
            "element": "element",
            "CN": "CN",
            "sublattice": "sublattice",
            "sign": "sign",
            "moment": "moment (μB)",
        }
        widths = {
            "atom": 105,
            "element": 80,
            "CN": 60,
            "sublattice": 90,
            "sign": 55,
            "moment": 100,
        }
        for column in columns:
            self.sites_tree.heading(column, text=headings[column])
            self.sites_tree.column(
                column,
                width=widths[column],
                minwidth=50,
                anchor="center",
                stretch=True,
            )
        sites_scrollbar = ttk.Scrollbar(
            sites_frame, orient="vertical", command=self.sites_tree.yview
        )
        self.sites_tree.configure(yscrollcommand=sites_scrollbar.set)
        self.sites_tree.grid(row=0, column=0, sticky="nsew")
        sites_scrollbar.grid(row=0, column=1, sticky="ns")
        self.sites_summary_var = self.tk.StringVar(
            value="n_up = 0 / n_down = 0 / n_zero = 0, net moment = 0 μB"
        )
        ttk.Label(
            sites_frame,
            textvariable=self.sites_summary_var,
            anchor="w",
            padding=(6, 4),
        ).grid(row=1, column=0, columnspan=2, sticky="ew")

        self._build_batch_workflow(batch_frame)

        status = ttk.Label(
            self.root,
            textvariable=self.status_var,
            relief="sunken",
            anchor="w",
            padding=(6, 3),
        )
        status.grid(row=1, column=0, sticky="ew")

    def _build_batch_workflow(self, parent: Any) -> None:
        ttk = self.ttk
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        notebook = ttk.Notebook(parent)
        notebook.grid(row=0, column=0, sticky="nsew")
        candidates_tab = ttk.Frame(notebook)
        prepare_tab = ttk.Frame(notebook)
        results_tab = ttk.Frame(notebook)
        notebook.add(candidates_tab, text="Candidates")
        notebook.add(prepare_tab, text="Prepare jobs")
        notebook.add(results_tab, text="Results")
        self.batch_notebook = notebook
        self._build_candidates_tab(candidates_tab)
        self._build_prepare_tab(prepare_tab)
        self._build_collected_results_tab(results_tab)

    def _build_candidates_tab(self, parent: Any) -> None:
        ttk = self.ttk
        self.candidates_canvas, parent = self._make_scrollable_frame(
            parent, padding=6
        )
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(3, weight=1)
        self.batch_workflow_help_label = ttk.Label(
            parent,
            text=(
                "Compare several initial spin states: generate candidates here, "
                "prepare SIESTA job folders, then load results after running them "
                "externally to compare total energy. See README 'Comparing "
                "magnetic states by total energy'."
            ),
            wraplength=720,
            justify="left",
            foreground="#555555",
        )
        self.batch_workflow_help_label.grid(
            row=0, column=0, sticky="ew", pady=(0, 5)
        )
        methods = ttk.LabelFrame(parent, text="Batch candidates", padding=5)
        methods.grid(row=1, column=0, sticky="ew")
        for index, method in enumerate(_BATCH_METHODS):
            ttk.Checkbutton(
                methods,
                text=method,
                variable=self.batch_method_vars[method],
            ).grid(row=index // 3, column=index % 3, sticky="w", padx=(0, 12))

        controls = ttk.Frame(parent)
        controls.grid(row=2, column=0, sticky="ew", pady=(4, 3))
        controls.columnconfigure(4, weight=1)
        ttk.Label(controls, text="n-configs").grid(row=0, column=0, sticky="w")
        self.batch_n_configs_spinbox = ttk.Spinbox(
            controls,
            from_=1,
            to=999,
            width=6,
            textvariable=self.batch_n_configs_var,
        )
        self.batch_n_configs_spinbox.grid(row=0, column=1, sticky="w", padx=(4, 12))
        ttk.Checkbutton(
            controls,
            text="Keep global spin inversion",
            variable=self.batch_keep_inversion_var,
        ).grid(row=0, column=2, columnspan=3, sticky="w")
        ttk.Label(controls, text="Output directory").grid(
            row=1, column=0, sticky="w", pady=(3, 0)
        )
        ttk.Entry(controls, textvariable=self.candidate_output_var).grid(
            row=1, column=1, columnspan=4, sticky="ew", padx=4, pady=(3, 0)
        )
        ttk.Button(
            controls,
            text="Browse...",
            command=self._choose_candidate_output,
        ).grid(row=1, column=5, padx=(0, 4), pady=(3, 0))
        self.generate_candidates_button = ttk.Button(
            controls,
            text="Generate candidates",
            command=self._generate_candidates,
        )
        self.generate_candidates_button.grid(row=1, column=6, pady=(3, 0))
        ttk.Label(controls, text="Manual groups file").grid(
            row=2, column=0, sticky="w", pady=(3, 0)
        )
        ttk.Entry(controls, textvariable=self.batch_group_file_var).grid(
            row=2, column=1, columnspan=4, sticky="ew", padx=4, pady=(3, 0)
        )
        ttk.Button(
            controls,
            text="Browse...",
            command=self._choose_batch_group_file,
        ).grid(row=2, column=5, padx=(0, 4), pady=(3, 0))

        table_frame = ttk.Frame(parent)
        table_frame.grid(row=3, column=0, sticky="nsew")
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        self.candidate_tree = ttk.Treeview(
            table_frame,
            columns=_CANDIDATE_COLUMNS,
            show="headings",
            height=5,
        )
        candidate_widths = {
            "config_id": 65,
            "method": 130,
            "n_up": 55,
            "n_down": 65,
            "net_spin": 75,
            "afm_score": 80,
            "file": 115,
        }
        for column in _CANDIDATE_COLUMNS:
            self.candidate_tree.heading(column, text=column)
            self.candidate_tree.column(
                column,
                width=candidate_widths[column],
                minwidth=50,
                anchor="center",
                stretch=column in {"method", "file"},
            )
        candidate_y = ttk.Scrollbar(
            table_frame, orient="vertical", command=self.candidate_tree.yview
        )
        candidate_x = ttk.Scrollbar(
            table_frame, orient="horizontal", command=self.candidate_tree.xview
        )
        self.candidate_tree.configure(
            yscrollcommand=candidate_y.set, xscrollcommand=candidate_x.set
        )
        self.candidate_tree.grid(row=0, column=0, sticky="nsew")
        candidate_y.grid(row=0, column=1, sticky="ns")
        candidate_x.grid(row=1, column=0, sticky="ew")
        self.candidate_messages = self.tk.Text(parent, height=3, wrap="word")
        self.candidate_messages.grid(row=4, column=0, sticky="ew", pady=(3, 0))
        self.candidate_messages.configure(state="disabled")
        self._configure_canvas_mousewheel(
            self.candidates_canvas,
            blocked=(self.candidate_tree,),
        )

    def _build_prepare_tab(self, parent: Any) -> None:
        ttk = self.ttk
        self.prepare_canvas, parent = self._make_scrollable_frame(parent, padding=6)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(4, weight=1)
        ttk.Label(
            parent,
            text=(
                "Select a complete make-input FDF, a directory containing "
                "manifest.csv, and a destination for the job folders."
            ),
            wraplength=650,
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 5))
        self._batch_path_row(
            parent,
            1,
            "Base input",
            self.base_input_var,
            self._choose_base_input,
        )
        self._batch_path_row(
            parent,
            2,
            "Candidates directory",
            self.candidates_dir_var,
            self._choose_candidates_dir,
        )
        self._batch_path_row(
            parent,
            3,
            "Output directory",
            self.jobs_output_var,
            self._choose_jobs_output,
        )
        self.prepare_jobs_button = ttk.Button(
            parent, text="Prepare job folders", command=self._prepare_job_folders
        )
        self.prepare_jobs_button.grid(
            row=4, column=0, columnspan=3, sticky="ew", pady=(5, 3)
        )
        self.job_folders_text = self.tk.Text(parent, height=9, wrap="none")
        self.job_folders_text.grid(
            row=5, column=0, columnspan=3, sticky="nsew", pady=(3, 0)
        )
        parent.rowconfigure(5, weight=1)
        self.job_folders_text.configure(state="disabled")
        self._configure_canvas_mousewheel(self.prepare_canvas)

    def _build_collected_results_tab(self, parent: Any) -> None:
        ttk = self.ttk
        self.collected_results_canvas, parent = self._make_scrollable_frame(
            parent, padding=6
        )
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(4, weight=1)
        self._batch_path_row(
            parent,
            0,
            "Jobs directory",
            self.jobs_dir_var,
            self._choose_jobs_dir,
        )
        ttk.Label(parent, text="or").grid(row=1, column=0, sticky="w")
        self._batch_path_row(
            parent,
            2,
            "Existing results.csv",
            self.results_csv_var,
            self._choose_results_csv,
        )
        self.collect_results_button = ttk.Button(
            parent,
            text="Collect / Load results",
            command=self._collect_or_load_results,
        )
        self.collect_results_button.grid(
            row=3, column=0, columnspan=3, sticky="ew", pady=(4, 3)
        )
        table_frame = ttk.Frame(parent)
        table_frame.grid(row=4, column=0, columnspan=3, sticky="nsew")
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        self.collected_results_tree = ttk.Treeview(
            table_frame,
            columns=_RESULT_COLUMNS,
            show="headings",
            height=8,
        )
        result_widths = {
            "config_id": 70,
            "total_energy": 100,
            "final_net_spin": 100,
            "sign_retention": 100,
            "collapsed_atoms": 105,
            "spin_population_source": 170,
            "scf_converged": 100,
            "geometry_converged": 125,
            "status": 130,
        }
        for column in _RESULT_COLUMNS:
            self.collected_results_tree.heading(column, text=column)
            self.collected_results_tree.column(
                column,
                width=result_widths[column],
                minwidth=60,
                anchor="center",
                stretch=column in {"spin_population_source", "status"},
            )
        results_y = ttk.Scrollbar(
            table_frame,
            orient="vertical",
            command=self.collected_results_tree.yview,
        )
        results_x = ttk.Scrollbar(
            table_frame,
            orient="horizontal",
            command=self.collected_results_tree.xview,
        )
        self.collected_results_tree.configure(
            yscrollcommand=results_y.set, xscrollcommand=results_x.set
        )
        self.collected_results_tree.grid(row=0, column=0, sticky="nsew")
        results_y.grid(row=0, column=1, sticky="ns")
        results_x.grid(row=1, column=0, sticky="ew")
        self.collected_results_tree.tag_configure(
            "unconverged", foreground="#777777"
        )
        self.collected_results_tree.tag_configure(
            "near_ground", background="#fff3b0"
        )
        ttk.Label(
            parent,
            text=(
                "Gray = SCF not converged. Yellow = converged candidates within "
                f"{_NEAR_GROUND_ENERGY_WINDOW_EV:g} eV of the lowest energy; "
                "review all highlighted states rather than selecting one automatically."
            ),
            wraplength=720,
            foreground="#555555",
        ).grid(row=5, column=0, columnspan=3, sticky="w", pady=(3, 0))
        self._configure_canvas_mousewheel(
            self.collected_results_canvas,
            blocked=(self.collected_results_tree,),
        )

    def _batch_path_row(
        self,
        parent: Any,
        row: int,
        label: str,
        variable: Any,
        command: Any,
    ) -> None:
        self.ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        self.ttk.Entry(parent, textvariable=variable).grid(
            row=row, column=1, sticky="ew", padx=5, pady=2
        )
        self.ttk.Button(parent, text="Browse...", command=command).grid(
            row=row, column=2, sticky="e"
        )

    def _entry_row(self, parent: Any, row: int, label: str, variable: Any) -> int:
        self.ttk.Label(parent, text=label, wraplength=110).grid(
            row=row, column=0, sticky="w"
        )
        entry = self.ttk.Entry(parent, textvariable=variable)
        entry.grid(
            row=row, column=1, sticky="ew", padx=(6, 0), pady=3
        )
        self.control_inputs.append(entry)
        return row + 1

    def _combo_row(
        self,
        parent: Any,
        row: int,
        label: str,
        variable: Any,
        values: Sequence[str],
    ) -> int:
        self.ttk.Label(parent, text=label, wraplength=110).grid(
            row=row, column=0, sticky="w"
        )
        combobox = self.ttk.Combobox(
            parent, textvariable=variable, values=list(values), state="readonly"
        )
        combobox.grid(row=row, column=1, sticky="ew", padx=(6, 0), pady=3)
        self.control_inputs.append(combobox)
        return row + 1

    def _option_frame(self, parent: Any, row: int, title: str) -> Any:
        frame = self.ttk.LabelFrame(parent, text=title, padding=6)
        frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=4)
        frame.columnconfigure(0, minsize=105, weight=0)
        frame.columnconfigure(1, minsize=160, weight=1)
        return frame

    def _help_row(self, parent: Any, row: int, text: str) -> None:
        self.ttk.Label(
            parent,
            text=text,
            foreground="#666666",
            wraplength=430,
            justify="left",
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 3))

    def _set_initial_pane_width(self) -> None:
        if self._pane_width_initialized:
            return
        self.root.update_idletasks()
        if self.main_pane.winfo_width() > 1 and len(self.main_pane.panes()) > 1:
            self.main_pane.sashpos(0, _LEFT_PANEL_MIN_WIDTH)
            self.main_pane.bind("<ButtonRelease-1>", self._enforce_left_pane_width)
            self._pane_width_initialized = True

    def _enforce_left_pane_width(self, _event: object | None = None) -> None:
        if len(self.main_pane.panes()) > 1 and self.main_pane.sashpos(0) < _LEFT_PANEL_MIN_WIDTH:
            self.main_pane.sashpos(0, _LEFT_PANEL_MIN_WIDTH)

    def _set_initial_results_pane_height(self) -> None:
        if self._results_pane_height_initialized:
            return
        self.root.update_idletasks()
        if self.results_pane.winfo_height() > 1 and len(self.results_pane.panes()) > 1:
            initial_height = int(self.results_pane.winfo_height() * 3 / 5)
            self.results_pane.sashpos(
                0, min(initial_height, self._maximum_preview_height())
            )
            self.results_pane.bind(
                "<ButtonRelease-1>", self._enforce_results_notebook_height
            )
            self._results_pane_height_initialized = True

    def _maximum_preview_height(self) -> int:
        return max(
            0,
            self.results_pane.winfo_height()
            - _RESULTS_NOTEBOOK_MIN_HEIGHT
            - _PANE_SASH_MARGIN,
        )

    def _enforce_results_notebook_height(
        self, _event: object | None = None
    ) -> None:
        if (
            len(self.results_pane.panes()) > 1
            and self.results_pane.sashpos(0) > self._maximum_preview_height()
        ):
            self.results_pane.sashpos(0, self._maximum_preview_height())

    def _show_method_options(self) -> None:
        for frame in self.method_option_frames.values():
            frame.grid_remove()
        method = self.method_var.get()
        active: list[str] = []
        if method == "layer":
            active.append("layer")
        if method in {"neighbor-bipartite", "checkerboard", "graph-coloring"}:
            active.append("neighbor")
        if method == "propagation-vector":
            active.append("propagation-vector")
        if method == "by-coordination":
            active.append("by-coordination")
        if method == "graph-coloring":
            active.append("graph-coloring")
        if method == "random":
            active.append("random")
        for name in active:
            self.method_option_frames[name].grid()
        if method == "neighbor-bipartite":
            self.frustrated_check.grid()
        else:
            self.frustrated_check.grid_remove()
        if method == "checkerboard":
            self.checker_tolerance_label.grid()
            self.checker_tolerance_entry.grid()
        else:
            self.checker_tolerance_label.grid_remove()
            self.checker_tolerance_entry.grid_remove()

    def _populate_magnetization_tree(self) -> None:
        for item in self.magnetization_tree.get_children():
            self.magnetization_tree.delete(item)
        for index, row in enumerate(self.magnetization_rows):
            coordination = (
                row.coordination
                if row.coordination is not None
                else ("-" if row.use else "")
            )
            self.magnetization_tree.insert(
                "",
                "end",
                iid=str(index),
                values=(
                    "✓" if row.use else "☐",
                    row.element,
                    row.label,
                    coordination,
                    row.value,
                    row.count,
                    row.role,
                ),
            )
        self.cli_options_var.set(
            equivalent_cli_options(self.magnetization_rows, self.method_var.get())
        )

    def _edit_magnetization_cell(self, event: Any) -> None:
        item = self.magnetization_tree.identify_row(event.y)
        column = self.magnetization_tree.identify_column(event.x)
        if not item or not column:
            return
        row_index = int(item)
        column_index = int(column.removeprefix("#")) - 1
        names = ("use", "element", "label", "CN", "value", "count", "role")
        name = names[column_index]
        row = self.magnetization_rows[row_index]
        if name == "use":
            method = self.method_var.get()
            toggle_magnetization_use(self.magnetization_rows, row_index, method)
            if method == "by-coordination" and row.coordination is not None:
                indices = ", ".join(str(index) for index in row.atom_indices) or "unknown"
                self._coordination_use_note = (
                    f"{row.element} CN={row.coordination} use={row.use}; atom "
                    f"indices: {indices}. Unchecking one coordination site keeps "
                    "the element selected; set moment 0 or use --exclude-atoms "
                    "to fully exclude those atoms."
                )
            self._table_changed(refresh=True)
            if self._coordination_use_note:
                self.status_var.set(self._coordination_use_note)
            return
        if name == "label" and self.method_var.get() != "by-coordination":
            return
        if name == "role" and self.method_var.get() != "by-species":
            return
        if name not in {"label", "value", "role"} or not row.use:
            return
        bbox = self.magnetization_tree.bbox(item, column)
        if not bbox:
            return
        if self._cell_editor is not None:
            self._cell_editor.destroy()
        x, y, width, height = bbox
        current = row.label if name == "label" else row.value if name == "value" else row.role
        variable = self.tk.StringVar(value=current)
        if name == "role":
            editor = self.ttk.Combobox(
                self.magnetization_tree,
                textvariable=variable,
                values=("up", "down"),
                state="readonly",
            )
        else:
            editor = self.ttk.Entry(self.magnetization_tree, textvariable=variable)
        self._cell_editor = editor
        editor.place(x=x, y=y, width=width, height=height)
        editor.focus_set()

        def close(*, save: bool) -> None:
            if self._cell_editor is None:
                return
            value = variable.get().strip()
            if save:
                if name == "value" and value:
                    try:
                        if float(value) < 0:
                            raise ValueError
                    except ValueError:
                        self.status_var.set("Moment values must be nonnegative numbers.")
                        return
                if name == "label":
                    row.label = value
                elif name == "value":
                    row.value = value
                else:
                    row.role = value
                self._table_changed()
            self._cell_editor.destroy()
            self._cell_editor = None

        editor.bind("<Return>", lambda _event: close(save=True))
        editor.bind("<Escape>", lambda _event: close(save=False))
        editor.bind("<FocusOut>", lambda _event: close(save=True))

    def _table_changed(self, *, refresh: bool = False) -> None:
        if refresh:
            self._refresh_magnetization_table()
        else:
            self._populate_magnetization_tree()
        if self.viewing_spin_path is not None:
            self.viewing_spin_path = None
            self.mode_var.set("generation mode (spin-file view ended)")
        self._schedule_live_update()

    def _readonly_text(self, parent: Any) -> Any:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        text = self.tk.Text(parent, wrap="word", font=("TkFixedFont", 10))
        scrollbar = self.ttk.Scrollbar(parent, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=scrollbar.set, state="disabled")
        text.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        return text

    def _install_traces(self) -> None:
        variables = (
            self.site_moment_file_var,
            self.site_comments_var,
            self.axis_var,
            self.layer_direction_var,
            self.fractional_layers_var,
            self.auto_cutoff_var,
            self.cutoff_var,
            self.tolerance_var,
            self.q_vector_var,
            self.afm_type_var,
            self.max_colors_var,
            self.color_spins_var,
            self.balance_colors_var,
            self.seed_var,
            self.allow_frustrated_var,
            self.color_mode_var,
        )
        for variable in variables:
            variable.trace_add("write", self._parameter_changed)
        self.method_var.trace_add("write", self._method_changed)
        self.slab_var.trace_add("write", self._structure_setting_changed)
        for variable in (
            self.anion_species_var,
            self.anion_cutoff_var,
            self.up_coordination_var,
            self.down_coordination_var,
            self.coordination_tolerance_var,
        ):
            variable.trace_add("write", self._coordination_setting_changed)

    @staticmethod
    def _split_words(text: str) -> tuple[str, ...]:
        return tuple(value for value in text.replace(",", " ").split() if value)

    def _method_changed(self, *_: object) -> None:
        if not self._traces_ready:
            return
        if self._table_after_id is not None:
            self.root.after_cancel(self._table_after_id)
            self._table_after_id = None
        if self.method_var.get() != "by-coordination":
            self._coordination_use_note = None
        self._show_method_options()
        if self.current_structure is not None:
            self._refresh_magnetization_table()
        self._parameter_changed()

    def _coordination_setting_changed(self, *_: object) -> None:
        if not self._traces_ready:
            return
        if self.method_var.get() != "by-coordination" or self.current_structure is None:
            self._parameter_changed()
            return
        if self.viewing_spin_path is not None:
            self.viewing_spin_path = None
            self.mode_var.set("generation mode (spin-file view ended)")
        if self._live_after_id is not None:
            self.root.after_cancel(self._live_after_id)
            self._live_after_id = None
        if self._table_after_id is not None:
            self.root.after_cancel(self._table_after_id)
        self._table_after_id = self.root.after(300, self._run_table_refresh)
        self.status_var.set("Coordination table refresh scheduled...")

    def _structure_setting_changed(self, *_: object) -> None:
        if not self._traces_ready or self.structure_path is None:
            return
        try:
            self.current_structure = read_structure(
                self.structure_path, slab=self.slab_var.get()
            )
            self._refresh_magnetization_table()
        except Exception as exc:
            self.status_var.set(f"Structure refresh failed: {exc}")
            return
        self._parameter_changed()

    def _reset_show_indices_for_structure(self, structure: Structure) -> None:
        self.show_atom_indices_var.set(
            len(structure.symbols) <= _AUTO_SHOW_INDICES_MAX_ATOMS
        )

    def _run_table_refresh(self) -> None:
        self._table_after_id = None
        self._refresh_magnetization_table()
        self._schedule_live_update()

    def _refresh_magnetization_table(self) -> None:
        if self.current_structure is None:
            self._coordination_fallback = None
            self.magnetization_rows = []
            self._populate_magnetization_tree()
            return
        method = self.method_var.get()
        rows, self._coordination_fallback = safe_magnetization_rows_from_structure(
            self.current_structure,
            method,
            existing_rows=self.magnetization_rows,
            anion_species=self._split_words(self.anion_species_var.get()) or None,
            anion_cutoff=self.anion_cutoff_var.get().strip() or "auto",
        )
        if self._coordination_fallback:
            self.status_var.set(self._coordination_fallback)
        self.magnetization_rows = rows
        self._populate_magnetization_tree()

    def _choose_structure(self, *, schedule: bool = True) -> bool:
        path = self.deps.filedialog.askopenfilename(
            title="Open structure",
            filetypes=[
                ("Structure files", "*.cif *.xyz *.fdf *.xv *.XV *.vasp"),
                ("POSCAR / CONTCAR", "POSCAR CONTCAR"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return False
        candidate = Path(path)
        try:
            structure = read_structure(candidate, slab=self.slab_var.get())
        except Exception as exc:
            self.status_var.set(f"Structure not loaded: {exc}")
            self.deps.messagebox.showerror("Open structure failed", str(exc))
            return False
        self.structure_path = candidate
        self.current_structure = structure
        self._reset_show_indices_for_structure(structure)
        self._coordination_use_note = None
        self.file_var.set(str(self.structure_path))
        self.viewing_spin_path = None
        self.mode_var.set("generation mode")
        self._reset_camera = True
        self.magnetization_rows = []
        self._refresh_magnetization_table()
        self.status_var.set(
            self._coordination_fallback
            or f"Loaded structure: {self.structure_path.name}"
        )
        if schedule:
            self._schedule_live_update()
        return True

    def _choose_site_moment_file(self) -> None:
        path = self.deps.filedialog.askopenfilename(
            title="Open site moment CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.site_moment_file_var.set(path)

    def _choose_candidate_output(self) -> None:
        path = self.deps.filedialog.askdirectory(title="Candidate output directory")
        if path:
            self.candidate_output_var.set(path)

    def _choose_batch_group_file(self) -> None:
        path = self.deps.filedialog.askopenfilename(
            title="Open manual groups file",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
        )
        if path:
            self.batch_group_file_var.set(path)

    def _choose_base_input(self) -> None:
        path = self.deps.filedialog.askopenfilename(
            title="Select complete make-input FDF",
            filetypes=[("FDF files", "*.fdf"), ("All files", "*.*")],
        )
        if path:
            self.base_input_var.set(path)

    def _choose_candidates_dir(self) -> None:
        path = self.deps.filedialog.askdirectory(
            title="Select directory containing manifest.csv"
        )
        if path:
            self.candidates_dir_var.set(path)

    def _choose_jobs_output(self) -> None:
        path = self.deps.filedialog.askdirectory(title="Job folders output directory")
        if path:
            self.jobs_output_var.set(path)

    def _choose_jobs_dir(self) -> None:
        path = self.deps.filedialog.askdirectory(title="Select jobs directory")
        if path:
            self.jobs_dir_var.set(path)
            self.results_csv_var.set("")

    def _choose_results_csv(self) -> None:
        path = self.deps.filedialog.askopenfilename(
            title="Open existing results.csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.results_csv_var.set(path)
            self.jobs_dir_var.set("")

    @staticmethod
    def _three_numbers(text: str, label: str) -> tuple[float, float, float]:
        values = [float(value) for value in text.replace(",", " ").split()]
        if len(values) != 3:
            raise ValueError(f"{label} must contain exactly three numbers")
        return values[0], values[1], values[2]

    def _batch_workflow_kwargs(self, methods: Sequence[str]) -> dict[str, object]:
        cutoff: str | float = (
            "auto" if self.auto_cutoff_var.get() else float(self.cutoff_var.get())
        )
        layer_direction = None
        direction_text = self.layer_direction_var.get().strip()
        if "layer" in methods and direction_text:
            layer_direction = self._three_numbers(direction_text, "layer direction")

        q_vector = None
        afm_type = None
        if "propagation-vector" in methods:
            selected_afm_type = self.afm_type_var.get()
            if selected_afm_type == "custom":
                q_vector = self._three_numbers(self.q_vector_var.get(), "q-vector")
            else:
                afm_type = selected_afm_type

        up_species, down_species = species_roles_from_rows(self.magnetization_rows)
        if "by-coordination" in methods:
            up_coordination = tuple(
                int(value)
                for value in self._split_words(self.up_coordination_var.get())
            )
            down_coordination = tuple(
                int(value)
                for value in self._split_words(self.down_coordination_var.get())
            )
            coordination_tolerance = int(self.coordination_tolerance_var.get())
        else:
            up_coordination = (6,)
            down_coordination = (4,)
            coordination_tolerance = 0
        if "graph-coloring" in methods:
            max_colors = int(self.max_colors_var.get())
            color_spins = self.color_spins_var.get().strip() or None
            balance_colors = self.balance_colors_var.get()
        else:
            max_colors = 4
            color_spins = None
            balance_colors = False
        return {
            "site_moment_file": self.site_moment_file_var.get().strip() or None,
            "axis": self.axis_var.get(),
            "layer_direction": layer_direction,
            "layer_tolerance": float(self.tolerance_var.get()),
            "fractional_layers": self.fractional_layers_var.get(),
            "cutoff": cutoff,
            "neighbor_shell": 1,
            "allow_frustrated": self.allow_frustrated_var.get(),
            "q_vector": q_vector,
            "afm_type": afm_type,
            "up_species": up_species,
            "down_species": down_species,
            "anion_species": self._split_words(self.anion_species_var.get()) or None,
            "anion_cutoff": self.anion_cutoff_var.get().strip() or "auto",
            "up_coordination": up_coordination,
            "down_coordination": down_coordination,
            "coordination_tolerance": coordination_tolerance,
            "max_colors": max_colors,
            "color_spins": color_spins,
            "balance_colors": balance_colors,
            "group_file": (
                self.batch_group_file_var.get().strip()
                if "manual-groups" in methods
                else None
            ),
            "seed_offset": int(self.seed_var.get()),
        }

    def _generate_candidates(self) -> None:
        try:
            if self.structure_path is None:
                raise ValueError("select a structure file first")
            methods = [
                method
                for method in _BATCH_METHODS
                if self.batch_method_vars[method].get()
            ]
            if not methods:
                raise ValueError("select at least one candidate method")
            output_dir = self.candidate_output_var.get().strip()
            if not output_dir:
                raise ValueError("select a candidate output directory")
            structure = read_structure(self.structure_path, slab=self.slab_var.get())
            species = magnetic_species_from_rows(self.magnetization_rows)
            result = run_candidate_generation(
                structure,
                species,
                methods,
                batch_moment_text_from_rows(self.magnetization_rows, methods),
                int(self.batch_n_configs_var.get()),
                output_dir,
                keep_global_spin_inversion=self.batch_keep_inversion_var.get(),
                site_comments=self.site_comments_var.get(),
                **self._batch_workflow_kwargs(methods),
            )
        except Exception as exc:
            self.status_var.set(f"Candidate generation failed: {exc}")
            self.deps.messagebox.showerror("Candidate generation failed", str(exc))
            return

        for item in self.candidate_tree.get_children():
            self.candidate_tree.delete(item)
        for values in candidate_table_rows(result):
            self.candidate_tree.insert("", "end", values=values)
        diagnostics = [f"NOTICE: {notice}" for notice in result.notices]
        diagnostics.extend(f"SKIPPED: {failure}" for failure in result.failures)
        self._set_text(
            self.candidate_messages,
            "\n".join(diagnostics) or "No warnings or skipped methods.",
        )
        self.candidates_dir_var.set(output_dir)
        self.status_var.set(
            f"Generated {len(result.manifest)} candidate(s): {result.manifest_path}"
        )

    def _prepare_job_folders(self) -> None:
        try:
            base_input = self.base_input_var.get().strip()
            candidates_dir = self.candidates_dir_var.get().strip()
            output_dir = self.jobs_output_var.get().strip()
            if not base_input or not candidates_dir or not output_dir:
                raise ValueError(
                    "select the base input, candidates directory, and output directory"
                )
            folders = prepare_job_folders(base_input, candidates_dir, output_dir)
        except Exception as exc:
            self.status_var.set(f"Job preparation failed: {exc}")
            self.deps.messagebox.showerror("Job preparation failed", str(exc))
            return
        self._set_text(
            self.job_folders_text,
            "\n".join(folder.name for folder in folders) + ("\n" if folders else ""),
        )
        self.jobs_dir_var.set(output_dir)
        self.results_csv_var.set("")
        self.status_var.set(f"Prepared {len(folders)} job folder(s) in {output_dir}")

    def _collect_or_load_results(self) -> None:
        try:
            jobs_dir = self.jobs_dir_var.get().strip() or None
            results_csv = self.results_csv_var.get().strip() or None
            rows = collect_or_load_results(
                jobs_dir=jobs_dir,
                results_csv=results_csv,
            )
        except Exception as exc:
            self.status_var.set(f"Results not loaded: {exc}")
            self.deps.messagebox.showerror("Collect / Load results failed", str(exc))
            return
        for item in self.collected_results_tree.get_children():
            self.collected_results_tree.delete(item)
        for row in results_table_rows(rows):
            self.collected_results_tree.insert(
                "", "end", values=row.values, tags=row.tags
            )
        if jobs_dir and not results_csv:
            self.results_csv_var.set(str(Path(jobs_dir) / "results.csv"))
        self.status_var.set(
            f"Displayed {len(rows)} result(s), sorted by ascending total energy."
        )

    def _collect_params(self) -> GenerationParams:
        if self.structure_path is None:
            raise ValueError("select a structure file first")
        species = magnetic_species_from_rows(self.magnetization_rows)
        if not species:
            raise ValueError("select at least one magnetic element in the table")
        method = self.method_var.get()
        moment = moment_text_from_rows(self.magnetization_rows, method)
        up_species, down_species = species_roles_from_rows(self.magnetization_rows)
        coordination_labels = tuple(
            (row.element, row.coordination, row.label)
            for row in self.magnetization_rows
            if row.use
            and row.coordination is not None
            and row.label not in {"", "-"}
        )
        layer_direction: tuple[float, float, float] | None = None
        direction_text = self.layer_direction_var.get().strip()
        if method == "layer" and direction_text:
            direction_values = [
                float(value) for value in direction_text.replace(",", " ").split()
            ]
            if len(direction_values) != 3:
                raise ValueError("layer direction must contain exactly three numbers")
            layer_direction = (
                direction_values[0],
                direction_values[1],
                direction_values[2],
            )
        q_vector: tuple[float, float, float] | None = None
        afm_type = self.afm_type_var.get()
        if method == "propagation-vector" and afm_type == "custom":
            values = [
                float(value)
                for value in self.q_vector_var.get().replace(",", " ").split()
            ]
            if len(values) != 3:
                raise ValueError("q-vector must contain exactly three numbers")
            q_vector = (values[0], values[1], values[2])
        selected_afm_type = (
            afm_type if method == "propagation-vector" and afm_type != "custom" else None
        )
        if method == "by-coordination":
            up_coordination = tuple(
                int(value)
                for value in self._split_words(self.up_coordination_var.get())
            )
            down_coordination = tuple(
                int(value)
                for value in self._split_words(self.down_coordination_var.get())
            )
            coordination_tolerance = int(self.coordination_tolerance_var.get())
        else:
            up_coordination = (6,)
            down_coordination = (4,)
            coordination_tolerance = 0
        if method == "graph-coloring":
            max_colors = int(self.max_colors_var.get())
            color_spins = self.color_spins_var.get().strip() or None
            balance_colors = self.balance_colors_var.get()
        else:
            max_colors = 4
            color_spins = None
            balance_colors = False
        seed = int(self.seed_var.get()) if method in {"random", "graph-coloring"} else 0
        cutoff: str | float = (
            "auto" if self.auto_cutoff_var.get() else float(self.cutoff_var.get())
        )
        return GenerationParams(
            structure_path=self.structure_path,
            magnetic_species=species,
            method=method,
            moment=moment,
            site_moment_file=self.site_moment_file_var.get().strip() or None,
            site_comments=self.site_comments_var.get(),
            axis=self.axis_var.get(),
            layer_direction=layer_direction,
            fractional_layers=self.fractional_layers_var.get(),
            cutoff=cutoff,
            layer_tolerance=float(self.tolerance_var.get()),
            slab=self.slab_var.get(),
            q_vector=q_vector,
            afm_type=selected_afm_type,
            allow_frustrated=self.allow_frustrated_var.get(),
            up_species=up_species,
            down_species=down_species,
            anion_species=self._split_words(self.anion_species_var.get()),
            anion_cutoff=self.anion_cutoff_var.get().strip() or "auto",
            up_coordination=up_coordination,
            down_coordination=down_coordination,
            coordination_tolerance=coordination_tolerance,
            coordination_labels=coordination_labels,
            max_colors=max_colors,
            color_spins=color_spins,
            balance_colors=balance_colors,
            seed=seed,
            color_mode=_COLOR_MODES[self.color_mode_var.get()],
        )

    def _parameter_changed(self, *_: object) -> None:
        if not self._traces_ready:
            return
        self._sync_cutoff_state()
        if self.viewing_spin_path is not None:
            self.viewing_spin_path = None
            self.mode_var.set("generation mode (spin-file view ended)")
        self._schedule_live_update()

    def _sync_cutoff_state(self) -> None:
        state = "disabled" if self.auto_cutoff_var.get() else "normal"
        self.cutoff_entry.configure(state=state)

    def _toggle_live_update(self) -> None:
        if self.live_update_var.get():
            self._schedule_live_update()
        elif self._live_after_id is not None:
            self.root.after_cancel(self._live_after_id)
            self._live_after_id = None
            self.status_var.set("Live update paused.")

    def _schedule_live_update(self) -> None:
        if (
            not self._traces_ready
            or not self.live_update_var.get()
            or self.structure_path is None
            or self.viewing_spin_path is not None
        ):
            return
        if self._live_after_id is not None:
            self.root.after_cancel(self._live_after_id)
        self._live_after_id = self.root.after(400, self._run_live_update)
        status = "Preview update scheduled..."
        if self._coordination_fallback:
            status = f"{self._coordination_fallback} Preview update scheduled..."
        self.status_var.set(status)

    def _run_live_update(self) -> None:
        self._live_after_id = None
        self._generate(show_dialog=False)

    def _generate_explicit(self) -> None:
        self._generate(show_dialog=True)

    def _generate(self, *, show_dialog: bool) -> None:
        if self._live_after_id is not None:
            self.root.after_cancel(self._live_after_id)
            self._live_after_id = None
        try:
            params = self._collect_params()
            result = run_generation(params)
            camera = None if self._reset_camera else self._capture_camera()
            figure = create_spin_figure(
                result.structure,
                result.spins,
                show_indices=self.show_atom_indices_var.get(),
                color_mode=params.color_mode,
            )
            self._restore_camera(figure, camera)
        except Exception as exc:
            self.status_var.set(f"Preview not updated: {exc}")
            if show_dialog:
                self.deps.messagebox.showerror("Generation failed", str(exc))
            return

        self._reset_camera = False
        self.current_result = result
        self.current_structure = result.structure
        self.current_spins = result.spins
        self.current_block = result.block
        self.viewing_spin_path = None
        self.mode_var.set("generation mode")
        analysis = format_analysis(result.report)
        report_warnings = set(result.report.get("warnings", []))
        extra_warnings = [
            warning for warning in result.warnings if warning not in report_warnings
        ]
        if extra_warnings:
            analysis += "\n\nWarnings:\n" + "\n".join(extra_warnings)
        self._set_text(self.analysis_text, analysis)
        self._set_text(self.spin_text, result.block)
        self._set_site_table(result)
        self._replace_figure(figure)
        self._set_exports_enabled(True)
        status = f"Generated {len(result.spins)} magnetic site(s)."
        if result.warnings:
            status += " " + " | ".join(result.warnings)
        if self._coordination_use_note:
            status += " " + self._coordination_use_note
        self.status_var.set(status)

    def _open_spin_file(self) -> None:
        if self._live_after_id is not None:
            self.root.after_cancel(self._live_after_id)
            self._live_after_id = None
        if self.structure_path is None and not self._choose_structure(schedule=False):
            return
        spin_path = self.deps.filedialog.askopenfilename(
            title="Open DM.InitSpin file",
            filetypes=[("FDF files", "*.fdf"), ("All files", "*.*")],
        )
        if not spin_path or self.structure_path is None:
            return
        try:
            structure = read_structure(self.structure_path, slab=self.slab_var.get())
            cutoff: str | float = (
                "auto" if self.auto_cutoff_var.get() else float(self.cutoff_var.get())
            )
            loaded = load_spin_file(
                spin_path,
                structure,
                cutoff=cutoff,
                axis=self.axis_var.get(),
                layer_tolerance=float(self.tolerance_var.get()),
                site_comments=self.site_comments_var.get(),
            )
            camera = None if self._reset_camera else self._capture_camera()
            figure = create_spin_figure(
                structure,
                loaded.spins,
                show_indices=self.show_atom_indices_var.get(),
                color_mode=_COLOR_MODES[self.color_mode_var.get()],
            )
            self._restore_camera(figure, camera)
        except Exception as exc:
            self.status_var.set(f"Spin file not opened: {exc}")
            self.deps.messagebox.showerror("Open spin file failed", str(exc))
            return

        self._reset_camera = False
        self.current_result = loaded
        self.current_structure = structure
        self.current_spins = loaded.spins
        self.current_block = loaded.block
        self.viewing_spin_path = Path(spin_path)
        self.mode_var.set(f"viewing: {self.viewing_spin_path.name}")
        analysis = format_validation(loaded.validation)
        if loaded.warnings:
            analysis += "\n" + "\n".join(f"WARNING: {w}" for w in loaded.warnings)
        self._set_text(self.analysis_text, analysis)
        self._set_text(self.spin_text, loaded.block)
        self._set_site_table(loaded)
        self._replace_figure(figure)
        self._set_exports_enabled(True)
        self.status_var.set(f"Viewing spin file: {self.viewing_spin_path.name}")

    def _show_atom_indices_changed(self) -> None:
        if (
            self.current_structure is None
            or self.current_result is None
            or self.current_result.structure is not self.current_structure
            or not self.current_spins
        ):
            return
        try:
            camera = self._capture_camera()
            figure = create_spin_figure(
                self.current_structure,
                self.current_spins,
                show_indices=self.show_atom_indices_var.get(),
                color_mode=_COLOR_MODES[self.color_mode_var.get()],
            )
            self._restore_camera(figure, camera)
            self._replace_figure(figure)
        except Exception as exc:
            self.status_var.set(f"Preview not updated: {exc}")

    def _capture_camera(self) -> _CameraState | None:
        if self.figure is None or not self.figure.axes:
            return None
        axis = next(
            (candidate for candidate in self.figure.axes if hasattr(candidate, "elev")),
            None,
        )
        if axis is None:
            return None
        return _CameraState(
            elev=float(axis.elev),
            azim=float(axis.azim),
            xlim=tuple(float(value) for value in axis.get_xlim()),
            ylim=tuple(float(value) for value in axis.get_ylim()),
            zlim=tuple(float(value) for value in axis.get_zlim()),
        )

    @staticmethod
    def _restore_camera(figure: Any, camera: _CameraState | None) -> None:
        if camera is None or not figure.axes:
            return
        axis = next(
            (candidate for candidate in figure.axes if hasattr(candidate, "view_init")),
            None,
        )
        if axis is None:
            return
        axis.view_init(elev=camera.elev, azim=camera.azim)
        axis.set_xlim(camera.xlim)
        axis.set_ylim(camera.ylim)
        axis.set_zlim(camera.zlim)

    def _replace_figure(self, figure: Any) -> None:
        old_figure = self.figure
        if old_figure is not None:
            # Figure.clear() notifies its toolbar, so clear while the old Tk
            # widgets still exist and only then destroy them.
            old_figure.clear()
        if self.toolbar is not None:
            self.toolbar.destroy()
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
        self.figure = figure
        self.canvas = self.deps.FigureCanvasTkAgg(figure, master=self.canvas_host)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.toolbar = self.deps.NavigationToolbar2Tk(
            self.canvas, self.toolbar_host, pack_toolbar=False
        )
        self.toolbar.update()
        self.toolbar.pack(fill="x")

    @staticmethod
    def _set_text(widget: Any, value: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", value)
        widget.configure(state="disabled")

    def _set_site_table(self, result: GenerationResult | SpinFileResult) -> None:
        for item in self.sites_tree.get_children():
            self.sites_tree.delete(item)
        for row in site_assignment_rows(result):
            self.sites_tree.insert(
                "",
                "end",
                values=(
                    row["atom"],
                    row["element"],
                    row["CN"],
                    row["sublattice"],
                    row["sign"],
                    f"{float(row['moment']):g}",
                ),
            )
        self.sites_summary_var.set(site_assignment_summary(result))

    def _set_exports_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for button in self.export_buttons:
            button.configure(state=state)

    def _save_spin(self) -> None:
        if not self.current_block:
            return
        destination = self.deps.filedialog.asksaveasfilename(
            title="Save DM.InitSpin block",
            defaultextension=".fdf",
            initialfile="afm_spin.fdf",
            filetypes=[("FDF files", "*.fdf"), ("All files", "*.*")],
        )
        if destination:
            self._run_export(
                lambda: export_spin_block(self.current_block, destination),
                "Saved spin block",
            )

    def _export_patched(self) -> None:
        if not self.current_block:
            return
        base = self.deps.filedialog.askopenfilename(
            title="Select base SIESTA input",
            filetypes=[("FDF files", "*.fdf"), ("All files", "*.*")],
        )
        if not base:
            return
        destination = self.deps.filedialog.asksaveasfilename(
            title="Export patched SIESTA input",
            defaultextension=".fdf",
            initialfile="input_afm.fdf",
            filetypes=[("FDF files", "*.fdf"), ("All files", "*.*")],
        )
        if destination:
            self._run_export(
                lambda: export_patched_input(base, self.current_block, destination),
                "Exported patched SIESTA input",
            )

    def _export_complete_input(self) -> None:
        if self.current_result is None:
            return
        destination = self.deps.filedialog.asksaveasfilename(
            title="Export complete SIESTA starting input",
            defaultextension=".fdf",
            initialfile="input.fdf",
            filetypes=[("FDF files", "*.fdf"), ("All files", "*.*")],
        )
        if not destination:
            return
        document = complete_input_document(self.current_result)
        self.deps.messagebox.showwarning(
            "Complete input is a starting point", "\n\n".join(document.warnings)
        )
        self._run_export(
            lambda: export_complete_input(self.current_result, destination),
            "Exported complete SIESTA starting input",
        )

    def _export_structure(self) -> None:
        if self.current_structure is None:
            return
        destination = self.deps.filedialog.asksaveasfilename(
            title="Export structure with moments",
            defaultextension=".xyz",
            initialfile="structure_with_moments.xyz",
            filetypes=[
                ("Extended XYZ", "*.xyz"),
                ("CIF", "*.cif"),
                ("All files", "*.*"),
            ],
        )
        if destination:
            self._run_export(
                lambda: export_structure_with_moments(
                    self.current_structure, self.current_spins, destination
                ),
                "Exported structure with moments",
            )

    def _run_export(self, operation: Any, success: str) -> None:
        try:
            destination = operation()
        except Exception as exc:
            self.status_var.set(f"Export failed: {exc}")
            self.deps.messagebox.showerror("Export failed", str(exc))
            return
        self.status_var.set(f"{success}: {destination}")

    def _close(self) -> None:
        self._unbind_controls_mousewheel()
        if self._live_after_id is not None:
            self.root.after_cancel(self._live_after_id)
            self._live_after_id = None
        if self._table_after_id is not None:
            self.root.after_cancel(self._table_after_id)
            self._table_after_id = None
        if self.figure is not None:
            self.figure.clear()
        self.root.destroy()


def main() -> int:
    """Launch the local Tk desktop application."""

    dependencies = _load_gui_dependencies()
    try:
        root = dependencies.tk.Tk()
    except dependencies.tk.TclError as exc:
        raise RuntimeError(
            "cannot start the desktop GUI: no display available or Tk failed to initialize"
        ) from exc
    DesktopApp(root, dependencies)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
