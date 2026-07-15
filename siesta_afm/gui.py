"""Tkinter desktop interface and testable GUI workflow controllers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .fdf_writer import patch_fdf_text, render_dm_init_spin
from .io import parse_dm_init_spin, read_structure
from .magnetic_sites import (
    DEFAULT_ELEMENT_MOMENTS,
    select_magnetic_sites,
)
from .neighbors import classify_coordination_geometry
from .ordering import SpinAssignment, analyze_coordination_sites
from .structure import Structure
from .validation import (
    ValidationReport,
    analyze_structure,
    format_analysis,
    format_validation,
    validate_spins,
)
from .visualize import create_spin_figure, plot_spin_pattern
from .workflows import generate_assignment


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
_COLOR_MODES = {"spin sign": "sign", "spin value": "value"}
_LEFT_PANEL_MIN_WIDTH = 600


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
                )
            )
        return rows

    selected = [
        index
        for index, symbol in enumerate(structure.symbols)
        if use_by_element.get(symbol.lower(), False)
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
        element for element in counts if use_by_element.get(element.lower(), False)
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
                True,
                element,
                labels_by_site.get((normalized, coordination), inferred_label),
                coordination,
                value or "",
                count,
                "-",
            )
        )
    for element, count in counts.items():
        if not use_by_element.get(element.lower(), False):
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
        self.current_result: GenerationResult | None = None
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
        self._pane_width_initialized = False

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
        self.live_update_var = tk.BooleanVar(value=True)
        self.mode_var = tk.StringVar(value="generation mode")
        self.status_var = tk.StringVar(value="Select a structure file to begin.")

    def _build_controls(self) -> None:
        ttk = self.ttk
        self.main_pane = ttk.PanedWindow(self.root, orient="horizontal")
        self.main_pane.grid(row=0, column=0, sticky="nsew")
        self.main_pane.bind(
            "<Map>", lambda _event: self.root.after_idle(self._set_initial_pane_width)
        )
        container = ttk.Frame(self.main_pane)
        self.main_pane.add(container, weight=0)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)
        canvas = self.tk.Canvas(container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        panel = ttk.Frame(canvas, padding=10)
        panel_window = canvas.create_window((0, 0), window=panel, anchor="nw")
        panel.bind(
            "<Configure>",
            lambda _event: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.bind(
            "<Configure>",
            lambda event: canvas.itemconfigure(panel_window, width=event.width),
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
            text="Double-click use, label, value, or role to edit. CN and count are read-only.",
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

        actions = ttk.LabelFrame(panel, text="Generate / View", padding=6)
        actions.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(8, 4))
        actions.columnconfigure((0, 1), weight=1)
        ttk.Button(actions, text="Generate", command=self._generate_explicit).grid(
            row=0, column=0, sticky="ew", padx=(0, 3)
        )
        ttk.Button(
            actions, text="Open spin file...", command=self._open_spin_file
        ).grid(row=0, column=1, sticky="ew", padx=(3, 0))
        row += 1

        exports = ttk.LabelFrame(panel, text="Export", padding=6)
        exports.grid(row=row, column=0, columnspan=2, sticky="ew", pady=4)
        exports.columnconfigure(0, weight=1)
        self.export_buttons = [
            ttk.Button(exports, text="DM.InitSpin block...", command=self._save_spin),
            ttk.Button(
                exports, text="Patched SIESTA input...", command=self._export_patched
            ),
            ttk.Button(
                exports,
                text="Structure with moments...",
                command=self._export_structure,
            ),
        ]
        for number, button in enumerate(self.export_buttons):
            button.grid(row=number, column=0, sticky="ew", pady=2)
            button.configure(state="disabled")

    def _build_results(self) -> None:
        ttk = self.ttk
        panel = ttk.Frame(self.main_pane, padding=(0, 10, 10, 10))
        self.main_pane.add(panel, weight=1)
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(1, weight=3)
        panel.rowconfigure(2, weight=2)

        ttk.Label(panel, textvariable=self.mode_var).grid(row=0, column=0, sticky="w")
        preview = ttk.LabelFrame(panel, text="Interactive 3D preview", padding=4)
        preview.grid(row=1, column=0, sticky="nsew", pady=(4, 6))
        preview.columnconfigure(0, weight=1)
        preview.rowconfigure(0, weight=1)
        self.canvas_host = ttk.Frame(preview)
        self.canvas_host.grid(row=0, column=0, sticky="nsew")
        self.toolbar_host = ttk.Frame(preview)
        self.toolbar_host.grid(row=1, column=0, sticky="ew")

        notebook = ttk.Notebook(panel)
        notebook.grid(row=2, column=0, sticky="nsew")
        analysis_frame = ttk.Frame(notebook)
        sites_frame = ttk.Frame(notebook)
        spin_frame = ttk.Frame(notebook)
        notebook.add(analysis_frame, text="Analysis")
        notebook.add(sites_frame, text="Sites")
        notebook.add(spin_frame, text="DM.InitSpin")
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

        status = ttk.Label(
            self.root,
            textvariable=self.status_var,
            relief="sunken",
            anchor="w",
            padding=(6, 3),
        )
        status.grid(row=1, column=0, sticky="ew")

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
            new_value = not row.use
            for candidate in self.magnetization_rows:
                if candidate.element.lower() == row.element.lower():
                    candidate.use = new_value
                    if not new_value:
                        candidate.label = ""
                        candidate.coordination = None
                        candidate.value = ""
                        candidate.role = "-"
            self._table_changed(refresh=True)
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
                show_indices=True,
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
                show_indices=True,
                color_mode=_COLOR_MODES[self.color_mode_var.get()],
            )
            self._restore_camera(figure, camera)
        except Exception as exc:
            self.status_var.set(f"Spin file not opened: {exc}")
            self.deps.messagebox.showerror("Open spin file failed", str(exc))
            return

        self._reset_camera = False
        self.current_result = None
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
