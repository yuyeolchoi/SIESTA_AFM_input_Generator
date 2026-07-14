"""Tkinter desktop interface and testable GUI workflow controllers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .fdf_writer import patch_fdf_text, render_dm_init_spin
from .io import parse_dm_init_spin, read_structure
from .ordering import SpinAssignment
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
    "checkerboard",
    "propagation-vector",
]
_COLOR_MODES = {"spin sign": "sign", "spin value": "value"}


@dataclass(frozen=True, slots=True)
class GenerationParams:
    """Widget-independent parameters for one AFM generation run."""

    structure_path: str | Path
    magnetic_species: tuple[str, ...]
    method: str = "layer"
    moment: str = "0.5"
    axis: str = "z"
    cutoff: str | float = "auto"
    layer_tolerance: float = 0.25
    slab: bool = False
    q_vector: tuple[float, float, float] | None = None
    allow_frustrated: bool = False
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
        axis=params.axis,
        layer_tolerance=params.layer_tolerance,
        cutoff=params.cutoff,
        allow_frustrated=params.allow_frustrated,
        q_vector=params.q_vector,
    )
    block = render_dm_init_spin(
        spins,
        method=assignment.method,
        magnetic_species=params.magnetic_species,
        metadata=assignment.metadata,
    )
    report = analyze_structure(
        structure,
        indices,
        magnetic_species=params.magnetic_species,
        cutoff=params.cutoff,
        axis=params.axis,
        layer_tolerance=params.layer_tolerance,
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
        self._reset_camera = True
        self._traces_ready = False

        root.title("SIESTA AFM initial-spin generator")
        root.geometry("1450x900")
        root.minsize(1050, 700)
        root.protocol("WM_DELETE_WINDOW", self._close)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)

        self._create_variables()
        self._build_controls()
        self._build_results()
        self._install_traces()
        self._traces_ready = True
        self._sync_cutoff_state()

    def _create_variables(self) -> None:
        tk = self.tk
        self.file_var = tk.StringVar(value="No structure selected")
        self.species_var = tk.StringVar(value="Cu")
        self.method_var = tk.StringVar(value="layer")
        self.moment_var = tk.StringVar(value="0.5")
        self.axis_var = tk.StringVar(value="z")
        self.auto_cutoff_var = tk.BooleanVar(value=True)
        self.cutoff_var = tk.StringVar(value="3.2")
        self.tolerance_var = tk.StringVar(value="0.25")
        self.slab_var = tk.BooleanVar(value=False)
        self.q_vector_var = tk.StringVar(value="0.5 0.5 0.5")
        self.allow_frustrated_var = tk.BooleanVar(value=False)
        self.color_mode_var = tk.StringVar(value="spin sign")
        self.live_update_var = tk.BooleanVar(value=True)
        self.mode_var = tk.StringVar(value="generation mode")
        self.status_var = tk.StringVar(value="Select a structure file to begin.")

    def _build_controls(self) -> None:
        ttk = self.ttk
        panel = ttk.Frame(self.root, padding=10)
        panel.grid(row=0, column=0, sticky="nsew")
        panel.columnconfigure(1, weight=1)

        ttk.Label(panel, text="Structure").grid(row=0, column=0, sticky="w")
        ttk.Button(
            panel, text="Open structure...", command=self._choose_structure
        ).grid(row=0, column=1, sticky="ew", padx=(6, 0))
        ttk.Label(panel, textvariable=self.file_var, wraplength=300).grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=(2, 10)
        )

        row = 2
        row = self._entry_row(panel, row, "Magnetic species", self.species_var)
        row = self._combo_row(panel, row, "Method", self.method_var, _METHODS)
        ttk.Label(panel, text="Moment magnitude").grid(row=row, column=0, sticky="w")
        ttk.Spinbox(
            panel,
            textvariable=self.moment_var,
            from_=0.0,
            to=20.0,
            increment=0.1,
            width=12,
        ).grid(row=row, column=1, sticky="ew", padx=(6, 0), pady=3)
        row += 1
        row = self._combo_row(panel, row, "Layer axis", self.axis_var, ["z", "x", "y"])
        ttk.Checkbutton(
            panel, text="Automatic first-shell cutoff", variable=self.auto_cutoff_var
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=3)
        row += 1
        ttk.Label(panel, text="Neighbor cutoff (Å)").grid(row=row, column=0, sticky="w")
        self.cutoff_entry = ttk.Entry(panel, textvariable=self.cutoff_var)
        self.cutoff_entry.grid(row=row, column=1, sticky="ew", padx=(6, 0), pady=3)
        row += 1
        row = self._entry_row(panel, row, "Layer tolerance (Å)", self.tolerance_var)
        ttk.Checkbutton(panel, text="Slab (periodic xy)", variable=self.slab_var).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=3
        )
        row += 1
        row = self._entry_row(panel, row, "q-vector", self.q_vector_var)
        ttk.Checkbutton(
            panel,
            text="Allow frustrated heuristic",
            variable=self.allow_frustrated_var,
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=3)
        row += 1
        row = self._combo_row(
            panel,
            row,
            "Color mode",
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
        panel = ttk.Frame(self.root, padding=(0, 10, 10, 10))
        panel.grid(row=0, column=1, sticky="nsew")
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
        spin_frame = ttk.Frame(notebook)
        notebook.add(analysis_frame, text="Analysis")
        notebook.add(spin_frame, text="DM.InitSpin")
        self.analysis_text = self._readonly_text(analysis_frame)
        self.spin_text = self._readonly_text(spin_frame)

        status = ttk.Label(
            self.root,
            textvariable=self.status_var,
            relief="sunken",
            anchor="w",
            padding=(6, 3),
        )
        status.grid(row=1, column=0, columnspan=2, sticky="ew")

    def _entry_row(self, parent: Any, row: int, label: str, variable: Any) -> int:
        self.ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        self.ttk.Entry(parent, textvariable=variable).grid(
            row=row, column=1, sticky="ew", padx=(6, 0), pady=3
        )
        return row + 1

    def _combo_row(
        self,
        parent: Any,
        row: int,
        label: str,
        variable: Any,
        values: Sequence[str],
    ) -> int:
        self.ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        self.ttk.Combobox(
            parent, textvariable=variable, values=list(values), state="readonly"
        ).grid(row=row, column=1, sticky="ew", padx=(6, 0), pady=3)
        return row + 1

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
            self.species_var,
            self.method_var,
            self.moment_var,
            self.axis_var,
            self.auto_cutoff_var,
            self.cutoff_var,
            self.tolerance_var,
            self.slab_var,
            self.q_vector_var,
            self.allow_frustrated_var,
            self.color_mode_var,
        )
        for variable in variables:
            variable.trace_add("write", self._parameter_changed)

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
        self.structure_path = Path(path)
        self.file_var.set(str(self.structure_path))
        self.viewing_spin_path = None
        self.mode_var.set("generation mode")
        self._reset_camera = True
        self.status_var.set(f"Loaded structure: {self.structure_path.name}")
        if schedule:
            self._schedule_live_update()
        return True

    def _collect_params(self) -> GenerationParams:
        if self.structure_path is None:
            raise ValueError("select a structure file first")
        species = tuple(
            value for value in self.species_var.get().replace(",", " ").split() if value
        )
        method = self.method_var.get()
        q_vector: tuple[float, float, float] | None = None
        if method == "propagation-vector":
            values = [
                float(value)
                for value in self.q_vector_var.get().replace(",", " ").split()
            ]
            if len(values) != 3:
                raise ValueError("q-vector must contain exactly three numbers")
            q_vector = (values[0], values[1], values[2])
        cutoff: str | float = (
            "auto" if self.auto_cutoff_var.get() else float(self.cutoff_var.get())
        )
        return GenerationParams(
            structure_path=self.structure_path,
            magnetic_species=species,
            method=method,
            moment=str(float(self.moment_var.get())),
            axis=self.axis_var.get(),
            cutoff=cutoff,
            layer_tolerance=float(self.tolerance_var.get()),
            slab=self.slab_var.get(),
            q_vector=q_vector,
            allow_frustrated=self.allow_frustrated_var.get(),
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
        self.status_var.set("Preview update scheduled...")

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
        if result.warnings:
            analysis += "\n\nWarnings:\n" + "\n".join(result.warnings)
        self._set_text(self.analysis_text, analysis)
        self._set_text(self.spin_text, result.block)
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
