"""Tkinter desktop interface and testable GUI workflow controller."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .fdf_writer import render_dm_init_spin
from .io import read_structure
from .ordering import SpinAssignment
from .structure import Structure
from .validation import analyze_structure, format_analysis
from .visualize import create_spin_figure
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
    structure: Structure
    magnetic_indices: list[int]
    assignment: SpinAssignment
    spins: dict[int, float]
    block: str
    report: dict[str, object]
    warnings: tuple[str, ...]


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
        structure,
        indices,
        assignment,
        spins,
        block,
        report,
        tuple(assignment.warnings),
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


def _load_gui_dependencies() -> _GuiDependencies:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except ImportError as exc:
        raise RuntimeError(
            "the desktop GUI requires Tkinter and matplotlib; install Python with "
            f"Tk support, then run '{_GUI_INSTALL_HINT}'"
        ) from exc
    canvas, toolbar = _load_matplotlib_tk_backend()
    return _GuiDependencies(tk, ttk, filedialog, messagebox, canvas, toolbar)


class DesktopApp:
    """Thin Tk shell around the widget-independent controller."""

    def __init__(self, root: Any, dependencies: _GuiDependencies) -> None:
        self.root = root
        self.deps = dependencies
        self.tk = dependencies.tk
        self.ttk = dependencies.ttk
        self.structure_path: Path | None = None
        self.result: GenerationResult | None = None
        self.figure: Any | None = None
        self.canvas: Any | None = None
        self.toolbar: Any | None = None

        root.title("SIESTA AFM initial-spin generator")
        root.geometry("1450x900")
        root.minsize(1050, 700)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)
        root.protocol("WM_DELETE_WINDOW", self._close)
        self._variables()
        self._controls()
        self._results()

    def _variables(self) -> None:
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
        self.status_var = tk.StringVar(value="Select a structure file to begin.")

    def _controls(self) -> None:
        panel = self.ttk.Frame(self.root, padding=10)
        panel.grid(row=0, column=0, sticky="nsew")
        panel.columnconfigure(1, weight=1)
        self.ttk.Button(
            panel, text="Open structure...", command=self._choose_structure
        ).grid(row=0, column=0, columnspan=2, sticky="ew")
        self.ttk.Label(panel, textvariable=self.file_var, wraplength=300).grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=(3, 10)
        )
        row = 2
        row = self._entry(panel, row, "Magnetic species", self.species_var)
        row = self._combo(panel, row, "Method", self.method_var, _METHODS)
        row = self._entry(panel, row, "Moment magnitude", self.moment_var)
        row = self._combo(panel, row, "Layer axis", self.axis_var, ["z", "x", "y"])
        self.ttk.Checkbutton(
            panel, text="Automatic first-shell cutoff", variable=self.auto_cutoff_var
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=3)
        row += 1
        row = self._entry(panel, row, "Neighbor cutoff (Å)", self.cutoff_var)
        row = self._entry(panel, row, "Layer tolerance (Å)", self.tolerance_var)
        self.ttk.Checkbutton(
            panel, text="Slab (periodic xy)", variable=self.slab_var
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=3)
        row += 1
        row = self._entry(panel, row, "q-vector", self.q_vector_var)
        self.ttk.Checkbutton(
            panel,
            text="Allow frustrated heuristic",
            variable=self.allow_frustrated_var,
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=3)
        row += 1
        row = self._combo(
            panel, row, "Color mode", self.color_mode_var, list(_COLOR_MODES)
        )
        self.ttk.Button(panel, text="Generate", command=self._generate).grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=(10, 3)
        )
        row += 1
        self.save_button = self.ttk.Button(
            panel, text="Save FDF...", command=self._save, state="disabled"
        )
        self.save_button.grid(row=row, column=0, columnspan=2, sticky="ew")

    def _results(self) -> None:
        panel = self.ttk.Frame(self.root, padding=(0, 10, 10, 10))
        panel.grid(row=0, column=1, sticky="nsew")
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(0, weight=3)
        panel.rowconfigure(1, weight=2)
        preview = self.ttk.LabelFrame(panel, text="Interactive 3D preview")
        preview.grid(row=0, column=0, sticky="nsew")
        self.canvas_host = self.ttk.Frame(preview)
        self.canvas_host.pack(fill="both", expand=True)
        self.toolbar_host = self.ttk.Frame(preview)
        self.toolbar_host.pack(fill="x")
        notebook = self.ttk.Notebook(panel)
        notebook.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
        analysis = self.ttk.Frame(notebook)
        spin = self.ttk.Frame(notebook)
        notebook.add(analysis, text="Analysis")
        notebook.add(spin, text="DM.InitSpin")
        self.analysis_text = self._text(analysis)
        self.spin_text = self._text(spin)
        self.ttk.Label(
            self.root, textvariable=self.status_var, relief="sunken", anchor="w"
        ).grid(row=1, column=0, columnspan=2, sticky="ew")

    def _entry(self, parent: Any, row: int, label: str, variable: Any) -> int:
        self.ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        self.ttk.Entry(parent, textvariable=variable).grid(
            row=row, column=1, sticky="ew", padx=(6, 0), pady=3
        )
        return row + 1

    def _combo(
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

    def _text(self, parent: Any) -> Any:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        widget = self.tk.Text(parent, wrap="word")
        widget.grid(row=0, column=0, sticky="nsew")
        widget.configure(state="disabled")
        return widget

    def _choose_structure(self) -> None:
        path = self.deps.filedialog.askopenfilename(
            filetypes=[
                ("Structure files", "*.cif *.xyz *.fdf *.xv *.XV *.vasp"),
                ("POSCAR / CONTCAR", "POSCAR CONTCAR"),
                ("All files", "*.*"),
            ]
        )
        if path:
            self.structure_path = Path(path)
            self.file_var.set(path)
            self.status_var.set(f"Loaded structure: {self.structure_path.name}")

    def _params(self) -> GenerationParams:
        if self.structure_path is None:
            raise ValueError("select a structure file first")
        species = tuple(self.species_var.get().replace(",", " ").split())
        q_vector = None
        if self.method_var.get() == "propagation-vector":
            values = [float(v) for v in self.q_vector_var.get().split()]
            if len(values) != 3:
                raise ValueError("q-vector must contain exactly three numbers")
            q_vector = (values[0], values[1], values[2])
        cutoff: str | float = (
            "auto" if self.auto_cutoff_var.get() else float(self.cutoff_var.get())
        )
        return GenerationParams(
            self.structure_path,
            species,
            self.method_var.get(),
            str(float(self.moment_var.get())),
            self.axis_var.get(),
            cutoff,
            float(self.tolerance_var.get()),
            self.slab_var.get(),
            q_vector,
            self.allow_frustrated_var.get(),
            _COLOR_MODES[self.color_mode_var.get()],
        )

    def _generate(self) -> None:
        try:
            params = self._params()
            result = run_generation(params)
            figure = create_spin_figure(
                result.structure,
                result.spins,
                show_indices=True,
                color_mode=params.color_mode,
            )
        except Exception as exc:
            self.status_var.set(f"Generation failed: {exc}")
            self.deps.messagebox.showerror("Generation failed", str(exc))
            return
        if self.figure is not None:
            self.figure.clear()
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
        self.result = result
        analysis = format_analysis(result.report)
        if result.warnings:
            analysis += "\n\nWarnings:\n" + "\n".join(result.warnings)
        self._set_text(self.analysis_text, analysis)
        self._set_text(self.spin_text, result.block)
        self.save_button.configure(state="normal")
        self.status_var.set(f"Generated {len(result.spins)} magnetic site(s).")

    @staticmethod
    def _set_text(widget: Any, value: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", value)
        widget.configure(state="disabled")

    def _save(self) -> None:
        if self.result is None:
            return
        path = self.deps.filedialog.asksaveasfilename(
            defaultextension=".fdf",
            initialfile="afm_spin.fdf",
            filetypes=[("FDF files", "*.fdf"), ("All files", "*.*")],
        )
        if path:
            Path(path).write_text(self.result.block, encoding="utf-8")
            self.status_var.set(f"Saved spin block: {path}")

    def _close(self) -> None:
        if self.figure is not None:
            self.figure.clear()
        self.root.destroy()


def main() -> int:
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
