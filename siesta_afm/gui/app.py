"""Tkinter desktop interface and testable GUI workflow controllers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..io import read_structure
from ..structure import Structure
from ..validation import format_analysis, format_validation
from ..visualize import create_spin_figure, element_spin_counts
from .controllers import (
    GenerationParams,
    GenerationResult,
    MagnetizationRow,
    SpinFileResult,
    _AUTO_SHOW_BONDS_MAX_ATOMS,
    _AUTO_SHOW_INDICES_MAX_ATOMS,
    _BATCH_METHODS,
    _CANDIDATE_COLUMNS,
    _COLOR_MODES,
    _GUI_INSTALL_HINT,
    _LEFT_PANEL_MIN_WIDTH,
    _METHODS,
    _NEAR_GROUND_ENERGY_WINDOW_EV,
    _PANE_SASH_MARGIN,
    _RESULT_COLUMNS,
    _RESULTS_NOTEBOOK_MIN_HEIGHT,
    _element_counts,
    _split_words,
    batch_moment_text_from_rows,
    candidate_table_rows,
    collect_or_load_results,
    complete_input_document,
    coordination_labels_from_rows,
    coordination_numbers_from_result,
    equivalent_cli_options,
    export_complete_input,
    export_patched_input,
    export_spin_block,
    export_structure_with_moments,
    load_spin_file,
    magnetic_species_from_rows,
    moment_text_from_rows,
    prepare_job_folders,
    results_table_rows,
    run_candidate_generation,
    run_generation,
    safe_magnetization_rows_from_structure,
    site_assignment_rows,
    site_assignment_summary,
    toggle_magnetization_use,
    workflow_kwargs_from_inputs,
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
        self.spin_element_vars: dict[str, Any] = {}
        self.spin_element_checkbuttons: dict[str, Any] = {}
        self.spin_element_labels: dict[str, Any] = {}
        self.spin_coordination_vars: dict[int, Any] = {}
        self.spin_coordination_checkbuttons: dict[int, Any] = {}
        self.spin_coordination_widgets: list[Any] = []
        self.spin_coordination_numbers: dict[int, int] = {}
        self._coordination_fallback: str | None = None
        self._coordination_use_note: str | None = None
        self._sites_sort_by_cn = False
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
        self.show_bonds_var = tk.BooleanVar(value=False)
        self.bond_radius_scale_var = tk.StringVar(value="1.0")
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
                "Double-click use, label, value, or role to edit (row must be "
                "checked). Label edits apply only when Method is "
                "by-coordination; role edits only when Method is by-species. "
                "CN and count are computed from the structure and read-only. "
                "In by-coordination mode, unchecking one coordination site "
                "keeps the element selected; set moment 0 or use "
                "--exclude-atoms to fully exclude those atoms."
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
        preview_controls = ttk.Frame(preview)
        preview_controls.grid(row=0, column=0, sticky="ew")
        ttk.Checkbutton(
            preview_controls,
            text="Show atom indices",
            variable=self.show_atom_indices_var,
            command=self._show_atom_indices_changed,
        ).grid(row=0, column=0, sticky="w", padx=2, pady=(0, 3))
        ttk.Checkbutton(
            preview_controls,
            text="Show bonds",
            variable=self.show_bonds_var,
            command=self._preview_display_options_changed,
        ).grid(row=0, column=1, sticky="w", padx=(10, 2), pady=(0, 3))
        ttk.Label(preview_controls, text="Bond radius scale").grid(
            row=0, column=2, sticky="e", padx=(10, 2), pady=(0, 3)
        )
        self.bond_radius_scale_spinbox = self.ttk.Spinbox(
            preview_controls,
            from_=0.1,
            to=5.0,
            increment=0.1,
            width=5,
            textvariable=self.bond_radius_scale_var,
            command=self._preview_display_options_changed,
        )
        self.bond_radius_scale_spinbox.grid(
            row=0, column=3, sticky="w", padx=2, pady=(0, 3)
        )
        self.bond_radius_scale_spinbox.bind(
            "<Return>", lambda _event: self._preview_display_options_changed()
        )
        self.bond_radius_scale_spinbox.bind(
            "<FocusOut>", lambda _event: self._preview_display_options_changed()
        )
        self.spin_elements_frame = ttk.Frame(preview_controls)
        self.spin_elements_frame.grid(
            row=1, column=0, columnspan=4, sticky="w", padx=2, pady=(0, 3)
        )
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
            if column == "CN":
                self.sites_tree.heading(
                    column, text=headings[column], command=self._toggle_sites_cn_sort
                )
            else:
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
        if name == "CN":
            self.status_var.set(
                "CN is computed from the structure's geometry and cannot be "
                "edited directly."
            )
            return
        if name == "label" and self.method_var.get() != "by-coordination":
            self.status_var.set(
                "Label is only editable when Method is set to by-coordination."
            )
            return
        if name == "role" and self.method_var.get() != "by-species":
            self.status_var.set(
                "Role is only editable when Method is set to by-species."
            )
            return
        if name not in {"label", "value", "role"} or not row.use:
            if name in {"label", "value", "role"} and not row.use:
                self.status_var.set(
                    "Check 'use' for this row before editing it."
                )
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

    def _method_changed(self, *_: object) -> None:
        if not self._traces_ready:
            return
        if self._table_after_id is not None:
            self.root.after_cancel(self._table_after_id)
            self._table_after_id = None
        if self.method_var.get() != "by-coordination":
            self._coordination_use_note = None
        self._reset_spin_coordination_numbers_for_result(None)
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

    def _reset_show_bonds_for_structure(self, structure: Structure) -> None:
        self.show_bonds_var.set(
            len(structure.symbols) <= _AUTO_SHOW_BONDS_MAX_ATOMS
        )

    def _reset_spin_elements_for_structure(self, structure: Structure) -> None:
        for widget in self.spin_elements_frame.winfo_children():
            widget.destroy()
        self.spin_element_vars.clear()
        self.spin_element_checkbuttons.clear()
        self.spin_element_labels.clear()
        self.spin_coordination_vars.clear()
        self.spin_coordination_checkbuttons.clear()
        self.spin_coordination_widgets.clear()
        self.spin_coordination_numbers.clear()
        self.ttk.Label(self.spin_elements_frame, text="Show spin for").grid(
            row=0, column=0, sticky="w"
        )
        for offset, element in enumerate(_element_counts(structure)):
            column = 1 + 2 * offset
            variable = self.tk.BooleanVar(value=True)
            checkbutton = self.ttk.Checkbutton(
                self.spin_elements_frame,
                text=element,
                variable=variable,
                command=self._preview_display_options_changed,
            )
            checkbutton.grid(row=0, column=column, sticky="w", padx=(6, 0))
            label = self.ttk.Label(self.spin_elements_frame, text="")
            label.grid(row=0, column=column + 1, sticky="w", padx=(2, 0))
            self.spin_element_vars[element] = variable
            self.spin_element_checkbuttons[element] = checkbutton
            self.spin_element_labels[element] = label

    def _reset_spin_coordination_numbers_for_result(
        self, result: GenerationResult | SpinFileResult | None
    ) -> None:
        for widget in self.spin_coordination_widgets:
            widget.destroy()
        self.spin_coordination_vars.clear()
        self.spin_coordination_checkbuttons.clear()
        self.spin_coordination_widgets.clear()
        self.spin_coordination_numbers.clear()
        if result is None or self.method_var.get() != "by-coordination":
            return
        coordinations = coordination_numbers_from_result(result)
        if not coordinations:
            return
        self.spin_coordination_numbers.update(coordinations)
        heading = self.ttk.Label(self.spin_elements_frame, text="Show spin for CN")
        heading.grid(row=1, column=0, sticky="w")
        self.spin_coordination_widgets.append(heading)
        for offset, coordination in enumerate(sorted(set(coordinations.values()))):
            variable = self.tk.BooleanVar(value=True)
            checkbutton = self.ttk.Checkbutton(
                self.spin_elements_frame,
                text=str(coordination),
                variable=variable,
                command=self._preview_display_options_changed,
            )
            checkbutton.grid(row=1, column=1 + offset, sticky="w", padx=(6, 0))
            self.spin_coordination_vars[coordination] = variable
            self.spin_coordination_checkbuttons[coordination] = checkbutton
            self.spin_coordination_widgets.append(checkbutton)

    def _update_spin_element_summary(self, spins: Mapping[int, float]) -> None:
        if self.current_structure is None:
            return
        counts = element_spin_counts(self.current_structure, spins)
        for element, label in self.spin_element_labels.items():
            n_up, n_down, n_zero = counts[element]
            label.configure(text=f"↑{n_up} ↓{n_down} ·{n_zero}")

    def _preview_display_kwargs(self) -> dict[str, object]:
        show_bonds = self.show_bonds_var.get()
        visible_spin_elements = (
            {
                element
                for element, variable in self.spin_element_vars.items()
                if variable.get()
            }
            if self.spin_element_vars
            else None
        )
        visible_coordination_numbers = (
            {
                coordination
                for coordination, variable in self.spin_coordination_vars.items()
                if variable.get()
            }
            if self.spin_coordination_vars
            else None
        )
        return {
            "show_indices": self.show_atom_indices_var.get(),
            "visible_spin_elements": visible_spin_elements,
            "coordination_numbers": self.spin_coordination_numbers or None,
            "visible_coordination_numbers": visible_coordination_numbers,
            "show_bonds": show_bonds,
            "bond_radius_scale": (
                float(self.bond_radius_scale_var.get()) if show_bonds else 1.0
            ),
        }

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
            anion_species=_split_words(self.anion_species_var.get()) or None,
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
        self._reset_show_bonds_for_structure(structure)
        self._reset_spin_elements_for_structure(structure)
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

    def _workflow_kwargs(
        self,
        methods: Sequence[str],
        *,
        seed_offset: str | int | None = None,
    ) -> dict[str, object]:
        return workflow_kwargs_from_inputs(
            methods,
            self.magnetization_rows,
            site_moment_file=self.site_moment_file_var.get(),
            axis=self.axis_var.get(),
            layer_direction=self.layer_direction_var.get(),
            layer_tolerance=self.tolerance_var.get(),
            fractional_layers=self.fractional_layers_var.get(),
            auto_cutoff=self.auto_cutoff_var.get(),
            cutoff=self.cutoff_var.get(),
            allow_frustrated=self.allow_frustrated_var.get(),
            q_vector=self.q_vector_var.get(),
            afm_type=self.afm_type_var.get(),
            anion_species=self.anion_species_var.get(),
            anion_cutoff=self.anion_cutoff_var.get(),
            up_coordination=self.up_coordination_var.get(),
            down_coordination=self.down_coordination_var.get(),
            coordination_tolerance=self.coordination_tolerance_var.get(),
            max_colors=self.max_colors_var.get(),
            color_spins=self.color_spins_var.get(),
            balance_colors=self.balance_colors_var.get(),
            group_file=self.batch_group_file_var.get(),
            seed_offset=(self.seed_var.get() if seed_offset is None else seed_offset),
        )

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
                coordination_labels=coordination_labels_from_rows(
                    self.magnetization_rows
                ),
                **self._workflow_kwargs(methods),
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
        workflow_kwargs = self._workflow_kwargs(
            [method],
            seed_offset=(
                self.seed_var.get()
                if method in {"random", "graph-coloring"}
                else 0
            ),
        )
        workflow_kwargs.pop("neighbor_shell")
        workflow_kwargs.pop("group_file")
        seed = workflow_kwargs.pop("seed_offset")
        return GenerationParams(
            structure_path=self.structure_path,
            magnetic_species=species,
            method=method,
            moment=moment,
            site_comments=self.site_comments_var.get(),
            slab=self.slab_var.get(),
            coordination_labels=coordination_labels_from_rows(
                self.magnetization_rows
            ),
            seed=seed,
            color_mode=_COLOR_MODES[self.color_mode_var.get()],
            **workflow_kwargs,
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
            self._reset_spin_coordination_numbers_for_result(result)
            camera = None if self._reset_camera else self._capture_camera()
            figure = create_spin_figure(
                result.structure,
                result.spins,
                color_mode=params.color_mode,
                **self._preview_display_kwargs(),
            )
            self._update_spin_element_summary(result.spins)
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
            self._reset_spin_coordination_numbers_for_result(loaded)
            camera = None if self._reset_camera else self._capture_camera()
            figure = create_spin_figure(
                structure,
                loaded.spins,
                color_mode=_COLOR_MODES[self.color_mode_var.get()],
                **self._preview_display_kwargs(),
            )
            self._update_spin_element_summary(loaded.spins)
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
        self._preview_display_options_changed()

    def _preview_display_options_changed(self) -> None:
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
                color_mode=_COLOR_MODES[self.color_mode_var.get()],
                **self._preview_display_kwargs(),
            )
            self._update_spin_element_summary(self.current_spins)
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
        if self._sites_sort_by_cn:
            self._sort_sites_tree()
        self.sites_summary_var.set(site_assignment_summary(result))

    def _sort_sites_tree(self) -> None:
        def key(item: str) -> tuple[int, ...]:
            values = self.sites_tree.item(item, "values")
            atom = int(values[0])
            if not self._sites_sort_by_cn:
                return (atom,)
            try:
                coordination = int(values[2])
            except (TypeError, ValueError):
                return (1, 0, atom)
            return (0, coordination, atom)

        items = sorted(self.sites_tree.get_children(), key=key)
        for position, item in enumerate(items):
            self.sites_tree.move(item, "", position)

    def _toggle_sites_cn_sort(self) -> None:
        self._sites_sort_by_cn = not self._sites_sort_by_cn
        self._sort_sites_tree()

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
        self._deactivate_canvas_mousewheel()
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
