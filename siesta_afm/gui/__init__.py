"""Desktop GUI package with a Tk-free controller API."""

import sys
from types import ModuleType

from . import app as _app
from .app import (
    DesktopApp,
    _CameraState,
    _GuiDependencies,
    _load_gui_dependencies,
    _load_matplotlib_tk_backend,
    main,
)
from .controllers import (
    GenerationParams,
    GenerationResult,
    MagnetizationRow,
    ResultTableRow,
    SpinFileResult,
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
    _coordination_combination_counts,
    _display_value,
    _element_counts,
    _energy_value,
    _split_words,
    _three_numbers,
    _truthy,
    batch_moment_text_from_rows,
    candidate_table_rows,
    collect_or_load_results,
    complete_input_document,
    coordination_labels_from_rows,
    detect_coordination_combinations,
    equivalent_cli_options,
    export_complete_input,
    export_patched_input,
    export_spin_block,
    export_structure_with_moments,
    load_results_csv,
    load_spin_file,
    magnetic_species_from_rows,
    magnetization_rows_from_structure,
    moment_text_from_rows,
    prepare_job_folders,
    results_table_rows,
    run_candidate_generation,
    run_generation,
    safe_magnetization_rows_from_structure,
    site_assignment_rows,
    site_assignment_summary,
    species_roles_from_rows,
    toggle_magnetization_use,
    workflow_kwargs_from_inputs,
)

__all__ = [
    "DesktopApp",
    "GenerationParams",
    "GenerationResult",
    "MagnetizationRow",
    "ResultTableRow",
    "SpinFileResult",
    "batch_moment_text_from_rows",
    "candidate_table_rows",
    "collect_or_load_results",
    "complete_input_document",
    "coordination_labels_from_rows",
    "detect_coordination_combinations",
    "equivalent_cli_options",
    "export_complete_input",
    "export_patched_input",
    "export_spin_block",
    "export_structure_with_moments",
    "load_results_csv",
    "load_spin_file",
    "magnetic_species_from_rows",
    "magnetization_rows_from_structure",
    "main",
    "moment_text_from_rows",
    "prepare_job_folders",
    "results_table_rows",
    "run_candidate_generation",
    "run_generation",
    "safe_magnetization_rows_from_structure",
    "site_assignment_rows",
    "site_assignment_summary",
    "species_roles_from_rows",
    "toggle_magnetization_use",
    "workflow_kwargs_from_inputs",
]


class _GuiModule(ModuleType):
    def __setattr__(self, name: str, value: object) -> None:
        super().__setattr__(name, value)
        if name != "_app" and hasattr(_app, name):
            setattr(_app, name, value)


sys.modules[__name__].__class__ = _GuiModule
