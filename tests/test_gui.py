from __future__ import annotations

import builtins
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ase.io import read as ase_read

from siesta_afm import gui
from siesta_afm.fdf_writer import render_dm_init_spin
from siesta_afm.io import parse_dm_init_spin, read_structure
from siesta_afm.structure import Structure
from siesta_afm.workflows import EnumerationResult


ROOT = Path(__file__).parents[1]


def test_windows_launcher_uses_tk_desktop_entrypoint() -> None:
    launcher = (ROOT / "run_gui.bat").read_text(encoding="utf-8")
    assert ".venv\\Scripts\\python.exe" in launcher
    assert "where python" in launcher
    assert "import tkinter, matplotlib" in launcher
    assert "-m siesta_afm.gui" in launcher
    assert 'pip install -e ".[gui]"' in launcher
    assert "streamlit" not in launcher.lower()
    assert "pause" in launcher.lower()


def test_gui_package_metadata_has_no_streamlit_dependency() -> None:
    metadata = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'gui = ["matplotlib>=3.7"]' in metadata
    assert "streamlit" not in metadata.lower()
    assert not (ROOT / "app.py").exists()


def test_generation_controller_runs_without_tk_widgets() -> None:
    result = gui.run_generation(
        gui.GenerationParams(
            structure_path=ROOT / "examples" / "CuO_bulk.cif",
            magnetic_species=("Cu",),
            method="layer",
            moment="0.5",
        )
    )
    assert len(result.spins) == 4
    assert result.assignment.method == "layer"
    assert result.report["number_of_magnetic_atoms"] == 4
    assert parse_dm_init_spin(result.block)
    rows = gui.site_assignment_rows(result)
    assert [row["atom"] for row in rows] == sorted(row["atom"] for row in rows)
    assert all(row["CN"] == "-" and row["sublattice"] == "-" for row in rows)


def test_gui_blank_moment_uses_defaults_and_site_comment_toggle() -> None:
    params = gui.GenerationParams(
        structure_path=ROOT / "examples" / "CuO_bulk.cif",
        magnetic_species=("Cu",),
        method="layer",
    )
    result = gui.run_generation(params)
    assert {abs(value) for value in result.spins.values()} == {1.0}
    assert "using built-in default initial moments" in "\n".join(result.warnings)
    assert "Cu=1.0" in "\n".join(result.warnings)
    assert "# Cu" in result.block

    without_comments = gui.run_generation(
        gui.GenerationParams(
            structure_path=params.structure_path,
            magnetic_species=params.magnetic_species,
            method=params.method,
            moment="0.5",
            site_comments=False,
        )
    )
    spin_rows = without_comments.block.split("%block DM.InitSpin", 1)[1]
    spin_rows = spin_rows.split("%endblock DM.InitSpin", 1)[0]
    assert "#" not in spin_rows


def test_gui_exposes_generation_methods_and_propagation_preset() -> None:
    assert {"random", "by-species", "by-coordination", "graph-coloring"}.issubset(
        gui._METHODS
    )
    assert set(gui._BATCH_METHODS) == set(gui._METHODS) | {
        "manual-groups",
        "frustrated",
    }
    result = gui.run_generation(
        gui.GenerationParams(
            structure_path=ROOT / "examples" / "CuO_bulk.cif",
            magnetic_species=("Cu",),
            method="propagation-vector",
            moment="0.5",
            afm_type="G",
        )
    )
    assert result.assignment.metadata["afm_type"] == "G"
    assert result.report["number_of_magnetic_atoms"] == 4


def test_gui_controller_passes_graph_coloring_options(tmp_path: Path) -> None:
    root3 = 3.0**0.5
    structure = tmp_path / "triangle.xyz"
    structure.write_text(
        "3\ntriangle\n"
        "Cu 0 0 0\n"
        "Cu 1 0 0\n"
        f"Cu 0.5 {root3 / 2:.12f} 0\n",
        encoding="utf-8",
    )
    result = gui.run_generation(
        gui.GenerationParams(
            structure_path=structure,
            magnetic_species=("Cu",),
            method="graph-coloring",
            moment="1",
            cutoff=1.01,
            max_colors=3,
            color_spins="+1,-1,0",
            seed=1,
        )
    )
    assert result.assignment.metadata["n_colors"] == 3
    assert set(result.spins.values()) == {-1.0, 0.0, 1.0}
    assert "proper graph coloring" in "\n".join(result.warnings)


def test_gui_controller_maps_noncollinear_graph_colors_to_angles(
    tmp_path: Path,
) -> None:
    root3 = 3.0**0.5
    structure = tmp_path / "triangle.xyz"
    structure.write_text(
        "3\ntriangle\n"
        "Cu 0 0 0\n"
        "Cu 1 0 0\n"
        f"Cu 0.5 {root3 / 2:.12f} 0\n",
        encoding="utf-8",
    )

    result = gui.run_generation(
        gui.GenerationParams(
            structure_path=structure,
            magnetic_species=("Cu",),
            method="graph-coloring",
            spin_mode="non-collinear",
            moment="1",
            cutoff=1.01,
            max_colors=3,
        )
    )

    angles = gui.angles_from_result(result)
    assert angles is not None
    assert set(result.spins.values()) == {1.0}
    assert {theta for theta, _ in angles.values()} == {90.0}
    assert sorted(phi for _, phi in angles.values()) == pytest.approx(
        [0.0, 120.0, 240.0]
    )
    assert "Spin non-collinear" in result.block
    complete = gui.complete_input_document(result, lda_u=False)
    assert "Spin non-collinear" in complete.text


def test_gui_controller_rejects_invalid_noncollinear_combinations(
    tmp_path: Path,
) -> None:
    structure = tmp_path / "pair.xyz"
    structure.write_text(
        "2\nCu pair\nCu 0 0 0\nCu 1 0 0\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="only supported with --method graph-coloring"):
        gui.run_generation(
            gui.GenerationParams(
                structure_path=structure,
                magnetic_species=("Cu",),
                method="layer",
                spin_mode="non-collinear",
                moment="1",
            )
        )
    with pytest.raises(ValueError, match="cannot be combined with --color-spins"):
        gui.run_generation(
            gui.GenerationParams(
                structure_path=structure,
                magnetic_species=("Cu",),
                method="graph-coloring",
                spin_mode="non-collinear",
                moment="1",
                cutoff=1.01,
                color_spins="+1,-1",
            )
        )


def test_gui_controller_runs_inverse_spinel_coordination_method() -> None:
    result = gui.run_generation(
        gui.GenerationParams(
            structure_path=ROOT / "tests" / "fixtures" / "inverse_spinel_coordination.cif",
            magnetic_species=("Ni", "Co"),
            method="by-coordination",
            moment="Ni@6=1.0 Co@4=2.0 Co@6=0.5",
            slab=True,
            anion_species=("O",),
        )
    )
    assert len(result.spins) == 6
    assert sum(value > 0 for value in result.spins.values()) == 4
    assert sum(value < 0 for value in result.spins.values()) == 2
    assert "inverse spinel" in "\n".join(result.warnings)
    assert gui.coordination_numbers_from_result(result) == (
        result.assignment.metadata["coordination_numbers"]
    )


def test_coordination_site_rows_show_sign_moment_and_zero_spin() -> None:
    result = gui.run_generation(
        gui.GenerationParams(
            structure_path=ROOT / "tests" / "fixtures" / "inverse_spinel_coordination.cif",
            magnetic_species=("Ni", "Co"),
            method="by-coordination",
            moment="Ni@6=2.0 Co@4=2.0 Co@6=0.0",
            slab=True,
            anion_species=("O",),
        )
    )
    rows = gui.site_assignment_rows(result)
    ni_rows = [row for row in rows if row["element"] == "Ni"]
    co4_rows = [row for row in rows if row["element"] == "Co" and row["CN"] == 4]
    co6_rows = [row for row in rows if row["element"] == "Co" and row["CN"] == 6]
    assert len(ni_rows) == len(co4_rows) == len(co6_rows) == 2
    assert all(
        row["CN"] == 6
        and row["sublattice"] == "up"
        and row["sign"] == "+"
        and row["moment"] == 2.0
        for row in ni_rows
    )
    assert all(
        row["sublattice"] == "down"
        and row["sign"] == "-"
        and row["moment"] == 2.0
        for row in co4_rows
    )
    assert all(
        row["sublattice"] == "up"
        and row["sign"] == "0"
        and row["moment"] == 0.0
        for row in co6_rows
    )
    assert gui.site_assignment_summary(result) == (
        "n_up = 2 / n_down = 2 / n_zero = 2, net moment = 0 μB"
    )


def test_detect_coordination_combinations_for_spinel_examples() -> None:
    inverse = read_structure(
        ROOT / "tests" / "fixtures" / "inverse_spinel_coordination.cif",
        slab=True,
    )
    assert gui.detect_coordination_combinations(
        inverse, ("Ni", "Co"), anion_species=("O",)
    ) == {("Ni", 6): 2, ("Co", 4): 2, ("Co", 6): 2}

    co3o4 = read_structure(ROOT / "examples" / "Co3O4_spinel_COD1538531.cif")
    assert gui.detect_coordination_combinations(co3o4, ("Co",)) == {
        ("Co", 4): 8,
        ("Co", 6): 16,
    }


def test_moment_text_is_derived_from_table_rows() -> None:
    rows = [
        gui.MagnetizationRow(True, "Ni", "Oh", 6, "2.0", 8),
        gui.MagnetizationRow(True, "Co", "Td", 4, "2.0", 8),
        gui.MagnetizationRow(True, "Co", "Oh", 6, "0.0", 16),
        gui.MagnetizationRow(False, "O", "", None, "", 32),
    ]
    assert gui.moment_text_from_rows(rows, "by-coordination") == (
        "Ni@6=2.0 Co@4=2.0 Co@6=0.0"
    )
    assert gui.moment_text_from_rows(rows, "layer") == "Ni=2.0 Co=2.0"
    assert gui.batch_moment_text_from_rows(
        rows, ("layer", "by-coordination")
    ) == (
        "Ni=2.0 Co=2.0 Ni@6=2.0 Co@4=2.0 Co@6=0.0"
    )
    assert gui.batch_moment_text_from_rows(rows, ("by-coordination",)) == (
        "Ni@6=2.0 Co@4=2.0 Co@6=0.0"
    )
    for row in rows:
        row.value = ""
    assert gui.moment_text_from_rows(rows, "by-coordination") is None


def test_workflow_kwargs_gate_method_specific_inputs_without_tk() -> None:
    rows = [
        gui.MagnetizationRow(True, "Ni", "Oh", 6, "2.0", 2, "up"),
        gui.MagnetizationRow(True, "Co", "Td", 4, "3.0", 2, "down"),
    ]
    inputs = {
        "site_moment_file": " moments.csv ",
        "axis": "x",
        "layer_direction": "1, 2 3",
        "layer_tolerance": "0.4",
        "fractional_layers": True,
        "auto_cutoff": False,
        "cutoff": "3.5",
        "allow_frustrated": True,
        "q_vector": "not parsed",
        "afm_type": "custom",
        "anion_species": "O, S",
        "anion_cutoff": "2.8",
        "up_coordination": "not parsed",
        "down_coordination": "not parsed",
        "coordination_tolerance": "not parsed",
        "max_colors": "not parsed",
        "color_spins": "not parsed",
        "balance_colors": True,
        "group_file": " ignored.yaml ",
        "seed_offset": "7",
    }

    layer = gui.workflow_kwargs_from_inputs(["layer"], rows, **inputs)
    assert layer == {
        "site_moment_file": "moments.csv",
        "axis": "x",
        "layer_direction": (1.0, 2.0, 3.0),
        "layer_tolerance": 0.4,
        "fractional_layers": True,
        "cutoff": 3.5,
        "neighbor_shell": 1,
        "allow_frustrated": True,
        "q_vector": None,
        "afm_type": None,
        "up_species": ("Ni",),
        "down_species": ("Co",),
        "anion_species": ("O", "S"),
        "anion_cutoff": "2.8",
        "up_coordination": (6,),
        "down_coordination": (4,),
        "coordination_tolerance": 0,
        "max_colors": 4,
        "color_spins": None,
        "balance_colors": False,
        "group_file": None,
        "seed_offset": 7,
    }

    inputs.update(
        up_coordination="5, 6",
        down_coordination="3 4",
        coordination_tolerance="2",
    )
    coordinated = gui.workflow_kwargs_from_inputs(
        ["layer", "by-coordination"], rows, **inputs
    )
    assert coordinated["layer_direction"] == (1.0, 2.0, 3.0)
    assert coordinated["up_coordination"] == (5, 6)
    assert coordinated["down_coordination"] == (3, 4)
    assert coordinated["coordination_tolerance"] == 2
    assert coordinated["q_vector"] is None
    assert coordinated["max_colors"] == 4
    assert coordinated["group_file"] is None


def test_coordination_use_toggle_is_independent_and_survives_refresh() -> None:
    structure = read_structure(
        ROOT / "tests" / "fixtures" / "inverse_spinel_coordination.cif",
        slab=True,
    )
    rows = gui.magnetization_rows_from_structure(
        structure, "by-coordination", anion_species=("O",)
    )
    co6_index = next(
        index
        for index, row in enumerate(rows)
        if row.element == "Co" and row.coordination == 6
    )
    toggled = gui.toggle_magnetization_use(rows, co6_index, "by-coordination")
    assert not toggled.use
    assert next(row for row in rows if row.element == "Co" and row.coordination == 4).use

    refreshed = gui.magnetization_rows_from_structure(
        structure,
        "by-coordination",
        existing_rows=rows,
        anion_species=("O",),
    )
    assert gui.magnetic_species_from_rows(refreshed) == ("Ni", "Co")
    assert gui.moment_text_from_rows(refreshed, "by-coordination") == (
        "Ni@6=2.0 Co@4=3.0"
    )
    assert not next(
        row for row in refreshed if row.element == "Co" and row.coordination == 6
    ).use
    assert next(
        row for row in refreshed if row.element == "Co" and row.coordination == 6
    ).atom_indices == (3, 4)

    with pytest.raises(ValueError, match=r"no initial moment specified.*Co@6"):
        gui.run_generation(
            gui.GenerationParams(
                structure_path=ROOT
                / "tests"
                / "fixtures"
                / "inverse_spinel_coordination.cif",
                magnetic_species=gui.magnetic_species_from_rows(refreshed),
                method="by-coordination",
                moment=gui.moment_text_from_rows(refreshed, "by-coordination"),
                slab=True,
                anion_species=("O",),
            )
        )


def test_noncoordination_use_toggle_still_applies_by_element() -> None:
    rows = [
        gui.MagnetizationRow(True, "Ni", "-", None, "2.0", 2),
        gui.MagnetizationRow(False, "O", "", None, "", 8),
    ]
    gui.toggle_magnetization_use(rows, 0, "layer")
    assert not rows[0].use
    assert (rows[0].label, rows[0].value) == ("", "")


def test_structure_rows_list_elements_defaults_and_method_grouping() -> None:
    structure = read_structure(
        ROOT / "tests" / "fixtures" / "inverse_spinel_coordination.cif",
        slab=True,
    )
    layer_rows = gui.magnetization_rows_from_structure(structure, "layer")
    assert [(row.element, row.count, row.use) for row in layer_rows] == [
        ("Ni", 2, True),
        ("Co", 4, True),
        ("O", 8, False),
    ]
    assert all(
        row.label == "-" and row.coordination is None
        for row in layer_rows
        if row.use
    )
    oxygen = next(row for row in layer_rows if row.element == "O")
    assert (oxygen.label, oxygen.coordination, oxygen.value) == ("", None, "")

    species_rows = gui.magnetization_rows_from_structure(
        structure, "by-species", existing_rows=layer_rows
    )
    assert gui.species_roles_from_rows(species_rows) == (("Ni",), ("Co",))

    coordination_rows = gui.magnetization_rows_from_structure(
        structure,
        "by-coordination",
        existing_rows=layer_rows,
        anion_species=("O",),
    )
    selected = [row for row in coordination_rows if row.use]
    assert [
        (row.element, row.label, row.coordination, row.count) for row in selected
    ] == [
        ("Ni", "Oh", 6, 2),
        ("Co", "Td", 4, 2),
        ("Co", "Oh", 6, 2),
    ]
    assert gui.equivalent_cli_options(coordination_rows, "by-coordination") == (
        "--magnetic-species Ni Co --moment Ni@6=2.0 Co@4=3.0 Co@6=3.0"
    )


def test_coordination_table_failure_falls_back_to_element_rows() -> None:
    structure = Structure(["Cu"], [[0, 0, 0]])
    rows, warning = gui.safe_magnetization_rows_from_structure(
        structure, "by-coordination"
    )
    assert [(row.element, row.label, row.coordination) for row in rows] == [
        ("Cu", "-", None)
    ]
    assert warning is not None
    assert "could not auto-detect an anion species" in warning


def test_coordination_fallback_reports_ambiguous_anion_candidates() -> None:
    structure = Structure(
        ["Cu", "O", "S"],
        [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        np.eye(3) * 4,
        (False, False, False),
    )
    _rows, warning = gui.safe_magnetization_rows_from_structure(
        structure, "by-coordination"
    )
    assert warning is not None
    assert "multiple possible anion species were found (O, S)" in warning
    assert "specify --anion-species" in warning


def test_coordination_geometry_reaches_generated_comments_and_can_be_edited() -> None:
    path = ROOT / "examples" / "CuO_bulk.cif"
    generated = gui.run_generation(
        gui.GenerationParams(
            structure_path=path,
            magnetic_species=("Cu",),
            method="by-coordination",
        )
    )
    assert "square-planar, CN=4" in generated.block
    assert "Td, CN=4" not in generated.block

    edited = gui.run_generation(
        gui.GenerationParams(
            structure_path=path,
            magnetic_species=("Cu",),
            method="by-coordination",
            coordination_labels=(("Cu", 4, "user-label"),),
        )
    )
    assert "user-label, CN=4" in edited.block

    spinel = gui.run_generation(
        gui.GenerationParams(
            structure_path=ROOT / "examples" / "Co3O4_spinel_COD1538531.cif",
            magnetic_species=("Co",),
            method="by-coordination",
        )
    )
    assert spinel.block.count("# Co  (Td, CN=4)") == 8
    assert spinel.block.count("# Co  (Oh, CN=6)") == 16


def test_default_gui_layout_keeps_visible_inputs_wide_enough(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dependencies = gui._load_gui_dependencies()
    try:
        root = dependencies.tk.Tk()
    except dependencies.tk.TclError:
        pytest.skip("Tk display is unavailable")
    try:
        try:
            root.attributes("-alpha", 0.0)
        except dependencies.tk.TclError:
            pass
        app = gui.DesktopApp(root, dependencies)
        root.update()
        visible_inputs = [widget for widget in app.control_inputs if widget.winfo_ismapped()]
        assert visible_inputs
        assert min(widget.winfo_width() for widget in visible_inputs) >= 120
        assert app.main_pane.sashpos(0) >= gui._LEFT_PANEL_MIN_WIDTH
        assert "make-input" in app.complete_input_action.cget("text")
        assert app.complete_input_action.cget("command")
        assert app.method_option_frames["layer"].winfo_ismapped()
        assert not app.method_option_frames["by-coordination"].winfo_ismapped()
        app.structure_path = (
            ROOT / "tests" / "fixtures" / "inverse_spinel_coordination.cif"
        )
        app.current_structure = read_structure(app.structure_path, slab=True)
        app.magnetization_rows = gui.magnetization_rows_from_structure(
            app.current_structure, "layer"
        )
        app.method_var.set("by-coordination")
        root.update()
        assert app.method_option_frames["by-coordination"].winfo_ismapped()
        assert not app.method_option_frames["layer"].winfo_ismapped()
        assert [
            (row.element, row.coordination)
            for row in app.magnetization_rows
            if row.use
        ] == [("Ni", 6), ("Co", 4), ("Co", 6)]

        app.method_var.set("graph-coloring")
        app.spin_mode_var.set("non-collinear")
        root.update()
        assert app.method_option_frames["graph-coloring"].winfo_ismapped()
        assert app.color_spins_entry.instate(["disabled"])
        assert app.balance_colors_check.instate(["disabled"])
        app.method_var.set("layer")
        root.update()
        assert not app.method_option_frames["graph-coloring"].winfo_ismapped()

        generated = gui.run_generation(
            gui.GenerationParams(
                structure_path=ROOT / "examples" / "CuO_bulk.cif",
                magnetic_species=("Cu",),
                method="layer",
                moment="0.5",
            )
        )
        destination = tmp_path / "button_input.fdf"
        exported: list[Path] = []
        monkeypatch.setattr(
            app.deps.filedialog,
            "asksaveasfilename",
            lambda **_kwargs: str(destination),
        )
        monkeypatch.setattr(
            app.deps.messagebox, "showwarning", lambda *_args, **_kwargs: None
        )

        def record_export(_result: object, path: str | Path, **_options: object) -> Path:
            exported.append(Path(path))
            return Path(path)

        monkeypatch.setattr(gui, "export_complete_input", record_export)
        app.current_result = generated
        app.complete_input_action.configure(state="normal")
        app.complete_input_action.invoke()
        assert exported == [destination]
    finally:
        root.destroy()


def test_control_panel_mousewheel_scrolls_with_primary_actions_fixed() -> None:
    dependencies = gui._load_gui_dependencies()
    try:
        root = dependencies.tk.Tk()
    except dependencies.tk.TclError:
        pytest.skip("Tk display is unavailable")
    try:
        try:
            root.attributes("-alpha", 0.0)
        except dependencies.tk.TclError:
            pass
        app = gui.DesktopApp(root, dependencies)
        root.geometry("1050x700")
        root.update()

        before = app.controls_canvas.yview()
        assert before[1] - before[0] < 1.0
        app.controls_canvas.event_generate("<Enter>")
        app.controls_canvas.event_generate("<MouseWheel>", delta=-120)
        root.update()
        after = app.controls_canvas.yview()
        assert after[0] > before[0]

        app.controls_canvas.yview_moveto(1.0)
        root.update()
        assert app.primary_actions.winfo_ismapped()
        assert app.generate_button.winfo_ismapped()
        assert app.complete_input_action.winfo_ismapped()
        assert "Compare several initial spin states" in (
            app.batch_workflow_help_label.cget("text")
        )
        app.controls_canvas.event_generate("<Leave>")
    finally:
        root.destroy()


def test_results_panel_vertical_pane_resizes_notebook() -> None:
    dependencies = gui._load_gui_dependencies()
    try:
        root = dependencies.tk.Tk()
    except dependencies.tk.TclError:
        pytest.skip("Tk display is unavailable")
    try:
        try:
            root.attributes("-alpha", 0.0)
        except dependencies.tk.TclError:
            pass
        app = gui.DesktopApp(root, dependencies)
        root.geometry("1050x700")
        root.update()

        assert str(app.results_pane.cget("orient")) == "vertical"
        assert len(app.results_pane.panes()) == 2
        notebook_height_before = app.results_notebook.winfo_height()
        sash_before = app.results_pane.sashpos(0)
        app.results_pane.sashpos(0, max(0, sash_before - 80))
        root.update()
        assert app.results_notebook.winfo_height() > notebook_height_before

        app.results_pane.sashpos(0, app.results_pane.winfo_height())
        app._enforce_results_notebook_height()
        root.update()
        assert (
            app.results_notebook.winfo_height()
            >= gui._RESULTS_NOTEBOOK_MIN_HEIGHT
        )
    finally:
        root.destroy()


def test_batch_tabs_mousewheel_scroll_without_stealing_treeview_wheel() -> None:
    dependencies = gui._load_gui_dependencies()
    try:
        root = dependencies.tk.Tk()
    except dependencies.tk.TclError:
        pytest.skip("Tk display is unavailable")
    try:
        try:
            root.attributes("-alpha", 0.0)
        except dependencies.tk.TclError:
            pass
        app = gui.DesktopApp(root, dependencies)
        root.geometry("1050x700")
        app.results_notebook.select(3)
        root.update()
        app.results_pane.sashpos(0, app._maximum_preview_height())
        root.update()

        for tab_index, canvas in enumerate(
            (
                app.candidates_canvas,
                app.prepare_canvas,
                app.collected_results_canvas,
            )
        ):
            app.batch_notebook.select(tab_index)
            root.update()
            canvas.yview_moveto(0.0)
            before = canvas.yview()
            assert before[1] - before[0] < 1.0
            canvas.event_generate("<Enter>")
            canvas.event_generate("<MouseWheel>", delta=-120)
            root.update()
            assert canvas.yview()[0] > before[0]
            canvas.event_generate("<Leave>")
            root.update()

        app.batch_notebook.select(0)
        for index in range(20):
            app.candidate_tree.insert(
                "",
                "end",
                values=(f"{index:03d}", "layer", 1, 1, 0, 1, "candidate.fdf"),
            )
        app.candidates_canvas.yview_moveto(1.0)
        app.candidate_tree.yview_moveto(0.0)
        root.update()
        outer_before = app.candidates_canvas.yview()
        tree_before = app.candidate_tree.yview()
        app.candidates_canvas.event_generate("<Enter>")
        app.candidate_tree.event_generate("<Enter>")
        app.candidate_tree.event_generate("<MouseWheel>", delta=-120)
        root.update()
        assert app.candidates_canvas.yview() == outer_before
        assert app.candidate_tree.yview()[0] > tree_before[0]
    finally:
        root.destroy()


def test_atom_index_default_tracks_each_new_structure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dependencies = gui._load_gui_dependencies()
    try:
        root = dependencies.tk.Tk()
    except dependencies.tk.TclError:
        pytest.skip("Tk display is unavailable")
    try:
        try:
            root.attributes("-alpha", 0.0)
        except dependencies.tk.TclError:
            pass
        app = gui.DesktopApp(root, dependencies)
        app.live_update_var.set(False)
        paths = iter(
            (
                ROOT / "tests" / "fixtures" / "NiCo2O4_311_pristine.cif",
                ROOT / "examples" / "CuO_bulk.cif",
            )
        )
        monkeypatch.setattr(
            app.deps.filedialog,
            "askopenfilename",
            lambda **_kwargs: str(next(paths)),
        )

        assert app._choose_structure(schedule=False)
        assert app.current_structure is not None
        assert len(app.current_structure.symbols) > gui._AUTO_SHOW_INDICES_MAX_ATOMS
        assert app.show_atom_indices_var.get() is False

        app.show_atom_indices_var.set(True)
        app._refresh_magnetization_table()
        assert app.show_atom_indices_var.get() is True

        app.show_atom_indices_var.set(False)
        assert app._choose_structure(schedule=False)
        assert app.current_structure is not None
        assert len(app.current_structure.symbols) < gui._AUTO_SHOW_INDICES_MAX_ATOMS
        assert app.show_atom_indices_var.get() is True
    finally:
        root.destroy()


def test_element_spin_checkboxes_rerender_with_real_tk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dependencies = gui._load_gui_dependencies()
    try:
        root = dependencies.tk.Tk()
    except dependencies.tk.TclError:
        pytest.skip("Tk display is unavailable")
    app = None
    try:
        try:
            root.attributes("-alpha", 0.0)
        except dependencies.tk.TclError:
            pass
        app = gui.DesktopApp(root, dependencies)
        app.live_update_var.set(False)
        monkeypatch.setattr(
            app.deps.filedialog,
            "askopenfilename",
            lambda **_kwargs: str(ROOT / "examples" / "CuO_bulk.cif"),
        )
        assert app._choose_structure(schedule=False)
        root.update()

        assert set(app.spin_element_checkbuttons) == {"Cu", "O"}
        assert set(app.spin_element_labels) == {"Cu", "O"}
        assert all(
            widget.winfo_ismapped()
            for widget in app.spin_element_checkbuttons.values()
        )
        assert all(
            label.cget("text") == "" for label in app.spin_element_labels.values()
        )
        assert app.show_bonds_var.get() is True

        app._generate(show_dialog=False)
        root.update()
        assert app.current_result is not None
        assert all(label.cget("text") for label in app.spin_element_labels.values())

        from siesta_afm.gui import app as gui_app

        original = gui_app.create_spin_figure
        visible_calls: list[set[str] | None] = []

        def record_create(*args: object, **kwargs: object) -> object:
            visible_calls.append(kwargs.get("visible_spin_elements"))  # type: ignore[arg-type]
            return original(*args, **kwargs)

        monkeypatch.setattr(gui_app, "create_spin_figure", record_create)
        cu_summary = app.spin_element_labels["Cu"].cget("text")
        app.spin_element_checkbuttons["Cu"].invoke()
        root.update()

        assert app.spin_element_vars["Cu"].get() is False
        assert visible_calls == [{"O"}]
        assert app.spin_element_labels["Cu"].cget("text") == cu_summary
        assert app.figure is not None
        assert app.spin_element_checkbuttons["Cu"].winfo_ismapped()
    finally:
        if app is not None and app.figure is not None:
            app.figure.clear()
        root.destroy()


def test_coordination_spin_checkboxes_combine_and_follow_method_with_real_tk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dependencies = gui._load_gui_dependencies()
    try:
        root = dependencies.tk.Tk()
    except dependencies.tk.TclError:
        pytest.skip("Tk display is unavailable")
    app = None
    try:
        try:
            root.attributes("-alpha", 0.0)
        except dependencies.tk.TclError:
            pass
        app = gui.DesktopApp(root, dependencies)
        app.live_update_var.set(False)
        monkeypatch.setattr(
            app.deps.filedialog,
            "askopenfilename",
            lambda **_kwargs: str(
                ROOT / "tests" / "fixtures" / "inverse_spinel_coordination.cif"
            ),
        )
        app.slab_var.set(True)
        assert app._choose_structure(schedule=False)
        app.method_var.set("by-coordination")
        root.update()
        app._generate(show_dialog=False)
        root.update()

        assert app.current_result is not None
        assert set(app.spin_coordination_checkbuttons) == {4, 6}
        assert all(variable.get() for variable in app.spin_coordination_vars.values())
        assert all(
            widget.winfo_ismapped()
            for widget in app.spin_coordination_checkbuttons.values()
        )

        from siesta_afm.gui import app as gui_app
        from siesta_afm.visualize import classify_spin_indices

        original = gui_app.create_spin_figure
        display_calls: list[dict[str, object]] = []

        def record_create(*args: object, **kwargs: object) -> object:
            display_calls.append(kwargs.copy())
            return original(*args, **kwargs)

        monkeypatch.setattr(gui_app, "create_spin_figure", record_create)
        app.spin_coordination_checkbuttons[4].invoke()
        root.update()
        cn_kwargs = display_calls[-1]
        assert cn_kwargs["visible_coordination_numbers"] == {6}
        nonmagnetic, up, down = classify_spin_indices(
            app.current_structure,
            app.current_spins,
            cn_kwargs["visible_spin_elements"],
            cn_kwargs["coordination_numbers"],
            cn_kwargs["visible_coordination_numbers"],
        )
        assert len(up) + len(down) == 4
        assert len(nonmagnetic) == len(app.current_structure) - 4

        app.spin_element_checkbuttons["Co"].invoke()
        root.update()
        combined_kwargs = display_calls[-1]
        assert "Co" not in combined_kwargs["visible_spin_elements"]
        nonmagnetic, up, down = classify_spin_indices(
            app.current_structure,
            app.current_spins,
            combined_kwargs["visible_spin_elements"],
            combined_kwargs["coordination_numbers"],
            combined_kwargs["visible_coordination_numbers"],
        )
        assert len(up) + len(down) == 2
        assert len(nonmagnetic) == len(app.current_structure) - 2

        app.method_var.set("layer")
        root.update()
        assert app.spin_coordination_checkbuttons == {}
        assert app.spin_coordination_vars == {}
        assert app.spin_coordination_numbers == {}
    finally:
        if app is not None and app.figure is not None:
            app.figure.clear()
        root.destroy()


def test_sites_cn_heading_toggles_coordination_and_atom_order() -> None:
    dependencies = gui._load_gui_dependencies()
    try:
        root = dependencies.tk.Tk()
    except dependencies.tk.TclError:
        pytest.skip("Tk display is unavailable")
    try:
        try:
            root.attributes("-alpha", 0.0)
        except dependencies.tk.TclError:
            pass
        app = gui.DesktopApp(root, dependencies)
        result = gui.run_generation(
            gui.GenerationParams(
                structure_path=(
                    ROOT / "tests" / "fixtures" / "inverse_spinel_coordination.cif"
                ),
                magnetic_species=("Ni", "Co"),
                method="by-coordination",
                moment="Ni@6=2.0 Co@4=2.0 Co@6=0.0",
                slab=True,
                anion_species=("O",),
            )
        )
        app._set_site_table(result)

        def displayed_atoms() -> list[int]:
            return [
                int(app.sites_tree.item(item, "values")[0])
                for item in app.sites_tree.get_children()
            ]

        atom_order = displayed_atoms()
        assert atom_order == sorted(atom_order)
        command = app.sites_tree.heading("CN", "command")
        assert command
        app.sites_tree.tk.call(command)
        cn_rows = [
            app.sites_tree.item(item, "values")
            for item in app.sites_tree.get_children()
        ]
        assert [int(row[2]) for row in cn_rows] == sorted(
            int(row[2]) for row in cn_rows
        )
        app.sites_tree.tk.call(command)
        assert displayed_atoms() == atom_order
    finally:
        root.destroy()


def test_gui_controller_passes_site_moment_csv(tmp_path: Path) -> None:
    site_file = tmp_path / "site_moments.csv"
    site_file.write_text("atom_index,element,moment\n1,Ni,3.0\n", encoding="utf-8")
    result = gui.run_generation(
        gui.GenerationParams(
            structure_path=ROOT / "tests" / "fixtures" / "inverse_spinel_coordination.cif",
            magnetic_species=("Ni", "Co"),
            method="by-coordination",
            moment="Ni@6=2.0 Co@4=2.0 Co@6=0.0",
            site_moment_file=site_file,
            slab=True,
            anion_species=("O",),
        )
    )
    assert result.spins[0] == 3.0
    assert result.spins[1] == 2.0


def test_batch_controller_rows_and_result_sorting(tmp_path: Path) -> None:
    enumeration = EnumerationResult(
        manifest=[
            {
                "config_id": "001",
                "method": "layer",
                "n_up": 1,
                "n_down": 1,
                "net_spin": 0.0,
                "afm_score": 1.0,
                "file": "afm_001.fdf",
            }
        ],
        failures=[],
        notices=[],
        written_files=[tmp_path / "afm_001.fdf"],
        manifest_path=tmp_path / "manifest.csv",
    )
    assert gui.candidate_table_rows(enumeration) == [
        ("001", "layer", "1", "1", "0", "1", "afm_001.fdf")
    ]

    rows = gui.results_table_rows(
        [
            {
                "config_id": "003",
                "total_energy": "",
                "scf_converged": "False",
                "status": "missing-output",
            },
            {
                "config_id": "002",
                "total_energy": -10.995,
                "scf_converged": True,
                "status": "ok",
            },
            {
                "config_id": "001",
                "total_energy": -11.0,
                "scf_converged": True,
                "status": "ok",
            },
        ]
    )
    assert [row.values[0] for row in rows] == ["001", "002", "003"]
    assert rows[0].tags == ("near_ground",)
    assert rows[1].tags == ("near_ground",)
    assert rows[2].tags == ("unconverged",)


def test_batch_prepare_and_existing_results_wrappers(tmp_path: Path) -> None:
    base = tmp_path / "base.fdf"
    base.write_text("SpinPolarized false\n", encoding="utf-8")
    candidates = tmp_path / "candidates"
    candidates.mkdir()
    (candidates / "spin.fdf").write_text(
        "%block DM.InitSpin\n1 1\n%endblock DM.InitSpin\n", encoding="utf-8"
    )
    (candidates / "manifest.csv").write_text(
        "config_id,method,n_up,n_down,net_spin,afm_score,file\n"
        "001,layer,1,0,1,0,spin.fdf\n",
        encoding="utf-8",
    )
    jobs = tmp_path / "jobs"
    folders = gui.prepare_job_folders(base, candidates, jobs)
    assert [folder.name for folder in folders] == ["001_layer"]
    assert (jobs / "folders.list").read_text(encoding="utf-8") == "001_layer\n"
    (folders[0] / "siesta.out").write_text(
        "siesta: E_KS(eV) = -1.5\nSCF cycle converged\n",
        encoding="utf-8",
    )
    collected = gui.collect_or_load_results(jobs_dir=jobs)
    assert collected[0]["config_id"] == "001"
    assert collected[0]["total_energy"] == -1.5
    assert collected[0]["scf_converged"] is True
    assert (jobs / "results.csv").is_file()

    existing = tmp_path / "results.csv"
    existing.write_text(
        "config_id,total_energy,scf_converged,status\n"
        "001,-1.25,True,ok\n",
        encoding="utf-8",
    )
    assert gui.collect_or_load_results(results_csv=existing) == [
        {
            "config_id": "001",
            "total_energy": "-1.25",
            "scf_converged": "True",
            "status": "ok",
        }
    ]


def test_batch_controllers_complete_workflow_without_tk(tmp_path: Path) -> None:
    structure_path = ROOT / "examples" / "CuO_bulk.cif"
    structure = read_structure(structure_path)
    candidates = tmp_path / "candidates"
    enumeration = gui.run_candidate_generation(
        structure,
        ("Cu",),
        ("layer",),
        "Cu=1",
        1,
        candidates,
        cutoff="auto",
    )
    assert len(gui.candidate_table_rows(enumeration)) == 1

    base_input = tmp_path / "base.fdf"
    generated = gui.run_generation(
        gui.GenerationParams(
            structure_path=structure_path,
            magnetic_species=("Cu",),
            method="layer",
            moment="1",
        )
    )
    gui.export_complete_input(generated, base_input)
    jobs = tmp_path / "jobs"
    folders = gui.prepare_job_folders(base_input, candidates, jobs)
    assert len(folders) == 1
    (folders[0] / "siesta.out").write_text(
        "siesta: E_KS(eV) = -11.25\nSCF cycle converged\n",
        encoding="utf-8",
    )

    rows = gui.collect_or_load_results(jobs_dir=jobs)
    displayed = gui.results_table_rows(rows)
    assert displayed[0].values[0:2] == ("001", "-11.25")
    assert displayed[0].values[6] == "True"
    assert displayed[0].tags == ("near_ground",)


def test_real_tk_buttons_complete_batch_workflow(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dependencies = gui._load_gui_dependencies()
    try:
        root = dependencies.tk.Tk()
    except dependencies.tk.TclError:
        pytest.skip("Tk display is unavailable")
    try:
        try:
            root.attributes("-alpha", 0.0)
        except dependencies.tk.TclError:
            pass
        app = gui.DesktopApp(root, dependencies)
        app.live_update_var.set(False)
        app.structure_path = ROOT / "examples" / "CuO_bulk.cif"
        app.current_structure = read_structure(app.structure_path)
        app.magnetization_rows = gui.magnetization_rows_from_structure(
            app.current_structure, "layer"
        )
        app.method_var.set("layer")
        app.batch_n_configs_var.set("1")
        app.candidate_output_var.set(str(tmp_path / "candidates"))
        app.results_notebook.select(3)
        root.update()
        errors: list[str] = []
        monkeypatch.setattr(
            app.deps.messagebox,
            "showerror",
            lambda _title, message: errors.append(str(message)),
        )
        button = app.generate_candidates_button
        button.event_generate("<ButtonPress-1>", x=5, y=5)
        root.update()
        button.event_generate("<ButtonRelease-1>", x=5, y=5)
        root.update()
        assert not errors
        assert (tmp_path / "candidates" / "manifest.csv").is_file()
        assert (tmp_path / "candidates" / "afm_001.fdf").is_file()
        assert len(app.candidate_tree.get_children()) == 1

        base_input = tmp_path / "base.fdf"
        generated = gui.run_generation(
            gui.GenerationParams(
                structure_path=app.structure_path,
                magnetic_species=("Cu",),
                method="layer",
                moment="1",
            )
        )
        gui.export_complete_input(generated, base_input)
        jobs = tmp_path / "jobs"
        app.base_input_var.set(str(base_input))
        app.jobs_output_var.set(str(jobs))
        app.batch_notebook.select(1)
        root.update()
        app.prepare_jobs_button.event_generate("<ButtonPress-1>", x=5, y=5)
        root.update()
        app.prepare_jobs_button.event_generate("<ButtonRelease-1>", x=5, y=5)
        root.update()
        folder = jobs / "001_layer"
        assert (folder / "RUN.fdf").is_file()
        assert app.job_folders_text.get("1.0", "end").strip() == "001_layer"

        (folder / "siesta.out").write_text(
            "siesta: E_KS(eV) = -11.25\nSCF cycle converged\n",
            encoding="utf-8",
        )
        app.batch_notebook.select(2)
        root.update()
        app.collect_results_button.event_generate("<ButtonPress-1>", x=5, y=5)
        root.update()
        app.collect_results_button.event_generate("<ButtonRelease-1>", x=5, y=5)
        root.update()
        assert not errors
        assert (jobs / "results.csv").is_file()
        result_items = app.collected_results_tree.get_children()
        assert len(result_items) == 1
        values = app.collected_results_tree.item(result_items[0], "values")
        assert values[0] == "001"
        assert values[1] == "-11.25"
        assert values[6] == "True"
        assert app.collected_results_tree.item(result_items[0], "tags") == (
            "near_ground",
        )
    finally:
        root.destroy()


_TK_SUBPROCESS_ENV_FLAG = "SIESTA_AFM_TK_SUBPROCESS_CHILD"


def test_real_tk_generate_buttons_share_edited_coordination_label(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # This test has a long history of failing only inside a full-suite run
    # on Windows (never in isolation): earlier tests in the same process
    # create and destroy Tk() roots, and that leftover interpreter/Tcl
    # state can shift widget geometry queries (bbox()) enough to make
    # event_generate() land on the wrong table cell. Re-running the exact
    # same test body in a fresh subprocess makes the "isolation" that was
    # previously only a manual debugging step (and thus periodically
    # reintroduced flakiness in CI) a structural property of the test
    # itself, without weakening what it actually asserts.
    if os.environ.get(_TK_SUBPROCESS_ENV_FLAG) == "1":
        _real_tk_generate_buttons_share_edited_coordination_label(
            monkeypatch, tmp_path
        )
        return
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-q",
            f"{__file__}::test_real_tk_generate_buttons_share_edited_coordination_label",
        ],
        cwd=ROOT,
        env={**os.environ, _TK_SUBPROCESS_ENV_FLAG: "1"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, (
        "isolated subprocess re-run failed:\n"
        f"{completed.stdout}\n{completed.stderr}"
    )


def _real_tk_generate_buttons_share_edited_coordination_label(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dependencies = gui._load_gui_dependencies()
    try:
        root = dependencies.tk.Tk()
    except dependencies.tk.TclError:
        pytest.skip("Tk display is unavailable")
    try:
        try:
            root.attributes("-alpha", 0.0)
        except dependencies.tk.TclError:
            pass
        app = gui.DesktopApp(root, dependencies)
        app.live_update_var.set(False)
        app.structure_path = ROOT / "examples" / "CuO_bulk.cif"
        app.current_structure = read_structure(app.structure_path)
        app.magnetization_rows = gui.magnetization_rows_from_structure(
            app.current_structure, "layer"
        )
        app.method_var.set("by-coordination")
        root.update()

        row_index = next(
            index
            for index, row in enumerate(app.magnetization_rows)
            if row.use and row.element == "Cu" and row.coordination == 4
        )
        item = str(row_index)
        x, y, width, height = app.magnetization_tree.bbox(item, "#3")
        for _ in range(2):
            app.magnetization_tree.event_generate(
                "<ButtonPress-1>", x=x + width // 2, y=y + height // 2
            )
            app.magnetization_tree.event_generate(
                "<ButtonRelease-1>", x=x + width // 2, y=y + height // 2
            )
            root.update()
        assert app._cell_editor is not None
        app._cell_editor.delete(0, "end")
        app._cell_editor.insert(0, "user-label")
        app._cell_editor.event_generate("<Return>")
        root.update()
        assert app.magnetization_rows[row_index].label == "user-label"

        errors: list[str] = []
        monkeypatch.setattr(
            app.deps.messagebox,
            "showerror",
            lambda _title, message: errors.append(str(message)),
        )
        app.generate_button.event_generate("<ButtonPress-1>", x=5, y=5)
        root.update()
        app.generate_button.event_generate("<ButtonRelease-1>", x=5, y=5)
        root.update()
        single_block = app.current_block
        assert not errors
        assert "user-label, CN=4" in single_block

        for method, variable in app.batch_method_vars.items():
            variable.set(method == "by-coordination")
        candidates = tmp_path / "candidates"
        app.batch_n_configs_var.set("1")
        app.candidate_output_var.set(str(candidates))
        app.results_notebook.select(3)
        root.update()
        button = app.generate_candidates_button
        button.event_generate("<ButtonPress-1>", x=5, y=5)
        root.update()
        button.event_generate("<ButtonRelease-1>", x=5, y=5)
        root.update()

        candidate_block = (candidates / "afm_001.fdf").read_text(encoding="utf-8")
        assert not errors
        assert "user-label, CN=4" in candidate_block
        assert single_block.count("user-label, CN=4") == candidate_block.count(
            "user-label, CN=4"
        )
        assert "square-planar, CN=4" not in single_block
        assert "square-planar, CN=4" not in candidate_block
    finally:
        root.destroy()


def test_gui_complete_input_export_uses_shared_roundtrippable_renderer(
    tmp_path: Path,
) -> None:
    generated = gui.run_generation(
        gui.GenerationParams(
            structure_path=ROOT / "examples" / "CuO_bulk.cif",
            magnetic_species=("Cu",),
            method="layer",
            moment="0.5",
        )
    )
    destination = gui.export_complete_input(generated, tmp_path / "input.fdf")
    reread = read_structure(destination)
    assert reread.symbols == generated.structure.symbols
    assert np.allclose(reread.positions, generated.structure.positions)
    assert parse_dm_init_spin(destination) == [
        (index + 1, value) for index, value in sorted(generated.spins.items())
    ]


def test_generation_controller_rejects_unknown_species() -> None:
    with pytest.raises(ValueError, match="magnetic atoms|species"):
        gui.run_generation(
            gui.GenerationParams(
                structure_path=ROOT / "examples" / "CuO_bulk.cif",
                magnetic_species=("Unobtainium",),
                method="layer",
                moment="0.5",
            )
        )


def test_missing_matplotlib_has_gui_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def without_matplotlib(name: str, *args: object, **kwargs: object) -> object:
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ImportError("simulated missing matplotlib")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_matplotlib)
    with pytest.raises(RuntimeError, match=r"pip install -e.*\[gui\]"):
        gui._load_matplotlib_tk_backend()


def test_main_wraps_tk_display_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeTclError(Exception):
        pass

    class FakeTkModule:
        TclError = FakeTclError

        @staticmethod
        def Tk() -> None:
            raise FakeTclError("no display")

    dependencies = SimpleNamespace(tk=FakeTkModule())
    monkeypatch.setattr(gui, "_load_gui_dependencies", lambda: dependencies)
    with pytest.raises(RuntimeError, match="no display available"):
        gui.main()


def test_spin_file_viewer_rejects_out_of_range_indices(tmp_path: Path) -> None:
    structure = Structure(["Cu"], [[0, 0, 0]], np.eye(3), (False, False, False))
    spin_file = tmp_path / "bad_spin.fdf"
    spin_file.write_text(
        "%block DM.InitSpin\n2 0.5\n%endblock DM.InitSpin\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="out of range"):
        gui.load_spin_file(spin_file, structure)


def test_spin_file_viewer_converts_indices_and_builds_preview_block(
    tmp_path: Path,
) -> None:
    structure = Structure(
        ["Cu", "O", "Cu"],
        [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        np.eye(3) * 4,
        (False, False, False),
    )
    spin_file = tmp_path / "spin.fdf"
    spin_file.write_text(
        "%block DM.InitSpin\n1 0.7\n3 -0.5\n%endblock DM.InitSpin\n",
        encoding="utf-8",
    )
    loaded = gui.load_spin_file(spin_file, structure)
    assert loaded.spins == {0: 0.7, 2: -0.5}
    assert loaded.validation.valid
    assert parse_dm_init_spin(loaded.block) == [(1, 0.7), (3, -0.5)]
    assert "# Cu" in loaded.block
    rows = gui.site_assignment_rows(loaded)
    assert [row["atom"] for row in rows] == [1, 3]
    assert all(row["CN"] == "-" and row["sublattice"] == "-" for row in rows)
    assert gui.site_assignment_summary(loaded).startswith("n_up = 1 / n_down = 1")


def test_spin_file_viewer_preserves_noncollinear_angles(tmp_path: Path) -> None:
    structure = Structure(
        ["Cu", "Cu"],
        [[0, 0, 0], [1, 0, 0]],
        np.eye(3) * 4,
        (False, False, False),
    )
    spin_file = tmp_path / "noncollinear.fdf"
    spin_file.write_text(
        "Spin non-collinear\n"
        "%block DM.InitSpin\n"
        "1 1.0 90.0 0.0\n"
        "2 1.0 90.0 180.0\n"
        "%endblock DM.InitSpin\n",
        encoding="utf-8",
    )

    loaded = gui.load_spin_file(spin_file, structure)

    assert gui.angles_from_result(loaded) == {
        0: (90.0, 0.0),
        1: (90.0, 180.0),
    }
    assert "Spin non-collinear" in loaded.block
    assert parse_dm_init_spin(loaded.block, include_angles=True) == [
        (1, 1.0, 90.0, 0.0),
        (2, 1.0, 90.0, 180.0),
    ]


def test_export_patched_input_is_idempotent_and_preserves_base(tmp_path: Path) -> None:
    base = tmp_path / "input.fdf"
    original = (ROOT / "examples" / "input.fdf").read_text(encoding="utf-8")
    base.write_text(original, encoding="utf-8")
    block = render_dm_init_spin(
        {0: 0.7, 2: -0.5}, method="layer", magnetic_species=["Cu"]
    )
    first = gui.export_patched_input(base, block, tmp_path / "input_afm.fdf")
    second = gui.export_patched_input(first, block, tmp_path / "input_afm_twice.fdf")
    assert base.read_text(encoding="utf-8") == original
    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")
    assert parse_dm_init_spin(first) == [(1, 0.7), (3, -0.5)]
    with pytest.raises(ValueError, match="must not overwrite"):
        gui.export_patched_input(base, block, base)


@pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array has been deprecated:DeprecationWarning"
)
def test_export_extxyz_preserves_initial_magnetic_moments(tmp_path: Path) -> None:
    structure = Structure(
        ["Cu", "O", "Cu"],
        [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        np.eye(3) * 4,
        (False, False, False),
    )
    output = gui.export_structure_with_moments(
        structure, {0: 0.7, 2: -0.5}, tmp_path / "moments.xyz"
    )
    atoms = ase_read(output)
    assert np.allclose(atoms.get_initial_magnetic_moments(), [0.7, 0.0, -0.5])
