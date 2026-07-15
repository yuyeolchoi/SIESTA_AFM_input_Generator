from __future__ import annotations

import builtins
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ase.io import read as ase_read

from siesta_afm import gui
from siesta_afm.fdf_writer import render_dm_init_spin
from siesta_afm.io import parse_dm_init_spin, read_structure
from siesta_afm.structure import Structure


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
    for row in rows:
        row.value = ""
    assert gui.moment_text_from_rows(rows, "by-coordination") is None


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


def test_default_gui_layout_keeps_visible_inputs_wide_enough() -> None:
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
