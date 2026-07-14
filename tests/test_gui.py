from __future__ import annotations

import builtins
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ase.io import read as ase_read

from siesta_afm import gui
from siesta_afm.fdf_writer import render_dm_init_spin
from siesta_afm.io import parse_dm_init_spin
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


def test_gui_exposes_f9_methods_and_propagation_preset() -> None:
    assert {"random", "by-species", "by-coordination"}.issubset(gui._METHODS)
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
