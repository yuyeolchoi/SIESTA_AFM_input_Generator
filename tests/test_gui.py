from __future__ import annotations

import builtins
from pathlib import Path
from types import SimpleNamespace

import pytest

from siesta_afm import gui
from siesta_afm.io import parse_dm_init_spin


ROOT = Path(__file__).parents[1]


def test_windows_launcher_uses_tk_desktop_entrypoint() -> None:
    launcher = (ROOT / "run_gui.bat").read_text(encoding="utf-8")
    assert "import tkinter, matplotlib" in launcher
    assert "-m siesta_afm.gui" in launcher
    assert "streamlit" not in launcher.lower()


def test_gui_package_metadata_has_no_streamlit_dependency() -> None:
    metadata = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'gui = ["matplotlib>=3.7"]' in metadata
    assert "streamlit" not in metadata.lower()
    assert not (ROOT / "app.py").exists()


def test_generation_controller_runs_without_tk_widgets() -> None:
    result = gui.run_generation(
        gui.GenerationParams(
            ROOT / "examples" / "CuO_bulk.cif", ("Cu",), "layer", "0.5"
        )
    )
    assert len(result.spins) == 4
    assert result.report["number_of_magnetic_atoms"] == 4
    assert parse_dm_init_spin(result.block)


def test_generation_controller_rejects_unknown_species() -> None:
    with pytest.raises(ValueError, match="magnetic atoms|species"):
        gui.run_generation(
            gui.GenerationParams(
                ROOT / "examples" / "CuO_bulk.cif", ("Unobtainium",), "layer", "0.5"
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

    monkeypatch.setattr(
        gui,
        "_load_gui_dependencies",
        lambda: SimpleNamespace(tk=FakeTkModule()),
    )
    with pytest.raises(RuntimeError, match="no display available"):
        gui.main()
