from __future__ import annotations

import builtins
import os
import subprocess
import sys
from pathlib import Path

import pytest

from siesta_afm import gui


def test_windows_launcher_uses_expected_environment_and_entrypoint() -> None:
    project_root = Path(__file__).resolve().parents[1]
    launcher = (project_root / "run_gui.bat").read_text(encoding="utf-8")
    assert ".venv\\Scripts\\python.exe" in launcher
    assert "where python" in launcher
    assert "-m siesta_afm.gui" in launcher
    assert 'pip install -e ".[gui]"' in launcher
    assert "pause" in launcher.lower()


def test_streamlit_gui_exposes_value_color_mode() -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(gui.__file__).run(timeout=10)
    assert not app.exception
    color_mode = next(item for item in app.selectbox if item.label == "Color mode")
    assert color_mode.options == ["spin sign", "spin value"]
    assert color_mode.value == "spin sign"
    color_mode.select("spin value").run()
    assert color_mode.value == "spin value"
    assert not app.exception


def test_gui_main_reports_missing_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def without_streamlit(name: str, *args: object, **kwargs: object) -> object:
        if name == "streamlit" or name.startswith("streamlit."):
            raise ImportError("simulated missing streamlit")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_streamlit)
    with pytest.raises(RuntimeError, match=r"pip install -e.*\[gui\]"):
        gui.main()


def test_module_entrypoint_bootstraps_outside_streamlit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gui, "_running_in_streamlit", lambda: False)
    monkeypatch.setattr(gui, "main", lambda: 23)
    monkeypatch.setattr(
        gui, "run", lambda: pytest.fail("run() called outside Streamlit")
    )
    assert gui._module_entrypoint() == 23


def test_module_entrypoint_renders_inside_streamlit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: list[bool] = []
    monkeypatch.setattr(gui, "_running_in_streamlit", lambda: True)
    monkeypatch.setattr(gui, "run", lambda: called.append(True))
    monkeypatch.setattr(
        gui, "main", lambda: pytest.fail("main() called inside Streamlit")
    )
    assert gui._module_entrypoint() is None
    assert called == [True]


def test_python_m_gui_reports_install_hint_without_streamlit(tmp_path: Path) -> None:
    # Shadow even an installed Streamlit so this subprocess exercises the
    # missing-extra path consistently in developer and CI environments.
    (tmp_path / "streamlit.py").write_text(
        'raise ImportError("simulated missing streamlit")\n', encoding="utf-8"
    )
    environment = os.environ.copy()
    project_root = Path(__file__).resolve().parents[1]
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(
            None,
            [str(tmp_path), str(project_root), environment.get("PYTHONPATH", "")],
        )
    )
    completed = subprocess.run(
        [sys.executable, "-m", "siesta_afm.gui"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "pip install -e" in completed.stderr
    assert "[gui]" in completed.stderr
