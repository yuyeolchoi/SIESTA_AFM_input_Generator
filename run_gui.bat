@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    where python >nul 2>&1
    if errorlevel 1 (
        echo [siesta-afm] Python was not found.
        echo Install Python 3.10 or newer, then try again.
        pause
        exit /b 1
    )
    set "PYTHON_EXE=python"
)

"%PYTHON_EXE%" -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo [siesta-afm] Streamlit is not installed for this Python environment.
    echo Install the GUI dependencies from this repository with:
    echo.
    echo     "%PYTHON_EXE%" -m pip install -e ".[gui]"
    echo.
    pause
    exit /b 1
)

"%PYTHON_EXE%" -m siesta_afm.gui
if errorlevel 1 (
    echo.
    echo [siesta-afm] The GUI exited with an error.
    pause
    exit /b 1
)

endlocal
