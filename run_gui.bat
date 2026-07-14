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

"%PYTHON_EXE%" -c "import tkinter, matplotlib" >nul 2>&1
if errorlevel 1 (
    echo [siesta-afm] Tkinter or matplotlib is unavailable for this Python environment.
    echo Install Python with Tk support and the GUI dependencies with:
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
