@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found. Creating it now...
    call "%~dp0setup_venv_windows.bat" --no-pause
    if errorlevel 1 (
        echo.
        echo Could not create the virtual environment.
        pause
        exit /b 1
    )
)

if exist ".venv\Scripts\pythonw.exe" (
    start "ABR Wave Analyser" ".venv\Scripts\pythonw.exe" "%~dp0mainWindow.py"
) else (
    ".venv\Scripts\python.exe" "%~dp0mainWindow.py"
)
