@echo off
setlocal

cd /d "%~dp0"

set "PYTHON_CMD="

py -3.10 -c "import sys" >nul 2>&1
if not errorlevel 1 set "PYTHON_CMD=py -3.10"

if not defined PYTHON_CMD (
    python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 10) else 1)" >nul 2>&1
    if not errorlevel 1 set "PYTHON_CMD=python"
)

if not defined PYTHON_CMD (
    for %%V in (3.12 3.11 3.9) do (
        if not defined PYTHON_CMD (
            py -%%V -c "import sys" >nul 2>&1
            if not errorlevel 1 set "PYTHON_CMD=py -%%V"
        )
    )
)

if not defined PYTHON_CMD (
    python -c "import sys; v = sys.version_info[:2]; raise SystemExit(0 if (3, 9) <= v <= (3, 12) else 1)" >nul 2>&1
    if not errorlevel 1 set "PYTHON_CMD=python"
)

if not defined PYTHON_CMD (
    echo Python was not found.
    echo Install Python 3.10 for Windows, then run this script again.
    goto :error
)

echo Using Python command: %PYTHON_CMD%

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment in "%CD%\.venv"...
    %PYTHON_CMD% -m venv ".venv"
    if errorlevel 1 goto :error
) else (
    echo Reusing existing virtual environment in "%CD%\.venv".
)

echo Upgrading pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :error

echo Installing required packages...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo.
echo Setup complete.
if /I not "%~1"=="--no-pause" pause
exit /b 0

:error
echo.
echo Setup failed.
if /I not "%~1"=="--no-pause" pause
exit /b 1
