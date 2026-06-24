@echo off
setlocal EnableExtensions

cd /d "%~dp0"

set "PYTHON_CMD="
set "INSTALL_PYTHON=0"
set "NO_PAUSE=0"

:parse_args
if "%~1"=="" goto :args_done
if /I "%~1"=="--install-python" set "INSTALL_PYTHON=1"
if /I "%~1"=="--no-pause" set "NO_PAUSE=1"
shift
goto :parse_args

:args_done
call :find_python

if defined PYTHON_CMD goto :python_ready

echo Python 3 was not found.
echo.

if "%INSTALL_PYTHON%"=="1" goto :install_missing_python

if "%NO_PAUSE%"=="0" (
    choice /C YN /M "Install Python 3.12 for the current user with winget now?"
    if not errorlevel 2 goto :install_missing_python
    echo.
)

echo Install Python 3 for Windows, then run this script again.
echo.
echo To let this script try installing Python 3.12 with winget, run:
echo     %~nx0 --install-python
goto :error

:install_missing_python
call :install_python
if errorlevel 1 goto :error
call :find_python

if not defined PYTHON_CMD (
    echo Python 3 was installed, but it was not found in this command prompt.
    echo Open a new Command Prompt window and run this script again.
    goto :error
)

:python_ready
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
if "%NO_PAUSE%"=="0" pause
exit /b 0

:error
echo.
echo Setup failed.
if "%NO_PAUSE%"=="0" pause
exit /b 1

:find_python
set "PYTHON_CMD="

for %%C in ("py -3" "python" "python3") do (
    if not defined PYTHON_CMD (
        %%~C -c "import sys; raise SystemExit(0 if sys.version_info[0] == 3 else 1)" >nul 2>&1
        if not errorlevel 1 set "PYTHON_CMD=%%~C"
    )
)

for %%P in ("%LocalAppData%\Programs\Python\Python*\python.exe" "%ProgramFiles%\Python*\python.exe" "%ProgramFiles(x86)%\Python*\python.exe") do (
    if not defined PYTHON_CMD (
        if exist "%%~fP" (
            "%%~fP" -c "import sys; raise SystemExit(0 if sys.version_info[0] == 3 else 1)" >nul 2>&1
            if not errorlevel 1 set "PYTHON_CMD="%%~fP""
        )
    )
)

exit /b 0

:install_python
where winget >nul 2>&1
if errorlevel 1 (
    echo winget was not found, so Python cannot be installed automatically.
    echo Install Python 3 from https://www.python.org/downloads/windows/ and run this script again.
    exit /b 1
)

echo Installing Python 3.12 for the current user with winget...
winget install --id Python.Python.3.12 --exact --source winget --scope user --accept-package-agreements --accept-source-agreements
if errorlevel 1 (
    echo winget could not install Python.
    exit /b 1
)

exit /b 0
