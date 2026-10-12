@echo off
setlocal

rem DeepDream Studio - main launcher (GUI).
rem Double-click this file to start the app; no command line needed.

set "IMG_ARG="
if not "%~1"=="" set "IMG_ARG=%~f1"

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

rem .venv is created beside the launcher. Give newcomers a clear error if they
rem try to run from a read-only location such as Program Files or an archive.
2>nul ( >"%SCRIPT_DIR%.deepdream-write-test" echo test )
if not exist "%SCRIPT_DIR%.deepdream-write-test" (
    echo ERROR: This folder is not writable: "%SCRIPT_DIR%"
    echo Move or extract DeepDream Studio to a folder you own, such as Documents,
    echo then run this launcher again. Setup needs to create a .venv folder here.
    pause
    exit /b 1
)
del "%SCRIPT_DIR%.deepdream-write-test" >nul 2>nul

rem Find a Python 3.11-3.14 interpreter to run the bootstrap. Versioned names
rem are tried first so a default "python" pointing at an older (or newer,
rem unsupported) release does not shadow a good install. Each candidate must
rem both execute code and report a supported version: on Windows "python" can
rem resolve to the Microsoft Store stub, which prints a message and exits
rem without ever running an interpreter.
set "PY="
call :find_py "py -3.14"
call :find_py "py -3.13"
call :find_py "py -3.12"
call :find_py "py -3.11"
call :find_py "py -3"
call :find_py "python3.14"
call :find_py "python3.13"
call :find_py "python3.12"
call :find_py "python3.11"
call :find_py "python"
call :find_py "python3"
if not defined PY (
    if exist ".venv\Scripts\python.exe" call :find_py ".venv\Scripts\python.exe"
)
if not defined PY (
    echo ERROR: Python 3.11-3.14 was not found on PATH.
    echo Install it first, then run this launcher again:
    echo   winget install Python.Python.3.14
    echo   or https://www.python.org/downloads/ ^(tick "Add python.exe to PATH"^)
    echo.
    echo See README.md "Install" for details.
    pause
    exit /b 1
)

%PY% app\bootstrap.py --gui
if errorlevel 1 (
    echo.
    echo ERROR: Setup failed. See the messages above.
    pause
    exit /b 1
)

set "LOG=%SCRIPT_DIR%deepdream-launch.log"
echo Launching DeepDream Studio ...
echo You can minimize this window while you use DeepDream Studio.
echo If the Studio window does not appear, open "%LOG%" for the error.
rem pythonw.exe runs without a console window, so its output is redirected to a
rem log file: otherwise a startup crash would exit silently with no message.
if defined IMG_ARG (
    "%SCRIPT_DIR%.venv\Scripts\pythonw.exe" -m app.gui "%IMG_ARG%" > "%LOG%" 2>&1
) else (
    "%SCRIPT_DIR%.venv\Scripts\pythonw.exe" -m app.gui > "%LOG%" 2>&1
)
if errorlevel 1 (
    echo.
    echo ERROR: DeepDream Studio exited with an error. See "%LOG%".
    pause
    exit /b 1
)
exit /b 0

rem Try one interpreter candidate: it must run code and report Python 3.11-3.14.
:find_py
if defined PY exit /b 0
%~1 -c "import sys; sys.exit(0 if (3, 11) <= sys.version_info[:2] <= (3, 14) else 1)" >nul 2>nul && set "PY=%~1"
exit /b 0
