@echo off
setlocal

rem DeepDream Studio - optional one-shot command-line preset.
rem For normal use, launch the GUI with "Launch.bat" (in the parent folder) instead.

if "%~1"=="" (
    echo This is the optional one-shot command-line preset.
    echo To launch DeepDream Studio normally, run "Launch.bat" in the parent folder.
    echo.
    echo Or drag an image file onto this script to dream it with the preset.
    pause
    exit /b 1
)

if not "%~2"=="" (
    echo ERROR: LaunchCLI.bat accepts one image at a time and uses the preset in this script.
    echo Use the CLI directly ^(python -m app.deepdream^) for batches or custom options.
    pause
    exit /b 1
)

set "IMG=%~f1"
rem Script lives in CLI\, so the repo root is one level up. Resolve it fully so
rem the not-writable message below shows a real path instead of "...\CLI\..\".
for %%I in ("%~dp0..") do set "REPO_DIR=%%~fI\"
cd /d "%REPO_DIR%"

rem .venv is created beside the main launcher. Give newcomers a clear error if they
rem try to run from a read-only location such as Program Files or an archive.
2>nul ( >"%REPO_DIR%.deepdream-write-test" echo test )
if not exist "%REPO_DIR%.deepdream-write-test" (
    echo ERROR: This folder is not writable: "%REPO_DIR%"
    echo Move or extract DeepDream Studio to a folder you own, such as Documents,
    echo then run this launcher again. Setup needs to create a .venv folder here.
    pause
    exit /b 1
)
del "%REPO_DIR%.deepdream-write-test" >nul 2>nul

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
    echo See README.md in the parent folder "Install" for details.
    pause
    exit /b 1
)

%PY% app\bootstrap.py
if errorlevel 1 (
    echo.
    echo ERROR: Setup failed. See the messages above.
    pause
    exit /b 1
)

set MODEL=vgg16
set LAYER=features.22
set ITER=100
set OCT=10
set SIZE=1600

echo.
echo ============================================
echo  DeepDream - command-line preset
echo    image : "%IMG%"
echo    layer : %LAYER%
echo    (first run downloads ~528 MB VGG16 weights)
echo ============================================
echo.

.venv\Scripts\python.exe -m app.deepdream --model %MODEL% --layer %LAYER% --iterations %ITER% --octaves %OCT% --size %SIZE% -- "%IMG%"
if errorlevel 1 (
    echo.
    echo ERROR: DeepDream failed. Check the output above for details.
    pause
    exit /b 1
)

echo.
echo Done! Results are in the "out" folder.
pause
exit /b 0

rem Try one interpreter candidate: it must run code and report Python 3.11-3.14.
:find_py
if defined PY exit /b 0
%~1 -c "import sys; sys.exit(0 if (3, 11) <= sys.version_info[:2] <= (3, 14) else 1)" >nul 2>nul && set "PY=%~1"
exit /b 0
