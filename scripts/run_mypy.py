"""Run mypy from the project's virtual environment.

Pre-commit runs this hook with ``language: system``, so the ``mypy`` executable
is not guaranteed to be on PATH. Locate it from the active virtual environment
(or a conventional local one) and fall back to ``python -m mypy``.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def _venv_candidates() -> list[Path]:
    candidates = [
        Path(os.environ[var]) for var in ("VIRTUAL_ENV", "CONDA_PREFIX") if var in os.environ
    ]
    candidates += [Path(name) for name in (".venv", "venv", ".env", "env")]
    return candidates


def _find_mypy() -> list[str]:
    commands: list[list[str]] = []
    for candidate in _venv_candidates():
        for scripts_dir in ("Scripts", "bin"):
            for exe in ("mypy.exe", "mypy"):
                path = candidate / scripts_dir / exe
                if path.is_file():
                    commands.append([str(path)])
    for candidate in _venv_candidates():
        for scripts_dir in ("Scripts", "bin"):
            for python in ("python.exe", "python", "python3"):
                path = candidate / scripts_dir / python
                if path.is_file():
                    commands.append([str(path), "-m", "mypy"])
    if commands:
        return commands[0]
    mypy = shutil.which("mypy")
    if mypy:
        return [mypy]
    return [sys.executable, "-m", "mypy"]


def main() -> int:
    return subprocess.call([*_find_mypy(), *sys.argv[1:]])


if __name__ == "__main__":
    sys.exit(main())
