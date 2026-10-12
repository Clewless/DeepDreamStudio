"""Smoke-test the installed wheel in isolation from the repository's app package.

Runtime dependencies and Tk must already be installed; Linux needs a display
(for example: xvfb-run -a python scripts/smoke_installed.py).
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    checkout = Path(__file__).resolve().parent.parent
    with tempfile.TemporaryDirectory(prefix="deepdream-installed-") as temporary:
        probe = (
            "from pathlib import Path; import app; "
            f"assert Path(app.__file__).resolve().parent != Path({str(checkout / 'app')!r}), "
            "'Loaded checkout instead of installed package'; "
            "from importlib.metadata import distribution; "
            "d = distribution('deepdream-studio'); "
            "assert {'deepdream', 'deepdream-studio'} <= {e.name for e in d.entry_points}; "
            "print('Installed version:', app.__version__)"
        )
        commands = [
            [sys.executable, "-I", "-c", probe],
            [sys.executable, "-I", "-m", "app.deepdream", "--help"],
            [
                str(
                    Path(sys.executable).parent
                    / ("deepdream.exe" if sys.platform == "win32" else "deepdream")
                ),
                "--help",
            ],
            [
                sys.executable,
                "-I",
                "-c",
                "from app.gui import DeepDreamGUI; "
                "window = DeepDreamGUI(); window.update(); window._on_close()",
            ],
        ]
        for command in commands:
            result = subprocess.run(
                command, cwd=temporary, capture_output=True, text=True, timeout=120
            )
            if result.returncode:
                raise RuntimeError(f"Installed-package smoke test failed:\n{result.stderr}")
    print("Installed wheel: package, CLI, and GUI smoke checks passed")


if __name__ == "__main__":
    main()
