"""First-launch bootstrap: find Python, create/repair the venv, install deps.

Call this with any Python 3 interpreter the launcher can find. It verifies the
version, creates ``.venv`` if needed, repairs a broken or too-old one (after
confirmation, by renaming so nothing is destroyed), and installs dependencies
into that environment via check_deps.py. It never installs or alters system
Python; when Python is missing or too old, exact per-OS commands are printed
for the user to run.

Stdlib only, Python 3.8+ so it can also run under an older interpreter.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
VENV = ROOT / ".venv"
MIN_PY: tuple[int, int] = (3, 11)
# 3.14 has wheels for the pinned dependencies and is supported except on
# macOS. Raise this only once the pinned requirements publish wheels for a new
# Python release; update MACOS_MAX_PY separately when macOS support changes.
MAX_PY: tuple[int, int] = (3, 14)
MACOS_MAX_PY: tuple[int, int] = (3, 13)


def max_supported_python(platform: str | None = None) -> tuple[int, int]:
    """Return the highest supported Python version for the target platform."""
    target_platform = sys.platform if platform is None else platform
    return MACOS_MAX_PY if target_platform == "darwin" else MAX_PY


def supported_range(platform: str | None = None) -> str:
    maximum = max_supported_python(platform)
    return f"{MIN_PY[0]}.{MIN_PY[1]}-{maximum[0]}.{maximum[1]}"


def configure_streams() -> None:
    """Force UTF-8 on stdout/stderr so non-ASCII paths cannot crash setup.

    The launcher's own messages are plain ASCII, but it echoes user paths and
    relays subprocess output, and a fresh install runs under whatever console
    codepage the machine has (cp1252 on most Windows systems).
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="backslashreplace")
            except (OSError, ValueError):
                pass


def venv_python(venv: Path | None = None) -> Path:
    venv = VENV if venv is None else venv
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def python_version(python: Path) -> tuple[int, int] | None:
    try:
        proc = subprocess.run(
            [str(python), "-c", "import sys; print('%d %d' % sys.version_info[:2])"],
            capture_output=True,
            text=True,
            errors="backslashreplace",
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    match = re.match(r"(\d+)\s+(\d+)", proc.stdout.strip())
    return (int(match.group(1)), int(match.group(2))) if match else None


def linux_distro() -> str:
    path = Path("/etc/os-release")
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace").lower()
    ids = ""
    for key in ("id=", "id_like="):
        m = re.search(rf"^{key}\"?([^\n\"]+)", text, re.MULTILINE)
        if m:
            ids += " " + m.group(1)
    return ids


def install_guidance(gui: bool = False) -> list[str]:
    lines = [
        f"DeepDream needs Python {supported_range()}, which was not found.",
        "Install it with the command for your system, then run the launcher again:",
    ]
    # The GUI needs Tkinter, which ships in separate OS packages on Linux.
    # Include them here so a minimal install does not fail twice in a row.
    if sys.platform == "win32":
        lines += [
            "  winget install Python.Python.3.14",
            "  or download from https://www.python.org/downloads/",
            '  (tick "Add python.exe to PATH" in the installer)',
        ]
    elif sys.platform == "darwin":
        # Homebrew ships Tkinter in a separate formula (python-tk@X.Y), so a
        # plain "brew install python" leaves the GUI unable to start.
        tk_brew = " python-tk@3.13" if gui else ""
        lines += [
            f"  brew install python@3.13{tk_brew}",
            "  or install Python 3.13 from https://www.python.org/downloads/",
        ]
    else:
        tk_deb = " python3-tk" if gui else ""
        tk_dnf = " python3-tkinter" if gui else ""
        tk_pac = " tk" if gui else ""
        ids = linux_distro()
        if "debian" in ids or "ubuntu" in ids:
            lines += [f"  sudo apt update && sudo apt install python3 python3-venv{tk_deb}"]
        elif "fedora" in ids or "rhel" in ids:
            lines += [f"  sudo dnf install python3{tk_dnf}"]
        elif "arch" in ids:
            lines += [f"  sudo pacman -S python{tk_pac}"]
        else:
            lines += [
                f"  Debian/Ubuntu:  sudo apt update && sudo apt install python3 python3-venv{tk_deb}",
                f"  Fedora/RHEL:    sudo dnf install python3{tk_dnf}",
                f"  Arch Linux:     sudo pacman -S python{tk_pac}",
            ]
    return lines


def venv_issue() -> str | None:
    """Return why the existing venv is unusable, or None if it is fine."""
    py = venv_python()
    if not py.exists():
        return "missing interpreter"
    version = python_version(py)
    if version is None:
        return "interpreter does not run"
    if version < MIN_PY or version > max_supported_python():
        return f"interpreter is Python {version[0]}.{version[1]} (supported: {supported_range()})"
    return None


def _backup_venv() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = ROOT / f".venv.bad-{stamp}"
    counter = 1
    while target.exists():
        target = ROOT / f".venv.bad-{stamp}-{counter}"
        counter += 1
    shutil.move(str(VENV), str(target))
    return target


def ensure_venv(assume_yes: bool) -> Path | None:
    """Create or repair .venv. Returns its interpreter path, or None on failure."""
    backup: Path | None = None
    issue = venv_issue()
    if issue is not None and VENV.exists():
        print(f"The existing .venv is unusable ({issue}).")
        if not assume_yes:
            try:
                answer = input("Replace it with a fresh virtual environment? [Y/n] ")
            except EOFError:
                answer = "n"
            if answer.strip().lower() in ("n", "no"):
                print("Cannot continue without a working virtual environment.")
                return None
        try:
            backup = _backup_venv()
        except OSError as exc:
            print(f"ERROR: could not move the old environment aside: {exc}")
            print(
                "Close anything using .venv (editors, terminals, other Python "
                "processes), then run the launcher again."
            )
            return None
        print(f"Moved the old environment to {backup.name} (nothing was deleted).")

    if not VENV.exists():
        print(f"Creating a virtual environment in {VENV.name} ...", flush=True)
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "venv", str(VENV)],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            print(f"ERROR: could not run the venv module: {exc}")
            for line in install_guidance():
                print(line)
            return None
        if proc.returncode != 0 or not venv_python().exists():
            print("ERROR: failed to create the virtual environment.")
            detail = (proc.stderr or proc.stdout or "").strip().splitlines()
            if detail:
                print("\n".join(detail[-5:]))
            if sys.platform.startswith("linux"):
                print("On Debian/Ubuntu, the venv module is a separate package:")
                print("  sudo apt install python3-venv")
            if backup is not None:
                print(f"Your previous environment is still available at {backup.name}.")
            return None

    return venv_python()


def run_check_deps(python: Path, gui: bool) -> int:
    cmd = [str(python), str(APP_DIR / "check_deps.py")]
    if gui:
        cmd.append("--gui")
    code = subprocess.run(cmd, cwd=ROOT).returncode
    if code != 0 and gui and not _target_has_tkinter(python):
        # pip cannot provide Tkinter; without this pointer the launcher ends
        # with a generic "Setup failed" and a new user is stuck.
        print()
        for line in tkinter_guidance(python):
            print(line)
    return code


def _target_has_tkinter(python: Path) -> bool:
    try:
        proc = subprocess.run(
            [str(python), "-c", "import tkinter"],
            capture_output=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def tkinter_guidance(python: Path | None = None) -> list[str]:
    head = "Tkinter is missing from that Python install (the GUI needs it; pip cannot install it)."
    if sys.platform == "win32":
        return [
            head,
            "Reinstall Python from https://www.python.org/downloads/",
            'with "tcl/tk and IDLE" enabled, then run the launcher again.',
        ]
    if sys.platform == "darwin":
        # Homebrew's Python formula does not include Tkinter, so reinstalling
        # Python alone would leave the user in the same broken state.
        version = python_version(python) if python is not None else sys.version_info[:2]
        if version is None:
            version = sys.version_info[:2]
        return [
            head,
            "If you installed Python with Homebrew, run:",
            f"  brew install python-tk@{version[0]}.{version[1]}",
            "then run the launcher again.",
            "Alternatively download Python from https://www.python.org/downloads/",
            "(the macOS installer includes Tkinter).",
        ]
    ids = linux_distro()
    if "debian" in ids or "ubuntu" in ids:
        return [
            head,
            "Run: sudo apt update && sudo apt install python3-tk -- then run the launcher again.",
        ]
    if "fedora" in ids or "rhel" in ids:
        return [head, "Run: sudo dnf install python3-tkinter -- then run the launcher again."]
    if "arch" in ids:
        return [head, "Run: sudo pacman -S tk -- then run the launcher again."]
    return [
        head,
        "Debian/Ubuntu:  sudo apt update && sudo apt install python3-tk",
        "Fedora/RHEL:    sudo dnf install python3-tkinter",
        "Arch Linux:     sudo pacman -S tk",
        "Then run the launcher again.",
    ]


def main() -> int:
    configure_streams()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gui", action="store_true", help="also require tkinter")
    parser.add_argument("-y", "--yes", action="store_true", help="do not prompt")
    args = parser.parse_args()

    # A working venv can be used even if the interpreter that launched this
    # script is older (it only needs stdlib to get here).
    if venv_issue() is None:
        return run_check_deps(venv_python(), args.gui)

    current = sys.version_info[:2]
    if current < MIN_PY or current > max_supported_python():
        print(f"This launcher is running under Python {current[0]}.{current[1]}.")
        for line in install_guidance(args.gui):
            print(line)
        return 1

    python = ensure_venv(args.yes)
    if python is None:
        return 1
    return run_check_deps(python, args.gui)


if __name__ == "__main__":
    sys.exit(main())
