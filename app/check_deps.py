"""Verify runtime dependencies and optionally install them on first launch.

Used by the dream/gui launcher scripts. Module specs are checked with
importlib.util.find_spec (no heavy imports), and a deep import of the real
stack runs once per requirements/python fingerprint so broken installs
(e.g. a torch/NumPy ABI mismatch) are caught before the app starts.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MARKER = ROOT / ".deps_ok"

GPU_REQUIREMENTS = "requirements-cuda.txt"
CPU_REQUIREMENTS = "requirements.txt"

REQUIRED = ["torch", "torchvision", "numpy", "PIL", "imageio", "imageio_ffmpeg"]
REQUIRED_GUI = [*REQUIRED, "tkinter"]
OPTIONAL_GUI = ["tkinterdnd2"]


def configure_streams() -> None:
    """Force UTF-8 on stdout/stderr so non-ASCII paths cannot crash the launcher.

    A fresh install runs under whatever console codepage the machine has (cp1252
    on most Windows systems), and the user's home directory or folder name may
    well contain non-ASCII characters.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="backslashreplace")
            except (OSError, ValueError):
                pass


def clear_marker() -> None:
    """Drop the cached verification marker, ignoring an unwritable location."""
    try:
        MARKER.unlink(missing_ok=True)
    except OSError:
        # A read-only install directory (or site-packages under "pip install .")
        # must not turn a cache invalidation into a crash.
        pass


def write_marker(text: str) -> None:
    """Cache a successful verification, ignoring an unwritable location.

    Failing to write the cache only costs a repeat of the import check on the
    next launch, so a read-only directory is never a reason to abort.
    """
    try:
        MARKER.write_text(text, encoding="utf-8")
    except OSError:
        pass


def missing(modules: list[str]) -> list[str]:
    return [m for m in modules if importlib.util.find_spec(m) is None]


def has_nvidia_gpu() -> bool:
    if sys.platform == "darwin":
        return False
    if shutil.which("nvidia-smi"):
        return True
    # nvidia-smi is not always on PATH (containers, custom installs), so check
    # the usual locations before falling back to CPU wheels.
    for candidate in (
        Path("/usr/lib/wsl/lib/nvidia-smi"),
        Path("/usr/bin/nvidia-smi"),
        Path(r"C:\Windows\System32\nvidia-smi.exe"),
        Path(r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"),
    ):
        if candidate.exists():
            return True
    return False


def pick_requirements() -> str:
    return GPU_REQUIREMENTS if has_nvidia_gpu() else CPU_REQUIREMENTS


def in_venv() -> bool:
    return sys.prefix != sys.base_prefix or hasattr(sys, "real_prefix")


def fingerprint(requirements: str, required: list[str]) -> str:
    h = hashlib.sha256()
    h.update(sys.version.split()[0].encode())
    h.update(" ".join(sorted(required)).encode())
    for name in sorted({requirements, "requirements.txt"}):
        path = ROOT / name
        if path.exists():
            h.update(name.encode())
            h.update(path.read_bytes())
    return h.hexdigest()[:16]


def deep_check_ok(required: list[str], quiet: bool = False) -> bool:
    req_file = pick_requirements()
    target_fp = fingerprint(req_file, required)
    if MARKER.exists():
        try:
            cached_fps = {
                line.strip()
                for line in MARKER.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
            if target_fp in cached_fps:
                return True
        except OSError:
            pass
    if not quiet:
        print(
            "Verifying that dependencies load correctly "
            "(first launch only; this can take a minute, please wait) ...",
            flush=True,
        )
    # Import the GUI entry point only for GUI launches: importing it requires
    # Tkinter, which is not available in every CLI-only Python installation.
    probe_imports = [*(f"import {m}" for m in required)]
    if "tkinter" in required:
        probe_imports.append("import app.gui")
    probe = "; ".join(probe_imports)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=ROOT,
            capture_output=True,
            text=True,
            errors="backslashreplace",
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        clear_marker()
        print(
            "Dependency check timed out (a dependency, often torch/CUDA, may be "
            "hung or misconfigured). Try reinstalling:",
            f"{sys.executable} -m pip install -r {pick_requirements()}",
        )
        return False
    if proc.returncode == 0:
        fps_to_record = {target_fp}
        if "tkinter" in required:
            # A successful GUI probe imports all core modules plus tkinter and app.gui;
            # record the CLI fingerprint as well to prevent cache ping-pong between launchers.
            fps_to_record.add(fingerprint(req_file, REQUIRED))
        write_marker("\n".join(sorted(fps_to_record)) + "\n")
        if not quiet:
            print("Dependencies look good.", flush=True)
        return True
    clear_marker()
    print("Dependencies are installed but failed to import correctly:")
    print("\n".join(proc.stderr.strip().splitlines()[-5:]))
    return False


def confirm_install(requirements: str, assume_yes: bool) -> bool:
    """Ask before touching the environment. Returns False if the user declined."""
    if not assume_yes:
        try:
            answer = input(f"Install missing dependencies from {requirements}? [Y/n] ")
        except EOFError:
            answer = "n"
        if answer.strip().lower() in ("n", "no"):
            print("Cannot continue without the required packages.")
            return False
    if not in_venv():
        print(
            "WARNING: not running inside a virtual environment (.venv).\n"
            "Installing anyway may modify your system Python. "
            "Prefer: python -m venv .venv, then rerun the launcher."
        )
        if assume_yes:
            return True
        try:
            confirm = input("Continue installing into system Python? [y/N] ")
        except EOFError:
            confirm = "n"
        if confirm.strip().lower() not in ("y", "yes"):
            return False
    return True


def install(requirements: str, required: list[str], gpu_detected: bool | None = None) -> int:
    if gpu_detected is None:
        gpu_detected = has_nvidia_gpu()
    print(
        f"Installing from {requirements} "
        f"({'NVIDIA GPU detected' if gpu_detected else 'no NVIDIA GPU detected'}) ...\n"
        "This downloads packages from the internet and may take several minutes; "
        "pip output follows.",
        flush=True,
    )
    # An old bundled pip cannot resolve current torch/numpy wheels and fails
    # with a confusing "no matching distribution" error. Best effort: if the
    # upgrade itself fails (e.g. offline), the install below reports the issue.
    print("Upgrading pip first ...", flush=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
        cwd=ROOT,
    )
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(ROOT / requirements)],
        cwd=ROOT,
    )
    if proc.returncode != 0:
        print(f"ERROR: pip install failed with exit code {proc.returncode}.")
        return proc.returncode
    importlib.invalidate_caches()
    still = missing(required)
    if still:
        print(f"ERROR: still missing after install: {', '.join(still)}")
        return 1
    clear_marker()
    return 0


def install_with_fallback(assume_yes: bool, required: list[str]) -> int:
    """Install dependencies, falling back to the CPU wheels if the GPU build fails.

    The CUDA wheels live on a separate index and are a multi-gigabyte download, so
    a blocked, throttled or partially mirrored network there must not leave a new
    user with no way to run the app at all: the standard wheels work everywhere,
    just without GPU acceleration.
    """
    preferred = pick_requirements()
    if not confirm_install(preferred, assume_yes):
        return 1
    if install(preferred, required) == 0:
        return 0
    if preferred == CPU_REQUIREMENTS:
        return 1
    print()
    print("The NVIDIA (CUDA) packages could not be installed.")
    print(
        "Trying the standard CPU packages instead - DeepDream Studio will run, "
        "just without GPU acceleration."
    )
    if install(CPU_REQUIREMENTS, required, gpu_detected=False) != 0:
        return 1
    print()
    print("Installed the CPU packages. To retry GPU acceleration later, run:")
    print(f"  {sys.executable} -m pip install -r {GPU_REQUIREMENTS}")
    return 0


def tkinter_hint() -> list[str]:
    """Per-OS commands for the Tkinter package, which pip cannot provide."""
    if sys.platform == "win32":
        return ["Reinstall Python from https://www.python.org/downloads/ with Tcl/Tk enabled."]
    if sys.platform == "darwin":
        # Homebrew ships Tkinter as a separate formula.
        version = f"{sys.version_info[0]}.{sys.version_info[1]}"
        return [f"macOS with Homebrew:  brew install python-tk@{version}"]
    return [
        "Debian/Ubuntu:  sudo apt install python3-tk",
        "Fedora/RHEL:    sudo dnf install python3-tkinter",
        "Arch Linux:     sudo pacman -S tk",
    ]


def main() -> int:
    configure_streams()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gui", action="store_true", help="also require tkinter")
    parser.add_argument("-y", "--yes", action="store_true", help="install without asking")
    args = parser.parse_args()

    required = REQUIRED_GUI if args.gui else REQUIRED
    absent = missing(required)
    if absent:
        print(f"Missing required packages: {', '.join(absent)}")
        if "tkinter" in absent:
            # pip cannot provide tkinter, so the OS package manager must.
            for line in tkinter_hint():
                print(line)
            return 1
        if install_with_fallback(args.yes, required) != 0:
            return 1
        if not deep_check_ok(required):
            return 1
        return 0

    if args.gui:
        absent_optional = missing(OPTIONAL_GUI)
        if absent_optional:
            print(
                f"Note: optional package(s) not installed: {', '.join(absent_optional)} "
                "(drag-and-drop disabled; browse-to-open still works)."
            )

    if not deep_check_ok(required):
        print(f"Try reinstalling: {sys.executable} -m pip install -r {pick_requirements()}")
        try:
            answer = input(f"Reinstall from {pick_requirements()} now? [y/N] ")
        except EOFError:
            answer = "n"
        if answer.strip().lower() in ("y", "yes"):
            if install_with_fallback(True, required) != 0:
                return 1
            if not deep_check_ok(required):
                return 1
            return 0
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
