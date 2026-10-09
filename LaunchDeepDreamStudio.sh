#!/usr/bin/env bash
# DeepDream Studio - main launcher (GUI).
# Run from a terminal:  ./LaunchDeepDreamStudio.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INITIAL_IMG=""
if [ -n "$1" ]; then
    if [[ "$1" = /* ]]; then
        INITIAL_IMG="$1"
    else
        INITIAL_IMG="$(pwd)/$1"
    fi
fi

# Do not build a virtual environment inside a downloaded/read-only application
# folder (for example, a macOS app translocation path or a mounted archive).
if ! ( : > "$SCRIPT_DIR/.deepdream-write-test" ) 2>/dev/null; then
    echo "ERROR: This folder is not writable: $SCRIPT_DIR"
    echo "Move or extract DeepDream Studio to a folder you own, such as Documents,"
    echo "then run this launcher again. Setup needs to create a .venv folder here."
    exit 1
fi
rm -f "$SCRIPT_DIR/.deepdream-write-test"
cd "$SCRIPT_DIR"

# Find a supported Python interpreter to run the bootstrap. Python 3.14 is
# excluded on macOS. Versioned names
# are tried first so a default "python3" pointing at an older (or newer,
# unsupported) release does not shadow a good install. Each candidate must
# actually run code and report a supported version, not merely be on PATH.
BOOTSTRAP_PY=""
for candidate in python3.14 python3.13 python3.12 python3.11 python3 python; do
    if [ -z "$BOOTSTRAP_PY" ] && command -v "$candidate" >/dev/null 2>&1; then
        if "$candidate" -c "import sys; sys.exit(0 if (3, 11) <= sys.version_info[:2] <= (3, 14) and (sys.platform != 'darwin' or sys.version_info[:2] <= (3, 13)) else 1)" >/dev/null 2>&1; then
            BOOTSTRAP_PY="$candidate"
        fi
    fi
done
if [ -z "$BOOTSTRAP_PY" ] && [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
    BOOTSTRAP_PY="$SCRIPT_DIR/.venv/bin/python"
fi
if [ -z "$BOOTSTRAP_PY" ]; then
    echo "ERROR: No supported Python was found (3.11-3.14; macOS supports 3.11-3.13)."
    echo "Install it with your package manager, then run the launcher again:"
    echo "  Debian/Ubuntu:  sudo apt update && sudo apt install python3 python3-venv python3-tk"
    echo "  Fedora/RHEL:    sudo dnf install python3 python3-tkinter"
    echo "  Arch Linux:     sudo pacman -S python tk"
    echo "  macOS:          brew install python@3.13 python-tk@3.13"
    echo "See README.md \"Install\" for details."
    exit 1
fi

"$BOOTSTRAP_PY" "$SCRIPT_DIR/app/bootstrap.py" --gui || exit 1

echo "Launching DeepDream Studio ..."
if [ -n "$INITIAL_IMG" ]; then
    exec "$SCRIPT_DIR/.venv/bin/python" -m app.gui "$INITIAL_IMG"
else
    exec "$SCRIPT_DIR/.venv/bin/python" -m app.gui
fi
