#!/usr/bin/env bash
# DeepDream Studio - optional one-shot command-line preset.
# For normal use, launch the GUI with ./Launch.sh (in the parent folder) instead.
set -e

if [ -z "$1" ]; then
    echo "This is the optional one-shot command-line preset."
    echo "To launch DeepDream Studio normally, run ./Launch.sh in the parent folder."
    echo ""
    echo "Usage: ./CLI/LaunchCLI.sh <image_path>"
    exit 1
fi

if [ -n "$2" ]; then
    echo "ERROR: LaunchCLI.sh accepts one image at a time and uses the preset in this script."
    echo "Use the CLI directly (python -m app.deepdream) for batches or custom options."
    exit 1
fi

if [[ "$1" = /* ]]; then
    IMG="$1"
else
    IMG="$(pwd)/$1"
fi

# Script lives in CLI/, so the repo root is one level up.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Setup creates .venv alongside the main launcher, so fail with a useful message
# rather than exposing a cryptic venv/pip permission error.
if ! ( : > "$REPO_DIR/.deepdream-write-test" ) 2>/dev/null; then
    echo "ERROR: This folder is not writable: $REPO_DIR"
    echo "Move or extract DeepDream Studio to a folder you own, such as Documents,"
    echo "then run this launcher again. Setup needs to create a .venv folder here."
    exit 1
fi
rm -f "$REPO_DIR/.deepdream-write-test"
cd "$REPO_DIR"

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
if [ -z "$BOOTSTRAP_PY" ] && [ -x "$REPO_DIR/.venv/bin/python" ]; then
    BOOTSTRAP_PY="$REPO_DIR/.venv/bin/python"
fi
if [ -z "$BOOTSTRAP_PY" ]; then
    echo "ERROR: No supported Python was found (3.11-3.14; macOS supports 3.11-3.13)."
    echo "Install it with your package manager, then run the launcher again:"
    echo "  Debian/Ubuntu:  sudo apt update && sudo apt install python3 python3-venv"
    echo "  Fedora/RHEL:    sudo dnf install python3"
    echo "  Arch Linux:     sudo pacman -S python"
    echo "  macOS:          brew install python@3.13"
    echo "See ../README.md \"Install\" for details."
    exit 1
fi

"$BOOTSTRAP_PY" "$REPO_DIR/app/bootstrap.py" || exit 1

MODEL="vgg16"
LAYER="features.22"
ITER=100
OCT=10
SIZE=1600

echo ""
echo "============================================"
echo " DeepDream - command-line preset"
echo "   image : $IMG"
echo "   layer : $LAYER"
echo "   (first run downloads ~528 MB VGG16 weights)"
echo "============================================"
echo ""

exec "$REPO_DIR/.venv/bin/python" -m app.deepdream --model "$MODEL" --layer "$LAYER" --iterations "$ITER" --octaves "$OCT" --size "$SIZE" -- "$IMG"
