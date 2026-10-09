"""Shared pytest configuration, fixtures, and environment setup for DeepDream tests."""

import functools
import sys
from pathlib import Path

import pytest

# Ensure repository root is on sys.path across all test runs
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@functools.cache
def is_tk_display_available() -> bool:
    """Return True if a Tkinter root window can be successfully created."""
    try:
        import tkinter as tk

        root = tk.Tk()
        root.destroy()
        return True
    except Exception:
        return False


@pytest.fixture(autouse=True)
def _auto_skip_gui_if_headless(request: pytest.FixtureRequest) -> None:
    """Automatically skip tests marked with @pytest.mark.gui if running headless."""
    if request.node.get_closest_marker("gui") and not is_tk_display_available():
        pytest.skip("Tk display not available (headless environment)")


@pytest.fixture
def tk_root():
    """Create and yield a temporary Tk root window, safely destroying it on teardown."""
    import tkinter as tk

    try:
        root = tk.Tk()
    except Exception as exc:
        pytest.skip(f"Tk display not available: {exc}")
    yield root
    try:
        root.destroy()
    except Exception:
        pass


@pytest.fixture
def sample_image_path(tmp_path: Path) -> str:
    """Provide a minimal test image path for tests requiring image input."""
    from PIL import Image

    path = str(tmp_path / "test_input.png")
    Image.new("RGB", (64, 48), color=(100, 150, 200)).save(path)
    return path
