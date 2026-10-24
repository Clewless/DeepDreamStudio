"""User settings persistence for DeepDream Studio.

Saves and restores GUI preferences (model, parameters, output directory,
window geometry, recent files) as JSON. Settings live next to the output
folder so source/launcher installs and pip installs each keep their own.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .gui_helpers import output_base_dir

SETTINGS_FILENAME = "deepdream-settings.json"
MAX_RECENT_FILES = 10

# Default settings; used when no settings file exists or a key is missing.
DEFAULTS: dict[str, Any] = {
    "model_key": "vgg16",
    "layer_key": None,  # None = use model's recommended layer
    "iterations": None,  # None = use model defaults
    "octaves": None,
    "octave_scale": None,
    "lr": None,
    "size": None,
    "jitter": None,
    "out_dir": "",
    "mode": "Single Image",
    "single_image_iters": None,
    "guidance_mode": "Unguided",
    "guide_path": "",
    "target_class": "",
    "video_quality": "Balanced",
    "video_crf": 17,
    "zoom_frames": 60,
    "zoom_scale": 1.03,
    "zoom_rot": 0.0,
    "zoom_fps": 30,
    "zoom_cx": 0.5,
    "zoom_cy": 0.5,
    "video_blend": 0.5,
    "timelapse": False,
    "save_frames": False,
    "no_cuda": False,
    "window_geometry": None,  # "WxH+X+Y" or None
    "recent_files": [],
}


def settings_path() -> Path:
    """Return the path to the settings JSON file."""
    return Path(output_base_dir()) / SETTINGS_FILENAME


def load_settings() -> dict[str, Any]:
    """Load settings from disk; returns defaults merged with saved values."""
    result = dict(DEFAULTS)
    try:
        path = settings_path()
        if path.is_file():
            with open(path, encoding="utf-8") as fh:
                saved = json.load(fh)
            if isinstance(saved, dict):
                for key, value in saved.items():
                    if key in DEFAULTS:
                        result[key] = value
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        pass
    return result


def save_settings(settings: dict[str, Any]) -> bool:
    """Save settings to disk. Returns True on success, False on failure."""
    try:
        path = settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        # Only persist known keys, in a stable order.
        to_save = {key: settings.get(key, DEFAULTS.get(key)) for key in DEFAULTS}
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(to_save, fh, indent=2)
        os.replace(tmp, path)
        return True
    except (OSError, TypeError, ValueError):
        return False


def add_recent_file(settings: dict[str, Any], path: str) -> list[str]:
    """Add a file to the recent-files list (most recent first, capped). Returns the new list."""
    recent = list(settings.get("recent_files") or [])
    # Normalize and deduplicate (case-insensitive on Windows).
    norm = os.path.normcase(os.path.abspath(path))
    recent = [p for p in recent if os.path.normcase(os.path.abspath(p)) != norm]
    recent.insert(0, path)
    recent = recent[:MAX_RECENT_FILES]
    settings["recent_files"] = recent
    return recent


def filter_existing_files(files: list[str]) -> list[str]:
    """Return only the files in the list that still exist on disk."""
    return [p for p in files if p and os.path.isfile(p)]
