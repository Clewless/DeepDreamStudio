"""Model metadata, default parameters, and filename validation shared by the CLI and GUI."""

import json
import os
import re
from pathlib import Path
from typing import Any, TypedDict

MODEL_ORDER = ("googlenet", "inceptionv3", "vgg16", "vgg19")

# Video container extensions accepted by video mode and zoom animations.
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")

INVALID_FILENAME_CHARS = frozenset('<>:"/\\|?*')
WINDOWS_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)
# Max --name stem length; the full output path is bounded by the CLI's MAX_PATH check.
MAX_NAME_LEN = 150


def filename_stem_error(name, what="--name", max_len=MAX_NAME_LEN):
    """Return an error message for an invalid filename stem, or None if valid."""
    if name in ("", ".", ".."):
        return f"Error: {what} must not be empty, '.', or '..'"
    bad = sorted(INVALID_FILENAME_CHARS & frozenset(name))
    if bad:
        return f"Error: {what} contains invalid filename character(s): {' '.join(bad)}"
    if any(ord(c) < 32 for c in name):
        return f"Error: {what} must not contain control characters"
    if name.split(".")[0].strip().upper() in WINDOWS_RESERVED_NAMES:
        return f"Error: {what} '{name}' is a reserved device name on Windows"
    if name.rstrip(" .") != name:
        return f"Error: {what} must not end with a space or a dot"
    if max_len is not None and len(name) > max_len:
        return f"Error: {what} too long ({len(name)} chars, max {max_len})"
    return None


MODEL_LABELS = {
    "googlenet": "GoogLeNet (InceptionV1)",
    "inceptionv3": "InceptionV3",
    "vgg16": "VGG16",
    "vgg19": "VGG19",
}

# One-line summary printed by the CLI when the weights are loaded.
MODEL_DESCRIPTIONS = {
    "googlenet": "GoogLeNet / InceptionV1: the original 2015 DeepDream model",
    "inceptionv3": "InceptionV3: deeper Inception; input must be >=75px (299px recommended)",
    "vgg16": "VGG16: deep conv stacks, strong hallucinations",
    "vgg19": "VGG19: deeper VGG, detailed textures",
}

# Shorter copy shown on the GUI model card.
MODEL_GUI_DESCRIPTIONS = {
    "googlenet": "The original 2015 DeepDream model",
    "inceptionv3": "Deeper Inception; 299px input recommended",
    "vgg16": "Deep conv stacks, strong hallucinations",
    "vgg19": "Deeper VGG, detailed textures",
}

# Ordered (key, display label) pairs the user may target, per model.
MODEL_LAYERS = {
    # GoogLeNet: direct children of the torchvision GoogLeNet module.
    "googlenet": [
        ("inception3a", "inception3a - fine texture swirls"),
        ("inception3b", "inception3b - soft, wispy patterns"),
        ("inception4a", "inception4a - simple shapes & edges"),
        ("inception4b", "inception4b - shapes forming"),
        ("inception4c", "inception4c - classic dogs, eyes, faces (default)"),
        ("inception4d", "inception4d - stronger hallucinations"),
        ("inception4e", "inception4e - bolder objects"),
        ("inception5a", "inception5a - large, abstract structures"),
        ("inception5b", "inception5b - high-level abstraction"),
    ],
    # InceptionV3: direct children of the torchvision Inception3 module.
    "inceptionv3": [
        ("Mixed_5b", "Mixed_5b - early textures & edges"),
        ("Mixed_5c", "Mixed_5c - fine patterns"),
        ("Mixed_5d", "Mixed_5d - soft swirls"),
        ("Mixed_6a", "Mixed_6a - simple shapes forming"),
        ("Mixed_6b", "Mixed_6b - richer shapes"),
        ("Mixed_6c", "Mixed_6c - objects appearing"),
        ("Mixed_6d", "Mixed_6d - complex feature combinations (default)"),
        ("Mixed_6e", "Mixed_6e - strong hallucinations"),
        ("Mixed_7a", "Mixed_7a - bold, complex objects"),
        ("Mixed_7b", "Mixed_7b - vivid structures"),
        ("Mixed_7c", "Mixed_7c - high-level abstraction"),
    ],
    # VGG16: indices into model.features. 31 layers (0-30).
    #   Block 1 (64ch):  conv+relu 0-1, conv+relu 2-3,  pool 4
    #   Block 2 (128ch): conv+relu 5-6, conv+relu 7-8,  pool 9
    #   Block 3 (256ch): conv+relu 10-11, 12-13, 14-15,  pool 16
    #   Block 4 (512ch): conv+relu 17-18, 19-20, 21-22,  pool 23
    #   Block 5 (512ch): conv+relu 24-25, 26-27, 28-29,  pool 30
    "vgg16": [
        ("features.3", "features.3 - block 1, fine textures"),
        ("features.8", "features.8 - block 2, soft patterns"),
        ("features.15", "features.15 - block 3, shapes & edges"),
        ("features.22", "features.22 - block 4, objects forming (default)"),
        ("features.29", "features.29 - block 5, deep hallucinations"),
    ],
    # VGG19: 37 layers (0-36). Same blocks as VGG16, but blocks 3/4/5 each
    # have 4 conv layers instead of 3.
    #   Block 1 (64ch):  relu 1, 3          pool 4
    #   Block 2 (128ch): relu 6, 8          pool 9
    #   Block 3 (256ch): relu 11, 13, 15, 17  pool 18
    #   Block 4 (512ch): relu 20, 22, 24, 26  pool 27
    #   Block 5 (512ch): relu 29, 31, 33, 35  pool 36
    "vgg19": [
        ("features.3", "features.3 - block 1, fine textures"),
        ("features.8", "features.8 - block 2, soft patterns"),
        ("features.17", "features.17 - block 3, shapes & edges"),
        ("features.26", "features.26 - block 4, objects forming (default)"),
        ("features.35", "features.35 - block 5, deep hallucinations"),
    ],
}

MODEL_RECOMMENDED = {
    "googlenet": "inception4c",
    "inceptionv3": "Mixed_6d",
    "vgg16": "features.22",
    "vgg19": "features.26",
}

MODEL_MIN_SIZE = {
    "googlenet": 15,
    "inceptionv3": 75,
    "vgg16": 32,
    "vgg19": 32,
}

# Oldest torch the engine is tested against; must match the torch floor in
# requirements.txt so the runtime check and the installer agree.
MIN_TORCH_VERSION = (2, 4, 1)

MODEL_DEFAULTS = {
    "googlenet": {
        "iterations": 20,
        "octaves": 0,
        "octave_scale": 1.4,
        "lr": 0.03,
        "jitter": 32,
        "size": 512,
    },
    "inceptionv3": {
        "iterations": 30,
        "octaves": 0,
        "octave_scale": 1.35,
        "lr": 0.025,
        "jitter": 32,
        "size": 512,
    },
    "vgg16": {
        "iterations": 20,
        "octaves": 0,
        "octave_scale": 1.32,
        "lr": 0.018,
        "jitter": 32,
        "size": 512,
    },
    "vgg19": {
        "iterations": 25,
        "octaves": 0,
        "octave_scale": 1.32,
        "lr": 0.018,
        "jitter": 32,
        "size": 512,
    },
}


# H.264 CRF and encoder speed preset per quality name.
class VideoQualityPreset(TypedDict):
    crf: int
    preset: str


VIDEO_QUALITY_PRESETS: dict[str, VideoQualityPreset] = {
    "small": {"crf": 28, "preset": "fast"},
    "balanced": {"crf": 17, "preset": "medium"},
    "lossless": {"crf": 0, "preset": "medium"},
}

VIDEO_FRAMES_META = "meta.json"


def write_video_frames_meta(
    frames_dir: str | Path,
    fps: float | int = 30,
    mode: str = "zoom",
    source: str | None = None,
    width: int = 0,
    height: int = 0,
    model: str | None = None,
    layers: str | list[str] | tuple[str, ...] | None = None,
    size: int | None = None,
    iterations: int | None = None,
    target_class: str | int | None = None,
    class_name: str | None = None,
    guide: str | None = None,
    custom_name: str | None = None,
    zoom_scale: float | None = None,
    zoom_rot: float | None = None,
    zoom_cx: float | None = None,
    zoom_cy: float | None = None,
    video_blend: float | None = None,
    lr: float | None = None,
    jitter: int | None = None,
    seed: int | None = None,
    total_frames: int | None = None,
    **extra: Any,
) -> bool:
    """Write a small meta.json next to a video frame sequence.
    Records the generation parameters so the .mp4 can be encoded later with
    systematic naming. Best-effort: never raises."""
    try:
        os.makedirs(frames_dir, exist_ok=True)
        meta: dict[str, Any] = {
            "mode": mode,
            "fps": float(fps) if fps else 30.0,
            "source": str(source) if source else None,
            "width": int(width) if width else 0,
            "height": int(height) if height else 0,
        }
        if model is not None:
            meta["model"] = str(model)
        if layers is not None:
            if isinstance(layers, (list, tuple)):
                meta["layers"] = [str(lay) for lay in layers]
            else:
                meta["layers"] = str(layers)
        for key, val, cast_fn in (
            ("size", size, int),
            ("iterations", iterations, int),
            ("zoom_scale", zoom_scale, float),
            ("zoom_rot", zoom_rot, float),
            ("zoom_cx", zoom_cx, float),
            ("zoom_cy", zoom_cy, float),
            ("video_blend", video_blend, float),
            ("lr", lr, float),
            ("jitter", jitter, int),
            ("seed", seed, int),
            ("total_frames", total_frames, int),
        ):
            if val is not None:
                try:
                    meta[key] = cast_fn(val)
                except (TypeError, ValueError):
                    pass
        # Keep null distinct from an older metadata file that never recorded
        # the seed; resume compatibility uses it to reject seeded/unseeded mixes.
        meta["seed"] = seed
        if target_class is not None:
            meta["target_class"] = target_class
        if class_name is not None:
            meta["class_name"] = str(class_name)
        if guide is not None:
            meta["guide"] = str(guide)
        if custom_name is not None:
            meta["custom_name"] = str(custom_name)
        for k, v in extra.items():
            if v is not None and k not in meta:
                meta[k] = v

        tmp = os.path.join(frames_dir, VIDEO_FRAMES_META + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        os.replace(tmp, os.path.join(frames_dir, VIDEO_FRAMES_META))
        return True
    except (OSError, TypeError, ValueError):
        return False


def read_video_frames_meta(frames_dir: str | Path | None) -> dict[str, Any]:
    """Read a video frame sequence's meta.json; returns {} when absent."""
    if not frames_dir:
        return {}
    try:
        with open(os.path.join(frames_dir, VIDEO_FRAMES_META), encoding="utf-8") as fh:
            meta = json.load(fh)
        return meta if isinstance(meta, dict) else {}
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}


_RESUME_PATH_KEYS = ("source", "guide")
_RESUME_FLOAT_KEYS = ("zoom_scale", "zoom_rot", "zoom_cx", "zoom_cy", "video_blend", "lr")
_RESUME_OPTIONAL_KEYS = ("target_class", "guide", "seed")


def _norm_resume_value(key: str, val: Any) -> Any:
    if val is None:
        return None
    if key in _RESUME_PATH_KEYS:
        return os.path.normcase(os.path.abspath(str(val)))
    if key == "layers":
        if isinstance(val, (list, tuple)):
            return [str(v) for v in val]
        return [str(val)]
    if key in _RESUME_FLOAT_KEYS:
        try:
            return float(val)
        except (TypeError, ValueError):
            return val
    if key in ("width", "height", "iterations", "jitter", "seed"):
        try:
            return int(val)
        except (TypeError, ValueError):
            return val
    return str(val)


def resume_mismatches(saved: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    """Compare a saved sequence's meta.json against the settings of a resumed run.
    Keys missing from either side are not compared. Returns human-readable
    differences; an empty list means the run can safely resume."""
    problems: list[str] = []
    for key, want in expected.items():
        if key in saved:
            have = saved[key]
        elif key in _RESUME_OPTIONAL_KEYS and "model" in saved:
            have = None
        else:
            continue
        a = _norm_resume_value(key, have)
        b = _norm_resume_value(key, want)
        if a is None and b is None:
            continue
        if key in _RESUME_FLOAT_KEYS and isinstance(a, float) and isinstance(b, float):
            if abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b)):
                continue
        elif a == b:
            continue
        problems.append(f"{key}: saved {have!r}, now {want!r}")
    return problems


def sanitize_stem_part(text: str | None, fallback: str = "dream") -> str:
    """Clean a string for safe inclusion in a filename stem."""
    if not text:
        return fallback
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", str(text)).strip(" ._")
    if not cleaned:
        return fallback
    if cleaned.split(".")[0].strip().upper() in WINDOWS_RESERVED_NAMES:
        return f"{fallback}_{cleaned}"
    return cleaned


def _safe_int(val: Any, default: int = 0) -> int:
    """int(val) with a fallback for None or garbage input."""
    if val is None:
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def build_systematic_video_name(
    source: str | None = None,
    custom_name: str | None = None,
    size: int | None = None,
    width: int | None = None,
    height: int | None = None,
    model: str | None = None,
    layers: str | list[str] | tuple[str, ...] | None = None,
    iterations: int | None = None,
    target_class: str | int | None = None,
    class_name: str | None = None,
    guide: str | None = None,
    mode: str | None = "zoom",
    crf: int | None = None,
    max_len: int = MAX_NAME_LEN,
) -> str:
    """Build a descriptive filename for an exported video.
    Combines input/custom name, resolution, model, layers/guidance, iterations,
    mode, and CRF, sanitized and bounded to *max_len*."""
    if custom_name and str(custom_name).strip():
        base_raw = str(custom_name).strip()
    elif source and str(source).strip():
        base_raw = Path(str(source).strip()).stem
    else:
        base_raw = "video_frames"
    base_clean = sanitize_stem_part(base_raw, fallback="video")

    size_part = ""
    size_int = _safe_int(size)

    if size_int > 0:
        size_part = f"{size_int}px"
    else:
        w_int = _safe_int(width)
        h_int = _safe_int(height)
        if w_int > 0 and h_int > 0:
            size_part = f"{w_int}x{h_int}"

    model_part = ""
    if model:
        m_slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(model).strip().lower()).strip("_")
        if m_slug:
            model_part = m_slug

    target_parts = []
    if layers:
        if isinstance(layers, (list, tuple)):
            layer_strs = [str(lay).strip() for lay in layers if str(lay).strip()]
        else:
            layer_strs = [
                item.strip() for item in re.split(r"[,+\s]+", str(layers)) if item.strip()
            ]
        cleaned_layers = [
            re.sub(r"[^a-zA-Z0-9_-]+", "-", lay.replace(".", "-")).strip("-") for lay in layer_strs
        ]
        cleaned_layers = [lay for lay in cleaned_layers if lay]
        if cleaned_layers:
            target_parts.append("_".join(cleaned_layers))

    if target_class is not None and str(target_class).strip():
        resolved_cname = class_name
        if not resolved_cname:
            try:
                from .imagenet_categories import lookup_category

                res = lookup_category(target_class)
                if res:
                    resolved_cname = res[1]
            except (ImportError, KeyError, ValueError):
                pass
        if resolved_cname:
            cslug = re.sub(r"[^a-zA-Z0-9]+", "-", str(resolved_cname)).strip("-")
        else:
            cslug = re.sub(r"[^a-zA-Z0-9]+", "-", str(target_class)).strip("-")
        if cslug:
            target_parts.append(f"class-{cslug}")

    if guide and str(guide).strip():
        gstem = Path(str(guide).strip()).stem
        gslug = re.sub(r"[^a-zA-Z0-9_]+", "-", gstem).strip("-")
        if gslug:
            target_parts.append(f"guide-{gslug}")

    targets_str = "_".join(target_parts)

    iter_part = ""
    iter_int = _safe_int(iterations)
    if iter_int > 0:
        iter_part = f"i{iter_int}"

    m_str = str(mode).lower() if mode else "zoom"
    mode_part = "zoom" if "zoom" in m_str else "video"

    crf_part = ""
    if crf is not None:
        try:
            crf_part = f"crf{int(crf)}"
        except (TypeError, ValueError):
            pass

    suffix_items = [
        p for p in (size_part, model_part, targets_str, iter_part, mode_part, crf_part) if p
    ]
    suffix = "_" + "_".join(suffix_items) if suffix_items else ""

    if max_len and len(base_clean) + len(suffix) > max_len:
        available = max(8, max_len - len(suffix))
        base_clean = base_clean[:available].rstrip(" ._")
        if not base_clean:
            base_clean = "video"

    stem = f"{base_clean}{suffix}"
    if max_len and len(stem) > max_len:
        stem = stem[:max_len].rstrip(" ._")

    return f"{stem}.mp4"


def resolve_name_conflict(directory: str | Path | None, filename: str) -> str:
    """Return an unused filename in *directory*, appending _1, _2, etc. if needed.
    Case-insensitive; returns *filename* unchanged when *directory* is not an
    existing directory."""
    if not directory or not filename:
        return filename
    dir_path = Path(directory)
    if not dir_path.is_dir():
        return filename

    try:
        existing_lower = {f.name.lower() for f in dir_path.iterdir() if f.is_file()}
    except OSError:
        existing_lower = set()

    if filename.lower() not in existing_lower:
        return filename

    p = Path(filename)
    stem = p.stem
    ext = p.suffix

    match = re.match(r"^(.*)_(\d+)$", stem)
    if match:
        base_stem = match.group(1)
        start_counter = int(match.group(2)) + 1
    else:
        base_stem = stem
        start_counter = 1

    counter = start_counter
    while True:
        candidate = f"{base_stem}_{counter}{ext}"
        if candidate.lower() not in existing_lower:
            return candidate
        counter += 1
