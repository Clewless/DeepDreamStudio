"""DeepDream Studio: PyTorch implementation of the classic 2015 DeepDream effect.

Uses torchvision ImageNet weights. Supports GoogLeNet (Inception v1), InceptionV3,
VGG16, and VGG19.

Usage:
    python -m app.deepdream photo.jpg --layer inception5b --octaves 0 --iterations 20

This is a package module using relative imports; run it as ``app.deepdream``
from the repo root rather than invoking the file directly.
"""

import argparse
import importlib.metadata
import math
import os
import random
import re
import subprocess
import sys
import time
from collections.abc import Collection
from dataclasses import dataclass
from typing import Any, NoReturn, cast

import numpy as np
import torch
import torchvision.models as models
from PIL import Image

from . import __version__, dream_engine
from .dream_engine import MAX_OCTAVES as MAX_OCTAVES
from .dream_engine import (
    MIN_OCTAVE_SIZE,
    TARGET_BASE_SIZE,
    DreamCallbacks,
    GuideFeatures,
    plan_octaves,
    validate_input_size,
)
from .dream_engine import auto_octave_count as auto_octave_count
from .dream_engine import octave_truncation_note as octave_truncation_note
from .dream_engine import resize as resize
from .media_io import (
    MEAN,
    load_image,
    load_image_native_size,
    load_image_with_size,
    pil_to_tensor,
    resize_dims,
    save_image,
    save_preview_frame,
    tensor_to_np,
    tensor_to_pil,
)
from .media_io import STD as STD
from .media_io import atomic_tmp_path as _atomic_tmp_path
from .media_io import load_pil_image as load_pil_image
from .media_io import publish_media as _publish_media
from .media_io import remove_stale_temps as _remove_stale_temps
from .media_io import save_frame_atomic as _save_frame_atomic
from .model_data import (
    MAX_NAME_LEN,
    MIN_TORCH_VERSION,
    MODEL_MIN_SIZE,
    VIDEO_EXTENSIONS,
    VIDEO_FRAMES_META,
    VIDEO_QUALITY_PRESETS,
    build_systematic_video_name,
    filename_stem_error,
    read_video_frames_meta,
    resolve_name_conflict,
    resume_mismatches,
    write_video_frames_meta,
)
from .models import (
    MODEL_REGISTRY,
    Device,
    build_model,
    forward_with_activations,
    resolve_layer,
    run_to_layers,
)
from .models import _forward_googlenet as _forward_googlenet
from .models import build_googlenet as build_googlenet
from .protocol import (
    EV_DONE,
    EV_ERROR,
    EV_FRAME,
    EV_ITERATION,
    EV_OCTAVE,
    EV_PID,
    EV_PREVIEW,
    EV_STATUS,
    format_event,
)


def _emit_structured_events() -> bool:
    """Keep event lines out of an interactive terminal, but in piped output.

    The GUI and integration tests read events from a pipe, while a human running
    the CLI directly would otherwise see the NUL-prefixed JSON on every step.
    """
    stream = sys.stdout
    if stream is None:
        return False
    isatty = getattr(stream, "isatty", None)
    if isatty is None:
        return True
    try:
        return not bool(isatty())
    except (OSError, ValueError):
        return True


def emit_event(event: str, **fields: Any) -> None:
    """Emit a structured progress event for the GUI, alongside normal output.

    Events are printed as text so the GUI can parse progress, but they are
    suppressed when stdout is an interactive terminal. A human still sees the
    ordinary progress lines; scripts can pipe the command to keep the events.
    """
    if not _emit_structured_events():
        return
    print(format_event(event, **fields), flush=True)


def _configure_streams() -> None:
    """Ensure standard output and error streams handle UTF-8 cleanly."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="backslashreplace")
            except OSError:
                pass


def _parse_version(v_str: str) -> tuple[int, ...]:
    """Parse a semver-like version string into a comparable tuple of integers."""
    parts = tuple(int(x) for x in v_str.split("+")[0].split(".")[:3] if x.isdigit())
    return parts + (0,) * (3 - len(parts))


def check_runtime_environment() -> None:
    """Verify minimum runtime requirements before CLI execution."""
    try:
        # Keep in step with the torch floor in requirements.txt: the engine
        # relies on antialiased resizing and torch.isfinite, both long stable.
        if _parse_version(importlib.metadata.version("torch")) < MIN_TORCH_VERSION:
            sys.stderr.write(
                f"Error: PyTorch >= {MIN_TORCH_VERSION[0]}.{MIN_TORCH_VERSION[1]} is required. "
                "Reinstall the dependencies with pip install -r requirements.txt\n"
            )
            sys.exit(1)
    except importlib.metadata.PackageNotFoundError:
        pass


try:  # Bundled with the repo; tolerate it being absent.
    from .imagenet_categories import CATEGORY_ALIASES
except (ImportError, AttributeError):  # pragma: no cover
    CATEGORY_ALIASES = {}

try:
    import imageio
    import imageio.v3 as iio
    import imageio_ffmpeg

    _VIDEO_AVAILABLE = True
except ImportError:
    _VIDEO_AVAILABLE = False

# Keep child processes (the ffmpeg audio remux below) from flashing a console
# window when this script is launched without one, as the GUI does.
# getattr: CREATE_NO_WINDOW only exists on Windows, and the sys.platform
# guard below is runtime-only, so mypy on Linux/macOS needs the fallback.
_NO_WINDOW: dict[str, Any] = (
    {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)} if sys.platform == "win32" else {}
)

DEFAULT_MODEL = "vgg16"
_activation_loss = dream_engine.activation_loss

# Filename-stem validation lives in model_data.py (shared with the GUI).
# Windows classic MAX_PATH is 260, macOS PATH_MAX is 1024, and Linux allows
# 4096. Using the Linux figure everywhere else disables the pre-flight check on
# the platform with the shortest limit, where deeply nested paths are common.
MAX_PATH = 260 if sys.platform == "win32" else (1024 if sys.platform == "darwin" else 4096)

# Valid ffmpeg -preset values for H.264, checked by resolve_video_quality().
FFMPEG_PRESETS = frozenset(
    {
        "ultrafast",
        "superfast",
        "veryfast",
        "faster",
        "fast",
        "medium",
        "slow",
        "slower",
        "veryslow",
        "placebo",
    }
)

# Audio-remux ffmpeg re-encodes the audio track to AAC; 600s is a stuck-process ceiling.
FFMPEG_REMUX_TIMEOUT = 600.0


def validate_filename_stem(name, what="--name", max_len=MAX_NAME_LEN):
    """Reject a filename stem that would fail (or silently misbehave) when saved."""
    error = filename_stem_error(name, what=what, max_len=max_len)
    if error:
        raise SystemExit(error)


def _path_len_utf16(path):
    """Length in UTF-16 code units, what Windows counts for MAX_PATH."""
    return len(path.encode("utf-16-le", errors="surrogatepass")) // 2


def validate_output_path(path, extra_suffix=""):
    """Reject an output path that would exceed the OS maximum path length when saved."""
    full = os.path.abspath(path)
    if full.startswith("\\\\?\\"):
        full = full[4:]
    if sys.platform == "win32":
        length = _path_len_utf16(full + extra_suffix)
        limit_name = f"Windows MAX_PATH is {MAX_PATH - 1}"
    else:
        # surrogatepass (like the Windows branch above): lone surrogates can
        # appear in synthetic paths, and must not crash the measurement.
        length = len((full + extra_suffix).encode("utf-8", errors="surrogatepass"))
        limit_name = f"system PATH_MAX is {MAX_PATH - 1}"
    if length >= MAX_PATH:
        raise SystemExit(
            f"Error: Output path too long ({length} units; "
            f"{limit_name}). "
            f"Use a shorter --name or --out."
        )


def ensure_output_dir(path):
    """Create the output directory, failing with a friendly message."""
    if os.path.exists(path) and not os.path.isdir(path):
        raise SystemExit(f"Error: output path '{path}' exists and is not a directory")
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as exc:
        raise SystemExit(f"Error: could not create output directory '{path}': {exc}") from exc


def _warn_on_overwrite(path: str) -> None:
    """Warn before replacing an existing video/timelapse output."""
    if path and os.path.exists(path):
        print(
            f"Warning: output file '{path}' already exists and will be "
            "overwritten. Use --name or --out to keep the previous result.",
            file=sys.stderr,
            flush=True,
        )


def apply_zoom_and_rotate(t, zoom_scale=1.03, zoom_rot=0.0, zoom_cx=0.5, zoom_cy=0.5):
    scale = 1.0 if zoom_scale is None else zoom_scale
    rot = 0.0 if zoom_rot is None else zoom_rot
    cx = 0.5 if zoom_cx is None else zoom_cx
    cy = 0.5 if zoom_cy is None else zoom_cy
    if not math.isfinite(scale) or scale < 1.0:
        raise ValueError(f"zoom_scale must be >= 1.0, got {zoom_scale}")
    if not math.isfinite(rot):
        raise ValueError(f"zoom_rot must be finite, got {zoom_rot}")
    for _name, _v in (("zoom_cx", cx), ("zoom_cy", cy)):
        if not math.isfinite(_v):
            raise ValueError(f"{_name} must be finite, got {_v}")
    # A no-op zoom must not round-trip through PIL: the uint8 quantization
    # error accumulates over dozens of animation frames.
    if scale <= 1.0 and rot == 0.0:
        return t.detach().clone()
    pil = tensor_to_pil(t)
    w, h = pil.size
    resample_bicubic = getattr(Image, "Resampling", Image).BICUBIC
    resample_lanczos = getattr(Image, "Resampling", Image).LANCZOS
    # Fill borders with the ImageNet mean color so rotation and out-of-bounds
    # crops do not introduce high-contrast edges into the feedback loop.
    fill_color = tuple(int(round(c * 255)) for c in MEAN)
    if scale > 1.0:
        nw = max(w + 1, int(round(w * scale)))
        nh = max(h + 1, int(round(h * scale)))
        zoomed = pil.resize((nw, nh), resample_lanczos)
        if rot != 0.0:
            zoomed = zoomed.rotate(rot, resample=resample_bicubic, fillcolor=fill_color)
        # Zoom toward the focal point (cx, cy) in normalized 0..1 coords; the
        # focal pixel stays put while the rest expands. cx=cy=0.5 is the legacy
        # center zoom. A focus outside [0, 1] composites onto a neutral
        # background rather than wrapping (tiling would show a seam).
        left = int(round(float(cx) * (nw - w)))
        top = int(round(float(cy) * (nh - h)))
        if 0 <= left <= nw - w and 0 <= top <= nh - h:
            pil = zoomed.crop((left, top, left + w, top + h))
        else:
            canvas = Image.new("RGB", (w, h), fill_color)
            # Overlap of the crop box with the scaled image, in scaled coords.
            sx0 = max(0, left)
            sy0 = max(0, top)
            sx1 = min(nw, left + w)
            sy1 = min(nh, top + h)
            if sx1 > sx0 and sy1 > sy0:
                dx0 = sx0 - left
                dy0 = sy0 - top
                canvas.paste(zoomed.crop((sx0, sy0, sx1, sy1)), (dx0, dy0))
            pil = canvas
    elif rot != 0.0:
        pil = pil.rotate(rot, resample=resample_bicubic, fillcolor=fill_color)
    return pil_to_tensor(pil)


def get_imagenet_categories():
    """Return the 1000 ImageNet category names from torchvision metadata or bundled module."""
    try:
        from .imagenet_categories import IMAGENET_CATEGORIES

        if IMAGENET_CATEGORIES:
            return list(IMAGENET_CATEGORIES)
    except (ImportError, AttributeError):
        pass
    try:
        return list(models.GoogLeNet_Weights.IMAGENET1K_V1.meta.get("categories", []))
    except (AttributeError, TypeError):
        return []


def resolve_target_class(spec):
    """Resolve an integer class ID or category name to (class_id, class_name)."""
    if spec is None:
        return None, None
    spec_clean = str(spec).strip().lower()
    if not spec_clean:
        return None, None
    cats = get_imagenet_categories()
    if not cats:
        raise SystemExit("Error: ImageNet category metadata is unavailable.")

    # Parse compound formats like "#130: flamingo", "130: flamingo", or "130 - flamingo"
    for sep in (":", "—", " - "):
        if sep in spec_clean:
            parts = spec_clean.split(sep, 1)
            candidate_id = parts[0].strip("# ").strip()
            if candidate_id.isdigit():
                idx = int(candidate_id)
                if 0 <= idx < len(cats):
                    label_part = parts[1].strip().strip("# ").strip()
                    if label_part:
                        cat_lower = cats[idx].lower()
                        alias_hit = CATEGORY_ALIASES.get(label_part) == idx
                        matches_label = (
                            label_part == cat_lower
                            or label_part in cat_lower
                            or cat_lower in label_part
                            or alias_hit
                        )
                        if not matches_label:
                            orig_spec = str(spec).strip()
                            orig_label = orig_spec.split(sep, 1)[1].strip()
                            print(
                                f"Warning: '{orig_spec}' names two different classes: "
                                f"#{idx} is '{cats[idx]}', not '{orig_label}'. "
                                f"Ignoring '{orig_label}' and dreaming '{cats[idx]}'. "
                                f"To dream '{orig_label}', pass it on its own "
                                f'via --class "{orig_label}".',
                                file=sys.stderr,
                            )
                    return idx, cats[idx]
                spec_clean = parts[1].strip()
                break

    candidate_num = spec_clean.lstrip("# ").strip()
    if candidate_num.isdigit():
        idx = int(candidate_num)
        if 0 <= idx < len(cats):
            return idx, cats[idx]
        raise SystemExit(f"Error: Target class index {idx} out of range (0-{len(cats) - 1}).")
    # Common alternative names (e.g. "monarch butterfly" -> class "monarch").
    alias_idx = CATEGORY_ALIASES.get(spec_clean)
    if alias_idx is not None and 0 <= alias_idx < len(cats):
        return alias_idx, cats[alias_idx]
    matches = [i for i, c in enumerate(cats) if spec_clean in c.lower()]
    if not matches:
        for alias, idx in CATEGORY_ALIASES.items():
            if spec_clean in alias and 0 <= idx < len(cats):
                return idx, cats[idx]
        raise SystemExit(
            f"Error: No ImageNet class found matching '{spec}'. "
            f"Examples: 'flamingo', 'monarch butterfly', 'tiger', 'pizza'."
        )
    for i in matches:
        if cats[i].lower() == spec_clean:
            return i, cats[i]
    return matches[0], cats[matches[0]]


def extract_guide_features(model, model_key, guide_path, targets, device, max_size=512):
    """Load guide image and extract normalized activation vectors for target layers."""
    if not os.path.isfile(guide_path):
        raise SystemExit(f"Error: Guide image not found: {guide_path}")
    guide_img = load_image(guide_path, max_size=max_size).to(device)
    validate_input_size(
        guide_img.shape[2],
        guide_img.shape[3],
        model_key,
        what="Guide image",
    )
    with torch.no_grad():
        acts = run_to_layers(model, model_key, guide_img, targets)
        guide_features = {}
        for name, a in acts.items():
            fg = a[0].reshape(a.shape[1], -1)  # (C, N)
            fg_norm = fg / (fg.norm(p=2, dim=0, keepdim=True) + 1e-7)
            guide_features[name] = fg_norm
    return guide_features


def _count_resumable_frames(frames_dir, limit=None):
    """Count the leading run of readable frame_*.png files. The first unreadable
    frame (a write cut short by a kill) is removed so it gets regenerated."""
    k = 0
    while limit is None or k < limit:
        cand = os.path.join(frames_dir, f"frame_{k + 1:06d}.png")
        if not os.path.isfile(cand):
            break
        try:
            with Image.open(cand) as test_img:
                test_img.verify()
            k += 1
        except Exception:
            try:
                os.remove(cand)
            except OSError:
                pass
            break
    return k


def _advance_jitter_rng(frame_count, iterations_per_frame, jitter):
    """Consume the jitter draws belonging to frames already present on disk."""
    if frame_count <= 0 or iterations_per_frame <= 0 or jitter <= 0:
        return
    for _ in range(frame_count * iterations_per_frame):
        random.randint(-jitter, jitter)
        random.randint(-jitter, jitter)


def _check_resume_compatible(frames_dir, count, expected):
    """Refuse to resume onto frames generated with different settings.
    Mixing them would yield a visibly inconsistent sequence, or a late encode
    failure when frame sizes differ."""
    problems = resume_mismatches(read_video_frames_meta(frames_dir), expected)
    want_size = (expected.get("width"), expected.get("height"))
    if all(want_size):
        for frame_idx in range(1, count + 1):
            frame_path = os.path.join(frames_dir, f"frame_{frame_idx:06d}.png")
            with Image.open(frame_path) as frame:
                if frame.size != want_size:
                    problems.append(
                        f"frame_{frame_idx:06d}.png is {frame.size[0]}x{frame.size[1]}, "
                        f"this run produces {want_size[0]}x{want_size[1]}"
                    )
                    break
    if problems:
        details = "\n".join(f"  - {p}" for p in problems)
        raise SystemExit(
            f"Error: cannot resume in '{frames_dir}': the saved frames were made "
            f"with different settings:\n{details}\n"
            "Rerun with the original settings, or start fresh without --resume "
            "(or use a different --frames-dir)."
        )


def _video_canvas_size(size, max_size):
    """Working (w, h) for a video: capped by max_size, rounded up to even for H.264."""
    w, h = resize_dims(size, max_size) or size
    return (w + w % 2, h + h % 2)


def _first_video_frame_size(video_path):
    """(w, h) of a video's first frame, decoding only that frame."""
    frames = iio.imiter(os.path.abspath(video_path))
    try:
        first = next(frames)
    except StopIteration:
        raise SystemExit(f"Error: Video has no frames: {video_path}") from None
    finally:
        close = getattr(frames, "close", None)
        if close is not None:
            close()
    return (int(first.shape[1]), int(first.shape[0]))


def _append_saved_video_frames(writer, frames_dir, count, scale=1):
    """Seed a resumed MP4 encode with the frames that were already generated."""
    for frame_idx in range(1, count + 1):
        frame_path = os.path.join(frames_dir, f"frame_{frame_idx:06d}.png")
        with Image.open(frame_path) as frame:
            writer.append_data(_upscale_frame(frame.convert("RGB"), scale))


def _video_output_params(crf, preset):
    """ffmpeg output parameters for an H.264 encode; unset values are omitted."""
    params = []
    if crf is not None:
        params += ["-crf", str(crf)]
    if preset is not None:
        params += ["-preset", str(preset)]
    return params


def _open_video_writer(path, fps, output_params):
    """imageio writer for an H.264 .mp4 with explicit ffmpeg output parameters."""
    kwargs: dict[str, Any] = {}
    if output_params:
        kwargs["output_params"] = output_params
    return imageio.get_writer(
        path,
        fps=fps,
        codec="libx264",
        # Always 4:2:0 for broad player support. The lossless preset recovers
        # chroma detail through 2x resolution, not through 4:4:4 output.
        pixelformat="yuv420p",
        macro_block_size=1,
        **kwargs,
    )


def _upscale_frame(img, scale):
    """Lanczos-upscale a PIL image or HxWx3 array for the encode stream.

    Always returns an ndarray. scale <= 1 converts without resizing.
    Only the writer path uses this: saved PNG frames stay at the dreamed native size.
    """
    if scale is None or scale <= 1:
        return img if isinstance(img, np.ndarray) else np.asarray(img)
    if isinstance(img, np.ndarray):
        pil = Image.fromarray(img)
    else:
        pil = img
    w, h = pil.size
    return np.asarray(pil.resize((w * scale, h * scale), Image.Resampling.LANCZOS))


def prepare_video_frames_dir(frames_dir, resume=False, clear=False):
    """Keep a fresh sequence from being mixed with a saved one.

    Atomic-save temps an interrupted write may have left behind are swept
    first: they are never part of a sequence, so they must not block the run
    that would overwrite them anyway. Everything else is the user's, and only
    --clear-frames may remove it -- except a finished temp kept for recovery
    (its ``.complete`` sidecar present), which only manual deletion removes.
    """
    if not frames_dir:
        return
    ensure_output_dir(frames_dir)
    remove_sequence_temps(frames_dir)
    if resume:
        return
    entries = sorted(os.listdir(frames_dir))
    if not entries:
        return
    if clear:
        clear_video_frames(frames_dir, entries)
        return
    raise _non_empty_frames_error(frames_dir, entries)


def _non_empty_frames_error(frames_dir, entries) -> SystemExit:
    return SystemExit(
        f"Error: frames directory '{frames_dir}' is not empty "
        f"({len(entries)} file(s), for example '{entries[0]}').\n"
        "Use an empty folder for a new sequence, pass --resume with the "
        "original settings to continue the saved one, or pass --clear-frames "
        "to replace it."
    )


def check_video_frames_dir_free(frames_dir, resume=False, clear=False) -> None:
    """Read-only preflight: fail before model load if prepare_video_frames_dir would refuse.

    Names the temp sweep would remove do not count, matching prepare_video_frames_dir.
    """
    if not frames_dir or resume or clear or not os.path.isdir(frames_dir):
        return
    entries = sorted(
        name
        for name in os.listdir(frames_dir)
        if not (_is_sweepable_temp(name) and not _is_preserved_output(name, frames_dir))
    )
    if entries:
        raise _non_empty_frames_error(frames_dir, entries)


def clear_video_frames(frames_dir, entries=None) -> None:
    """Delete a saved sequence so a new one can start in the same folder.

    Refuses to touch anything this app did not write, so a mistyped
    --frames-dir cannot cost the user unrelated files. Outputs preserved for
    recovery (temp plus ``.complete`` sidecar) are this app's, but only manual
    deletion may remove them.
    """
    if entries is None:
        entries = sorted(os.listdir(frames_dir))
    foreign = [name for name in entries if not is_sequence_file(name)]
    if foreign:
        raise SystemExit(
            f"Error: --frames-dir '{frames_dir}' contains files that are not part "
            f"of a saved frame sequence (for example '{foreign[0]}'). Use an "
            "empty, dedicated output folder."
        )
    preserved = {name for name in entries if _is_preserved_output(name, frames_dir)}
    for name in entries:
        if name in preserved:
            continue
        try:
            os.remove(os.path.join(frames_dir, name))
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise SystemExit(
                f"Error: could not clear the frame sequence in '{frames_dir}': {exc}"
            ) from exc


# Iteration frames written by single-image --frames/--timelapse runs.
_STALE_FRAME_RE = re.compile(r"^o\d+_i\d+\.png$", re.IGNORECASE)
# Sequential per-frame filenames written by zoom/video generation when a
# frames dir is given. Kept distinct from the single-image `o{oct}_i{iter}`
# timelapse pattern so the two encoders can tell their inputs apart.
VIDEO_FRAME_PATTERN = re.compile(r"^frame_(\d+)\.png$", re.IGNORECASE)
# Atomic-save temps from an interrupted frame or metadata write.
_SEQUENCE_TEMP_RE = re.compile(r"^\.frame_\d+\.tmp\.png$", re.IGNORECASE)
_SEQUENCE_TEMP_NAMES = (VIDEO_FRAMES_META + ".tmp",)
# Atomic-save temps from an iteration-frame write (save_image -> publish_media)
# and their `.complete` sidecars. A temp with its sidecar is a finished output
# a locked destination kept from being published, so it must survive cleanup.
_ITERATION_TEMP_RE = re.compile(r"^o\d+_i\d+\.png\.p\d+\.tmp(?:\.complete)?$", re.IGNORECASE)
_COMPLETE_SUFFIX = ".complete"


def is_sequence_file(name: str) -> bool:
    """True for the files this app writes into a --frames-dir."""
    return (
        VIDEO_FRAME_PATTERN.match(name) is not None
        or _STALE_FRAME_RE.match(name) is not None
        or _ITERATION_TEMP_RE.match(name) is not None
        or name == VIDEO_FRAMES_META
    )


def _is_preserved_output(name: str, frames_dir: str | os.PathLike) -> bool:
    """True for a finished temp awaiting manual recovery, or its live sidecar.

    publish_media writes ``<temp>.complete`` before replacing and leaves both
    behind when the destination stays locked, so the completed output can be
    recovered. Engine sweeps and --clear-frames skip the pair the way GUI
    cleanup does; a sidecar whose temp is already gone is stale, not a
    recovery candidate.
    """
    if name.lower().endswith(_COMPLETE_SUFFIX):
        base = name[: -len(_COMPLETE_SUFFIX)]
        return _ITERATION_TEMP_RE.match(base) is not None and os.path.exists(
            os.path.join(frames_dir, base)
        )
    return _ITERATION_TEMP_RE.match(name) is not None and os.path.exists(
        os.path.join(frames_dir, name + _COMPLETE_SUFFIX)
    )


def _is_sweepable_temp(name: str) -> bool:
    """True for names the temp sweep considers (a preserved pair is still skipped)."""
    return (
        name in _SEQUENCE_TEMP_NAMES
        or _SEQUENCE_TEMP_RE.match(name) is not None
        or _ITERATION_TEMP_RE.match(name) is not None
    )


def remove_sequence_temps(frames_dir: str | os.PathLike | None) -> None:
    """Drop atomic-save temps an interrupted write may have left behind.

    Unmarked temps are partial writes. A temp kept for recovery, with its
    ``.complete`` sidecar, is a finished output instead: it and the sidecar
    are left for manual recovery, and stale sidecars whose temp is gone go.
    """
    if not frames_dir or not os.path.isdir(frames_dir):
        return
    try:
        entries = os.listdir(frames_dir)
    except OSError:
        return
    for name in entries:
        if not _is_sweepable_temp(name):
            continue
        if _is_preserved_output(name, frames_dir):
            if not name.lower().endswith(_COMPLETE_SUFFIX):
                print(
                    f"Note: preserved output from an earlier run kept at "
                    f"'{os.path.join(frames_dir, name)}'; delete it once recovered.",
                    file=sys.stderr,
                    flush=True,
                )
            continue
        try:
            os.remove(os.path.join(frames_dir, name))
        except OSError:
            pass


def clear_stale_frames(frames_dir: str | os.PathLike | None, allow_delete: bool = False) -> None:
    """Clear old iteration frames only when explicitly requested."""
    if not frames_dir or not os.path.isdir(frames_dir):
        return
    remove_sequence_temps(frames_dir)
    try:
        entries = os.listdir(frames_dir)
    except OSError as exc:
        raise SystemExit(f"Error: cannot inspect --frames-dir '{frames_dir}': {exc}") from exc
    foreign = [
        name
        for name in entries
        if not _STALE_FRAME_RE.match(name) and not _is_preserved_output(name, frames_dir)
    ]
    if foreign:
        raise SystemExit(
            f"Error: --frames-dir '{frames_dir}' contains files not recognized as "
            f"iteration frames (for example '{foreign[0]}'). Use an empty, dedicated "
            "output folder."
        )
    stale_frames = [name for name in entries if _STALE_FRAME_RE.match(name)]
    if stale_frames and not allow_delete:
        raise SystemExit(
            f"Error: --frames-dir '{frames_dir}' already contains iteration frames. "
            "Choose an empty folder or pass --clear-frames to replace them."
        )
    for fname in stale_frames:
        try:
            os.remove(os.path.join(frames_dir, fname))
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise SystemExit(
                f"Error: could not clear iteration frames in '{frames_dir}': {exc}"
            ) from exc


def gradient_ascent(
    model: torch.nn.Module,
    model_key: str,
    img: torch.Tensor,
    targets: Collection[str],
    lr: float,
    jitter: int,
    guide_features: GuideFeatures | None = None,
    target_class: int | None = None,
) -> tuple[torch.Tensor, float]:
    """Compatibility adapter around the numerical engine's ascent step."""
    return dream_engine.gradient_ascent(
        model,
        model_key,
        img,
        targets,
        lr,
        jitter,
        guide_features=guide_features,
        target_class=target_class,
        layer_capture=run_to_layers,
        full_capture=forward_with_activations,
    )


def deepdream(
    model: torch.nn.Module | None,
    model_key: str,
    img: torch.Tensor,
    targets: Collection[str],
    iterations: int,
    lr: float,
    octave_scale: float,
    num_octaves: int,
    jitter: int,
    device: Device,
    frames_dir: str | os.PathLike[str] | None = None,
    guide_features: GuideFeatures | None = None,
    target_class: int | None = None,
    preview_file: str | os.PathLike[str] | None = None,
    clear_frames: bool = False,
) -> torch.Tensor:
    """Attach CLI logging, structured progress, and media output to the engine."""

    def start() -> None:
        # The engine invokes this only after all numerical/input validation.
        if frames_dir and iterations > 0:
            clear_stale_frames(frames_dir, allow_delete=clear_frames)
        elif frames_dir:
            print("Note: --iterations is 0, so no frames will be written or changed.", flush=True)

    def note(message: str) -> None:
        print(f"Note: {message}", flush=True)

    def octave(current: int, total: int, current_img: torch.Tensor) -> None:
        print(
            f"Octave {current}/{total}  image {current_img.shape[3]}x{current_img.shape[2]}",
            flush=True,
        )
        emit_event(EV_OCTAVE, current=current, total=total)

    def iteration(
        oi: int, current: int, total: int, current_img: torch.Tensor, loss: float
    ) -> None:
        print(f"  iteration {current}/{total}   loss {loss:9.1f}", flush=True)
        emit_event(EV_ITERATION, current=current, total=total)
        if frames_dir:
            save_image(current_img, os.path.join(frames_dir, f"o{oi}_i{current}.png"))

    def octave_end(current_img: torch.Tensor) -> None:
        if preview_file:
            try:
                if save_preview_frame(tensor_to_np(current_img), preview_file):
                    print(f"[PREVIEW] {preview_file}", flush=True)
                    emit_event(EV_PREVIEW, path=os.fspath(preview_file))
            except (OSError, ValueError):
                pass

    def step(current_img: torch.Tensor) -> tuple[torch.Tensor, float]:
        return gradient_ascent(
            cast(torch.nn.Module, model),
            model_key,
            current_img,
            targets,
            lr,
            jitter,
            guide_features=guide_features,
            target_class=target_class,
        )

    return dream_engine.deepdream(
        model,
        model_key,
        img,
        targets,
        iterations,
        lr,
        octave_scale,
        num_octaves,
        jitter,
        device,
        guide_features=guide_features,
        target_class=target_class,
        callbacks=DreamCallbacks(
            start=start, note=note, octave=octave, iteration=iteration, octave_end=octave_end
        ),
        step=step,
    )


def resolve_video_quality(video_quality="balanced", crf=None, preset=None):
    """Resolve H.264 CRF, speed preset, and encode scale from a quality name.

    Returns (crf, preset, scale). scale comes from the named preset only:
    the lossless preset encodes at 2x resolution; a custom --crf override
    never changes geometry.
    """
    vq_key = (video_quality or "balanced").lower().strip()
    if vq_key in VIDEO_QUALITY_PRESETS:
        target_crf = VIDEO_QUALITY_PRESETS[vq_key]["crf"]
        target_preset = VIDEO_QUALITY_PRESETS[vq_key]["preset"]
        target_scale = VIDEO_QUALITY_PRESETS[vq_key]["scale"]
    else:
        raise ValueError(
            f"Unknown video quality preset '{video_quality}'. "
            f"Valid presets: {sorted(VIDEO_QUALITY_PRESETS)}"
        )

    if crf is not None:
        try:
            target_crf = int(crf)
        except (TypeError, ValueError):
            raise ValueError(f"CRF must be an integer between 0 and 51, got {crf!r}") from None
    if preset is not None:
        target_preset = str(preset)
        if target_preset.lower() not in FFMPEG_PRESETS:
            raise ValueError(
                f"Unknown ffmpeg preset {target_preset!r}. Valid presets: {sorted(FFMPEG_PRESETS)}"
            )
        # ffmpeg presets are lowercase; normalize so a mixed-case caller
        # value ("Medium") can't pass validation yet fail inside the encoder.
        target_preset = target_preset.lower()

    if not (0 <= target_crf <= 51):
        raise ValueError(f"CRF must be between 0 and 51, got {target_crf}")

    return target_crf, target_preset, target_scale


@dataclass
class VideoExportConfig:
    """Video animation and encoding parameters."""

    fps: float = 30.0
    crf: int = 17
    preset: str = "medium"
    preview_file: str | None = None
    frames_dir: str | None = None
    zoom_cx: float = 0.5
    zoom_cy: float = 0.5
    source: str | None = None
    size: int | None = None
    custom_name: str | None = None
    class_name: str | None = None
    guide: str | None = None
    resume: bool = False
    seed: int | None = None
    scale: int = 1
    clear_frames: bool = False


def dream_zoom_animation(
    model,
    model_key,
    img,
    targets,
    num_frames,
    iterations_per_frame,
    lr,
    jitter,
    zoom_scale,
    zoom_rot,
    device,
    out_path,
    fps=30,
    guide_features=None,
    target_class=None,
    preview_file=None,
    crf=17,
    preset="medium",
    zoom_cx=0.5,
    zoom_cy=0.5,
    frames_dir=None,
    source=None,
    size=None,
    custom_name=None,
    class_name=None,
    guide=None,
    resume=False,
    config: VideoExportConfig | None = None,
    seed: int | None = None,
    scale=1,
    clear_frames=False,
):
    if config is not None:
        fps = config.fps
        preview_file = config.preview_file if preview_file is None else preview_file
        crf = config.crf
        preset = config.preset
        scale = config.scale
        zoom_cx = config.zoom_cx
        zoom_cy = config.zoom_cy
        frames_dir = config.frames_dir if frames_dir is None else frames_dir
        source = config.source if source is None else source
        size = config.size if size is None else size
        custom_name = config.custom_name if custom_name is None else custom_name
        class_name = config.class_name if class_name is None else class_name
        guide = config.guide if guide is None else guide
        resume = resume if resume else getattr(config, "resume", False)
        clear_frames = clear_frames or getattr(config, "clear_frames", False)
        seed = config.seed if seed is None else seed

    if seed is not None:
        random.seed(seed)

    if not _VIDEO_AVAILABLE:
        raise SystemExit(
            "Error: 'imageio' and 'imageio-ffmpeg' are required for video export.\n"
            "Please install them via: pip install imageio imageio-ffmpeg"
        )
    # Generate/save split: out_path=None means "frames only" (no .mp4
    # encode); the caller saves the video later via compile_video_frames().
    # frames_dir=None with an out_path preserves the legacy direct-encode.
    encode = out_path is not None
    if not encode and not frames_dir:
        raise ValueError("dream_zoom_animation needs out_path and/or frames_dir")
    if encode:
        ensure_output_dir(os.path.dirname(out_path) or ".")
        _warn_on_overwrite(out_path)
    img = img.to(device)
    # Ensure dimensions are even for H.264 encoder compatibility
    ih, iw = img.shape[2], img.shape[3]
    even_w = iw if (iw % 2 == 0) else iw + 1
    even_h = ih if (ih % 2 == 0) else ih + 1
    if (even_w, even_h) != (iw, ih):
        img = resize(img, (even_h, even_w))
    validate_input_size(img.shape[2], img.shape[3], model_key)
    if iterations_per_frame > 0:
        dream_engine._validate_guide_features(guide_features, targets)

    prepare_video_frames_dir(frames_dir, resume, clear_frames)
    if (
        resume
        and frames_dir
        and any(
            (match := VIDEO_FRAME_PATTERN.match(name)) and int(match.group(1)) > num_frames
            for name in os.listdir(frames_dir)
        )
    ):
        raise SystemExit(
            "Error: the saved sequence has frames beyond --zoom-frames. "
            "Use the original frame count or a new, empty --frames-dir."
        )

    start_frame = 0
    if resume and frames_dir and os.path.isdir(frames_dir):
        start_frame = _count_resumable_frames(frames_dir, limit=num_frames)
        if start_frame > 0:
            _check_resume_compatible(
                frames_dir,
                start_frame,
                {
                    "mode": "zoom",
                    "source": source,
                    "width": even_w,
                    "height": even_h,
                    "model": model_key,
                    "layers": targets,
                    "iterations": iterations_per_frame,
                    "lr": lr,
                    "jitter": jitter,
                    "target_class": target_class,
                    "guide": guide,
                    "zoom_scale": zoom_scale,
                    "zoom_rot": zoom_rot,
                    "zoom_cx": zoom_cx,
                    "zoom_cy": zoom_cy,
                    "seed": seed,
                },
            )
            if seed is not None:
                _advance_jitter_rng(start_frame, iterations_per_frame, jitter)

    if frames_dir:
        ensure_output_dir(frames_dir)
        write_video_frames_meta(
            frames_dir,
            fps=fps,
            mode="zoom",
            source=source,
            width=even_w,
            height=even_h,
            model=model_key,
            layers=targets,
            size=size,
            iterations=iterations_per_frame,
            target_class=target_class,
            class_name=class_name,
            guide=guide,
            custom_name=custom_name,
            zoom_scale=zoom_scale,
            zoom_rot=zoom_rot,
            zoom_cx=zoom_cx,
            zoom_cy=zoom_cy,
            lr=lr,
            jitter=jitter,
            seed=seed,
            total_frames=num_frames,
        )

    if start_frame >= num_frames and not encode:
        print(f"All {num_frames} frames already present in {frames_dir}.", flush=True)
        return frames_dir

    if start_frame > 0:
        last_frame_path = os.path.join(frames_dir, f"frame_{start_frame:06d}.png")
        with Image.open(last_frame_path) as last_frame_pil:
            last_rgb = last_frame_pil.convert("RGB")
        img = pil_to_tensor(last_rgb).to(device)
        img = apply_zoom_and_rotate(img, zoom_scale, zoom_rot, zoom_cx, zoom_cy).to(device)
        print(
            f"Resuming zoom animation from frame {start_frame + 1}/{num_frames} "
            f"({start_frame} existing frames found)...",
            flush=True,
        )

    crf_str = f", crf={crf}" if (encode and crf is not None) else ""
    frames_str = f", frames -> {frames_dir}" if frames_dir else ""
    enc_w, enc_h = (even_w * scale, even_h * scale) if encode and scale > 1 else (even_w, even_h)
    geom = (
        f"{even_w}x{even_h}"
        if (enc_w, enc_h) == (even_w, even_h)
        else f"{even_w}x{even_h} -> {enc_w}x{enc_h}"
    )
    print(
        f"Generating zoom animation: {num_frames} frames @ {fps} fps "
        f"({geom}, zoom_scale={zoom_scale}, rot={zoom_rot}°"
        f", focus=({zoom_cx:.3f}, {zoom_cy:.3f}){crf_str}{frames_str})...",
        flush=True,
    )
    output_params = _video_output_params(crf, preset)

    # Encode to a temp file and rename only on success: an encoder error or a
    # kill mid-encode must not leave a truncated .mp4 at the real output path.
    # Pid-qualified (see _atomic_tmp_path) so concurrent writers can't collide.
    temp_out = _atomic_tmp_path(out_path, ".temp.mp4") if encode else None
    if encode and temp_out:
        _remove_stale_temps(temp_out, out_path + ".temp.mp4")
    keep_temp = False
    writer = None
    try:
        if encode and temp_out:
            writer = _open_video_writer(temp_out, fps, output_params)
            if start_frame > 0 and frames_dir:
                _append_saved_video_frames(writer, frames_dir, start_frame, scale)
        try:
            loss = 0.0
            for f in range(start_frame, num_frames):
                for _ in range(iterations_per_frame):
                    img, loss = gradient_ascent(
                        model,
                        model_key,
                        img,
                        targets,
                        lr,
                        jitter,
                        guide_features=guide_features,
                        target_class=target_class,
                    )
                frame_np = tensor_to_np(img)
                if writer is not None:
                    writer.append_data(_upscale_frame(frame_np, scale))
                if frames_dir:
                    _save_frame_atomic(frame_np, frames_dir, f + 1)
                if (
                    preview_file
                    and (f % 2 == 0 or f == num_frames - 1)
                    and save_preview_frame(frame_np, preview_file)
                ):
                    print(f"[PREVIEW] {preview_file}", flush=True)
                    emit_event(EV_PREVIEW, path=preview_file)
                print(
                    f"  Frame {f + 1}/{num_frames}   loss {loss:9.1f}",
                    flush=True,
                )
                emit_event(EV_FRAME, current=f + 1, total=num_frames)
                if f < num_frames - 1:
                    img = apply_zoom_and_rotate(img, zoom_scale, zoom_rot, zoom_cx, zoom_cy).to(
                        device
                    )
        finally:
            if writer is not None:
                writer.close()
        if frames_dir:
            print(f"Saved {num_frames} frames -> {frames_dir}", flush=True)
        if not encode or not temp_out:
            return frames_dir
        try:
            _publish_media(temp_out, out_path)
        except PermissionError:
            keep_temp = True
            raise
    except BaseException:
        if not keep_temp and temp_out and os.path.exists(temp_out):
            try:
                os.remove(temp_out)
            except OSError:
                pass
        raise
    return out_path


def dream_video_file(
    model,
    model_key,
    video_in_path,
    video_out_path,
    targets,
    iterations_per_frame,
    lr,
    jitter,
    blend_alpha,
    device,
    max_size=512,
    guide_features=None,
    target_class=None,
    preview_file=None,
    crf=17,
    preset="medium",
    frames_dir=None,
    custom_name=None,
    class_name=None,
    guide=None,
    resume=False,
    config: VideoExportConfig | None = None,
    seed: int | None = None,
    scale=1,
    clear_frames=False,
):
    if config is not None:
        preview_file = config.preview_file if preview_file is None else preview_file
        crf = config.crf
        preset = config.preset
        scale = config.scale
        frames_dir = config.frames_dir if frames_dir is None else frames_dir
        custom_name = config.custom_name if custom_name is None else custom_name
        class_name = config.class_name if class_name is None else class_name
        guide = config.guide if guide is None else guide
        resume = resume if resume else getattr(config, "resume", False)
        clear_frames = clear_frames or getattr(config, "clear_frames", False)
        seed = config.seed if seed is None else seed

    if seed is not None:
        random.seed(seed)

    if not _VIDEO_AVAILABLE:
        raise SystemExit(
            "Error: 'imageio' and 'imageio-ffmpeg' are required for video processing.\n"
            "Please install them via: pip install imageio imageio-ffmpeg"
        )
    if not os.path.isfile(video_in_path):
        raise SystemExit(f"Error: Video input not found: {video_in_path}")
    # Generate/save split, mirroring dream_zoom_animation: video_out_path=None
    # means "frames only" (no .mp4 encode); save later via compile_video_frames().
    encode = video_out_path is not None
    if not encode and not frames_dir:
        raise ValueError("dream_video_file needs video_out_path and/or frames_dir")

    if encode:
        ensure_output_dir(os.path.dirname(video_out_path) or ".")
        _warn_on_overwrite(video_out_path)
    # Decode and validate the first frame before any destructive frame-folder
    # preparation. A missing/corrupt/too-small video fails before --clear-frames
    # removes a sequence, but only the first frame is checked: a video that
    # fails later in the run can still leave the old sequence cleared.
    first_frame_size = _first_video_frame_size(video_in_path)
    first_canvas = _video_canvas_size(first_frame_size, max_size)
    validate_input_size(first_canvas[1], first_canvas[0], model_key, what="Video frame")
    if iterations_per_frame > 0:
        dream_engine._validate_guide_features(guide_features, targets)
    prepare_video_frames_dir(frames_dir, resume, clear_frames)

    fps = 30.0
    n_frames = None
    try:
        meta = iio.immeta(video_in_path)
        meta_fps = float(meta.get("fps", 0.0) or 0.0)
        if math.isfinite(meta_fps) and meta_fps > 0:
            fps = meta_fps
        meta_dur = float(meta.get("duration", 0.0) or 0.0)
        if math.isfinite(meta_dur) and meta_dur > 0 and fps > 0:
            n_frames = int(round(meta_dur * fps))
    except (OSError, ValueError, AttributeError, RuntimeError):
        pass
    if not n_frames:
        try:
            props = iio.improps(video_in_path)
            props_fps = float(getattr(props, "fps", 0.0) or 0.0)
            if math.isfinite(props_fps) and props_fps > 0:
                fps = props_fps
            if hasattr(props, "shape") and len(props.shape) >= 4 and props.shape[0] > 0:
                n_frames = int(props.shape[0])
        except (OSError, ValueError, AttributeError, RuntimeError):
            pass

    start_frame = 0
    if resume and frames_dir and os.path.isdir(frames_dir):
        start_frame = _count_resumable_frames(frames_dir)

    crf_str = f", crf={crf}" if (encode and crf is not None) else ""
    dest_str = video_out_path if encode else "(frames only)"
    frames_str = f", frames -> {frames_dir}" if frames_dir else ""
    print(
        f"Processing video: {video_in_path} -> {dest_str} "
        f"(fps={fps:.2f}, blend={blend_alpha:.2f}{crf_str}{frames_str})...",
        flush=True,
    )
    temp_out = _atomic_tmp_path(video_out_path, ".temp.mp4") if encode else None
    remux_out = _atomic_tmp_path(video_out_path, ".remux.temp.mp4") if encode else None
    final_out = video_out_path
    output_params = _video_output_params(crf, preset)

    # Clear leftover temp files from a previous crashed run so they can't be
    # appended to (or fail to overwrite) and corrupt this encode. Legacy
    # fixed-suffix temps from older versions are cleared too.
    if encode and temp_out and remux_out:
        _remove_stale_temps(
            temp_out, remux_out, video_out_path + ".temp.mp4", video_out_path + ".remux.temp.mp4"
        )

    prev_dreamed = None
    if start_frame > 0:
        canvas_w, canvas_h = _video_canvas_size(first_frame_size, max_size)
        _check_resume_compatible(
            frames_dir,
            start_frame,
            {
                "mode": "video",
                "source": video_in_path,
                "width": canvas_w,
                "height": canvas_h,
                "model": model_key,
                "layers": targets,
                "iterations": iterations_per_frame,
                "lr": lr,
                "jitter": jitter,
                "target_class": target_class,
                "guide": guide,
                "video_blend": blend_alpha,
                "seed": seed,
            },
        )
        if seed is not None:
            _advance_jitter_rng(start_frame, iterations_per_frame, jitter)
        last_frame_path = os.path.join(frames_dir, f"frame_{start_frame:06d}.png")
        with Image.open(last_frame_path) as last_frame_pil:
            prev_dreamed = pil_to_tensor(last_frame_pil.convert("RGB")).to(device)
        print(
            f"Resuming video processing from frame {start_frame + 1} "
            f"({start_frame} existing frames found)...",
            flush=True,
        )

    meta_kwargs: dict[str, Any] = {
        "fps": fps,
        "mode": "video",
        "source": video_in_path,
        "model": model_key,
        "layers": targets,
        "size": max_size,
        "iterations": iterations_per_frame,
        "target_class": target_class,
        "class_name": class_name,
        "guide": guide,
        "custom_name": custom_name,
        "video_blend": blend_alpha,
        "lr": lr,
        "jitter": jitter,
        "seed": seed,
    }

    frame_idx = 0
    keep_temp = False
    writer = None
    writer_seeded = start_frame <= 0
    try:
        # Created inside the outer try (like dream_zoom_animation) so a
        # get_writer() failure still runs the temp-file cleanup below
        # instead of leaking a partial *.temp.mp4.
        if encode and temp_out:
            writer = _open_video_writer(temp_out, fps, output_params)
        try:
            loss = 0.0
            target_canvas_size = None
            for raw_frame in iio.imiter(os.path.abspath(video_in_path)):
                frame_idx += 1
                pil_frame = Image.fromarray(raw_frame).convert("RGB")
                if target_canvas_size is None:
                    target_canvas_size = _video_canvas_size(pil_frame.size, max_size)
                    if pil_frame.size != target_canvas_size:
                        pil_frame = pil_frame.resize(
                            target_canvas_size, getattr(Image, "Resampling", Image).LANCZOS
                        )
                    validate_input_size(
                        target_canvas_size[1],
                        target_canvas_size[0],
                        model_key,
                        what="Video frame",
                    )
                    if frames_dir and start_frame == 0:
                        # A fresh sequence records its geometry before the first
                        # frame. A resumed one keeps its existing metadata until
                        # the run completes, so a rejected/short source cannot
                        # overwrite the saved total.
                        write_video_frames_meta(
                            frames_dir,
                            width=target_canvas_size[0],
                            height=target_canvas_size[1],
                            total_frames=int(n_frames) if n_frames else None,
                            **meta_kwargs,
                        )
                elif pil_frame.size != target_canvas_size:
                    pil_frame = pil_frame.resize(
                        target_canvas_size, getattr(Image, "Resampling", Image).LANCZOS
                    )

                if frame_idx <= start_frame:
                    continue

                # Seed the encoder only after the source has proven long enough
                # to contain the saved frames. Otherwise a shortened source
                # would produce a video longer than its input.
                if writer is not None and not writer_seeded and frames_dir:
                    _append_saved_video_frames(writer, frames_dir, start_frame, scale)
                    writer_seeded = True

                frame_t = pil_to_tensor(pil_frame).to(device)

                if prev_dreamed is not None:
                    # blend_alpha is the previous-dream weight (see README):
                    # higher = smoother, more persistent dream memory;
                    # lower = tracks the live frame more closely.
                    cur_img = (1.0 - blend_alpha) * frame_t + blend_alpha * prev_dreamed
                else:
                    cur_img = frame_t

                for _ in range(iterations_per_frame):
                    cur_img, loss = gradient_ascent(
                        model,
                        model_key,
                        cur_img,
                        targets,
                        lr,
                        jitter,
                        guide_features=guide_features,
                        target_class=target_class,
                    )

                prev_dreamed = cur_img.detach()
                frame_np = tensor_to_np(cur_img)
                if writer is not None:
                    writer.append_data(_upscale_frame(frame_np, scale))
                if frames_dir:
                    _save_frame_atomic(frame_np, frames_dir, frame_idx)

                if (
                    preview_file
                    and (frame_idx % 2 == 0)
                    and save_preview_frame(frame_np, preview_file)
                ):
                    print(f"[PREVIEW] {preview_file}", flush=True)
                    emit_event(EV_PREVIEW, path=preview_file)

                total_str = f"/{int(n_frames)}" if n_frames else ""
                print(f"  Frame {frame_idx}{total_str}   loss {loss:9.1f}", flush=True)
                emit_event(EV_FRAME, current=frame_idx, total=int(n_frames) if n_frames else None)
            # The live preview is throttled to every 2nd frame above, so a
            # video with an odd frame count would end showing the penultimate
            # frame. Refresh once more with the final frame (a no-op
            # duplicate when the last frame was already previewed).
            if (
                preview_file
                and prev_dreamed is not None
                and save_preview_frame(tensor_to_np(prev_dreamed), preview_file)
            ):
                print(f"[PREVIEW] {preview_file}", flush=True)
                emit_event(EV_PREVIEW, path=preview_file)
            if start_frame > frame_idx:
                raise SystemExit(
                    f"Error: cannot resume in '{frames_dir}': the saved sequence has "
                    f"{start_frame} frames, but the source video has only {frame_idx}. "
                    "Start a new sequence with --clear-frames (without --resume) or "
                    "use a different --frames-dir."
                )
            if writer is not None and not writer_seeded and frames_dir:
                # The source ended exactly at the saved frame count; encode the
                # complete saved sequence without adding new frames.
                _append_saved_video_frames(writer, frames_dir, start_frame, scale)
                writer_seeded = True
        finally:
            if writer is not None:
                writer.close()

        if frame_idx == 0:
            raise SystemExit(f"Error: No readable frames found in video: {video_in_path}")

        if frames_dir and target_canvas_size is not None:
            write_video_frames_meta(
                frames_dir,
                width=target_canvas_size[0],
                height=target_canvas_size[1],
                total_frames=frame_idx,
                **meta_kwargs,
            )
            print(f"Saved {frame_idx} frames -> {frames_dir}", flush=True)

        if not encode or not temp_out:
            return frames_dir

        # Remux the original audio (atomic; falls back to silent on failure).
        try:
            return _remux_audio(temp_out, video_in_path, final_out)
        except PermissionError:
            # Destination locked: _remux_audio already warned that the completed
            # video is preserved at its temp path, so keep it.
            keep_temp = True
            raise
    except BaseException:
        # Only this run's temp files are removed: a pre-existing output at the
        # real path must survive a failed run.
        if not keep_temp:
            for p in (temp_out, remux_out):
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
        raise


def list_video_frames(frames_dir):
    """Sorted frame PNGs in *frames_dir*; prefers frame_NNNNNN.png, then sorted
    o*_i* or generic PNGs so timelapse folders can be re-exported. GUI preview
    temps are skipped; unreadable folders return [] instead of raising."""
    try:
        names = os.listdir(frames_dir)
    except OSError:
        return []
    numbered = []
    for name in names:
        m = VIDEO_FRAME_PATTERN.match(name)
        if m:
            numbered.append((int(m.group(1)), name))
    if numbered:
        numbered.sort()
        return [os.path.join(frames_dir, name) for _, name in numbered]
    octave_frames = []
    for name in names:
        m = re.match(r"^o(\d+)_i(\d+)\.png$", name, re.IGNORECASE)
        if m:
            octave_frames.append((int(m.group(1)), int(m.group(2)), name))
    if octave_frames:
        octave_frames.sort()
        return [os.path.join(frames_dir, name) for _, _, name in octave_frames]
    generic = [
        n
        for n in names
        if n.lower().endswith(".png")
        and not n.lower().startswith(("_gui_preview", "."))
        and not n.lower().endswith(".tmp")
    ]
    generic.sort(
        key=lambda text: [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", text)]
    )
    return [os.path.join(frames_dir, n) for n in generic]


def _encode_png_sequence(files, out_path, fps=30, crf=17, preset="medium", what="video", scale=1):
    """Encode sorted PNG *files* to an H.264 .mp4 at *out_path* (atomic).

    scale multiplies the encode geometry (lossless preset = 2); source PNGs
    are resized per-frame so only the output MP4 grows.
    """
    ensure_output_dir(os.path.dirname(out_path) or ".")
    with Image.open(files[-1]) as last_img:
        fw, fh = last_img.size
    base_w = max(2, fw - (fw % 2))
    base_h = max(2, fh - (fh % 2))
    scale = scale if scale and scale > 1 else 1
    target_w, target_h = base_w * scale, base_h * scale
    crf_str = f", crf={crf}" if crf is not None else ""
    print(
        f"Compiling {len(files)} frames into {what}: {out_path} "
        f"({target_w}x{target_h} @ {fps} fps{crf_str})...",
        flush=True,
    )
    output_params = _video_output_params(crf, preset)
    temp_out = _atomic_tmp_path(out_path, ".temp.mp4")
    _remove_stale_temps(temp_out, out_path + ".temp.mp4")
    try:
        writer = _open_video_writer(temp_out, fps, output_params)
        total_frames = len(files)
        try:
            for idx, path in enumerate(files, 1):
                with Image.open(path) as frame_img:
                    img = frame_img.convert("RGB")
                if img.size != (base_w, base_h):
                    img = img.resize((base_w, base_h), getattr(Image, "Resampling", Image).LANCZOS)
                writer.append_data(_upscale_frame(np.asarray(img), scale))
                pct = int(idx / total_frames * 100)
                print(f"Encoding frame {idx}/{total_frames} ({pct}%)", flush=True)
                emit_event(EV_FRAME, current=idx, total=total_frames)
        except Exception:
            # Announce before writer.close(), which can wait on ffmpeg. A Stop
            # during that wait must not hide the failure.
            emit_event(EV_ERROR)
            raise
        finally:
            writer.close()
        # Caller atomically moves temp_out to the real path (possibly via an
        # audio remux); a failure there cleans it up.
        return temp_out
    except BaseException:
        if os.path.exists(temp_out):
            try:
                os.remove(temp_out)
            except OSError:
                pass
        raise


def has_audio_stream(path):
    """True when a media file has an audio stream worth remuxing."""
    if not path or not os.path.isfile(path):
        return False
    ext = os.path.splitext(path)[1].lower()
    if ext not in VIDEO_EXTENSIONS:
        # A still image (the usual zoom-animation source) has no audio at all.
        return False
    try:
        meta = iio.immeta(path)
    except (OSError, ValueError, RuntimeError, AttributeError):
        return False
    for key in ("audio", "has_audio"):
        value = meta.get(key)
        if isinstance(value, bool):
            return value
    # Older imageio builds do not report audio; assume a video has one and let
    # the remux fall back to the silent track if the assumption is wrong.
    return True


def _warn_remux_failure(res=None):
    """Report a failed audio remux, including ffmpeg's own last diagnostic line."""
    detail = ""
    stderr = getattr(res, "stderr", None) if res is not None else None
    if stderr:
        lines = [ln.strip() for ln in str(stderr).splitlines() if ln.strip()]
        if lines:
            detail = f" Last ffmpeg message: {lines[-1]}"
    print(
        f"Warning: Audio remux failed; output will have no audio track.{detail}",
        flush=True,
    )


def _remux_audio(silent_mp4, audio_source, final_out):
    """Mux *audio_source*'s audio onto *silent_mp4* to *final_out*, atomically.

    The audio is re-encoded to AAC through a temp file and the final path is
    returned; a failed remux warns and falls back to the silent video."""
    remux_out = _atomic_tmp_path(final_out, ".remux.temp.mp4")
    _remove_stale_temps(remux_out, final_out + ".remux.temp.mp4")
    target = silent_mp4
    try:
        print("Remuxing audio track (AAC)...", flush=True)
        emit_event(EV_STATUS, text="Remuxing audio...")
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        res = subprocess.run(
            [
                ffmpeg_exe,
                "-y",
                "-i",
                os.path.abspath(silent_mp4),
                "-i",
                os.path.abspath(audio_source),
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0?",
                "-shortest",
                os.path.abspath(remux_out),
            ],
            # ffmpeg's stderr is its diagnostics channel; capture it so a
            # failure can say why instead of just "remux failed".
            capture_output=True,
            text=True,
            errors="backslashreplace",
            timeout=FFMPEG_REMUX_TIMEOUT,
            **_NO_WINDOW,
        )
        if res.returncode == 0 and os.path.isfile(remux_out) and os.path.getsize(remux_out) > 0:
            target = remux_out
        else:
            if os.path.exists(remux_out):
                try:
                    os.remove(remux_out)
                except OSError:
                    pass
            _warn_remux_failure(res)
    except (OSError, subprocess.SubprocessError, RuntimeError):
        # RuntimeError: imageio_ffmpeg raises it when it can neither find nor
        # download its bundled ffmpeg, which is the offline/restricted-network
        # case, so it must degrade to a silent track rather than a traceback.
        if os.path.exists(remux_out):
            try:
                os.remove(remux_out)
            except OSError:
                pass
        print("Warning: Audio remux failed; output will have no audio track.", flush=True)
    _publish_media(target, final_out)
    if target is remux_out and os.path.exists(silent_mp4):
        try:
            os.remove(silent_mp4)
        except OSError:
            pass
    return final_out


def compile_video_frames(
    frames_dir, out_path, fps=30, crf=17, preset="medium", audio_from=None, scale=1
):
    """Encode a saved frame sequence (frame_*.png) to .mp4.
    The save half of the generate/save split, so frames export at any quality
    without re-running the dream; audio from *audio_from* or the meta is carried
    over as AAC."""
    if not _VIDEO_AVAILABLE:
        raise SystemExit(
            "Error: 'imageio' and 'imageio-ffmpeg' are required for video export.\n"
            "Please install them via: pip install imageio imageio-ffmpeg"
        )
    if not os.path.isdir(frames_dir):
        raise SystemExit(f"Error: frames directory '{frames_dir}' does not exist.")
    files = list_video_frames(frames_dir)
    if not files:
        raise SystemExit(f"Error: No PNG frames found in '{frames_dir}'.")
    if audio_from is None:
        audio_from = read_video_frames_meta(frames_dir).get("source")
    if not has_audio_stream(audio_from):
        # A zoom animation's source is a still image, so there is nothing to
        # remux; skip ffmpeg entirely rather than run it for no reason.
        audio_from = None
    _warn_on_overwrite(out_path)
    temp_out = _encode_png_sequence(files, out_path, fps=fps, crf=crf, preset=preset, scale=scale)
    keep_temp = False
    try:
        if audio_from and os.path.isfile(audio_from):
            try:
                return _remux_audio(temp_out, audio_from, out_path)
            except PermissionError:
                keep_temp = True
                raise
        try:
            _publish_media(temp_out, out_path)
        except PermissionError:
            keep_temp = True
            raise
        return out_path
    except BaseException:
        if not keep_temp and os.path.exists(temp_out):
            try:
                os.remove(temp_out)
            except OSError:
                pass
        raise


def compile_timelapse(frames_dir, out_path, fps=30, crf=17, preset="medium", scale=1):
    if not _VIDEO_AVAILABLE:
        raise SystemExit("Error: 'imageio' and 'imageio-ffmpeg' are required for timelapse export.")
    if not os.path.isdir(frames_dir):
        raise SystemExit(f"Error: frames directory '{frames_dir}' does not exist.")

    # Parse (octave, iteration) while scanning so a non-frame file can never
    # reach the sort key: the regex here is the only filter, and its capture
    # groups are the sort key.
    parsed = []
    for fname in os.listdir(frames_dir):
        m = re.match(r"^o(\d+)_i(\d+)\.png$", fname, re.IGNORECASE)
        if m:
            parsed.append((int(m.group(1)), int(m.group(2)), fname))
    parsed.sort()
    files = [os.path.join(frames_dir, fname) for _, _, fname in parsed]
    if not files:
        raise SystemExit(f"Error: No PNG frames found in '{frames_dir}'.")

    _warn_on_overwrite(out_path)
    temp_out = _encode_png_sequence(
        files, out_path, fps=fps, crf=crf, preset=preset, scale=scale, what="timelapse"
    )
    try:
        _publish_media(temp_out, out_path, "timelapse")
    except PermissionError:
        # Destination locked: keep the completed temp so the user can recover it.
        raise
    except BaseException:
        if os.path.exists(temp_out):
            try:
                os.remove(temp_out)
            except OSError:
                pass
        raise
    return out_path


def _run_encode_frames_mode(
    args: argparse.Namespace,
    video_crf: int,
    video_preset: str,
    fps_explicit: bool = False,
    video_scale: int = 1,
) -> None:
    """Encode-only mode: --encode-frames DIR --out OUT.mp4|DIR (no dreaming).
    Reads fps/audio defaults from the sequence meta.json unless overridden."""
    frames_dir = args.encode_frames
    if not os.path.isdir(frames_dir):
        raise SystemExit(f"Error: frames directory '{frames_dir}' does not exist.")
    files = list_video_frames(frames_dir)
    if not files:
        raise SystemExit(f"Error: No PNG frames found in '{frames_dir}'.")
    meta = read_video_frames_meta(frames_dir)
    out = args.out
    if os.path.isdir(out) or not out.lower().endswith(".mp4"):
        ensure_output_dir(out)
        candidate = build_systematic_video_name(
            source=meta.get("source") or os.path.basename(os.path.normpath(frames_dir)),
            custom_name=meta.get("custom_name"),
            size=meta.get("size"),
            width=meta.get("width"),
            height=meta.get("height"),
            model=meta.get("model"),
            layers=meta.get("layers"),
            iterations=meta.get("iterations"),
            target_class=meta.get("target_class"),
            class_name=meta.get("class_name"),
            guide=meta.get("guide"),
            mode=meta.get("mode", "zoom"),
            crf=video_crf if video_crf is not None else 17,
        )
        resolved = resolve_name_conflict(out, candidate)
        out = os.path.join(out, resolved)
    audio_from = args.encode_audio_from or meta.get("source")
    if not has_audio_stream(audio_from):
        audio_from = None
    temp_suffix = (
        f".p{os.getpid()}.remux.temp.mp4"
        if (audio_from and os.path.isfile(audio_from))
        else f".p{os.getpid()}.temp.mp4"
    )
    validate_output_path(out, extra_suffix=temp_suffix + ".complete")
    if args.zoom_fps is not None and (fps_explicit or not meta.get("fps")):
        fps = args.zoom_fps
    else:
        try:
            fps = float(meta.get("fps") or 30)
        except (TypeError, ValueError):
            fps = 30
        if not math.isfinite(fps) or fps <= 0:
            fps = 30
    if fps <= 0:
        raise SystemExit(f"Error: --fps must be > 0, got {fps}")
    t0 = time.time()
    ensure_output_dir(os.path.dirname(out) or ".")
    compile_video_frames(
        frames_dir,
        out,
        fps=fps,
        crf=video_crf,
        preset=video_preset,
        scale=video_scale,
        audio_from=audio_from,
    )
    print(f"Done in {time.time() - t0:.1f}s -> {out}", flush=True)
    emit_event(EV_DONE, path=out)


def build_cli_parser() -> argparse.ArgumentParser:
    """Construct the command-line argument parser."""
    ap = argparse.ArgumentParser(description="DeepDream Studio (torchvision)")
    ap.add_argument("--version", action="version", version=f"DeepDream Studio {__version__}")
    ap.add_argument("image", nargs="*", help="input image(s) or video path(s)")
    ap.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=sorted(MODEL_REGISTRY),
        help=f"backbone model (default: {DEFAULT_MODEL})",
    )
    ap.add_argument(
        "--layer",
        default=None,
        help="layer(s), comma separated. Defaults to recommended layer.",
    )
    ap.add_argument(
        "--guide",
        default=None,
        help="guide image path for style/feature-matching dreams",
    )
    ap.add_argument(
        "--class",
        dest="target_class",
        default=None,
        help="target ImageNet class name or index (e.g. 'flamingo', 'pizza', 130)",
    )
    ap.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="gradient ascent steps per octave (default: model-specific, "
        "e.g. 20 for GoogLeNet/VGG16, 25 for VGG19, 30 for InceptionV3)",
    )
    ap.add_argument("--octaves", type=int, default=None, help="number of octaves (0 = auto)")
    ap.add_argument(
        "--octave-scale",
        type=float,
        default=None,
        help="scale factor between octaves (default: model-specific, "
        "e.g. 1.4 for GoogLeNet, 1.35 for InceptionV3, 1.32 for VGG)",
    )
    ap.add_argument(
        "--lr",
        type=float,
        default=None,
        help="gradient ascent step size (default: model-specific, "
        "e.g. 0.03 for GoogLeNet, 0.025 for InceptionV3, 0.018 for VGG)",
    )
    ap.add_argument(
        "--size",
        type=int,
        default=None,
        help="max image dimension (0 = native, default: 512)",
    )
    ap.add_argument(
        "--jitter",
        type=int,
        default=None,
        help="max random shift per iteration (default: 32)",
    )
    ap.add_argument("--out", default="out", help="output directory")
    ap.add_argument("--name", default=None, help="output filename stem")
    ap.add_argument("--frames", action="store_true", help="save every iteration as a frame")
    ap.add_argument(
        "--clear-frames",
        action="store_true",
        help="with --frames/--timelapse, replace existing iteration frames in --frames-dir; "
        "with --zoom/--video, replace a saved frame sequence there. "
        "Files this app did not write are never removed",
    )
    ap.add_argument(
        "--frames-dir",
        default=None,
        help="directory for frames (default: <out>/frames or "
        "<out>/frames_<name> for multi-image runs; used by the GUI to isolate queued jobs). "
        "With --zoom/--video it persists the dreamed video frames as "
        "frame_*.png (+ meta.json) for later --encode-frames exports, and "
        "must be empty unless --resume or --clear-frames is given",
    )
    ap.add_argument(
        "--preview-file",
        default=None,
        help="path to write intermediate preview frames during processing",
    )
    ap.add_argument(
        "--timelapse",
        action="store_true",
        help="compile iteration frames into an MP4 timelapse (auto-enables --frames)",
    )
    ap.add_argument(
        "--timelapse-fps",
        type=int,
        default=30,
        help="timelapse frame rate (default: 30)",
    )
    ap.add_argument(
        "--zoom",
        action="store_true",
        help="generate an infinite zoom / feedback animation video (.mp4)",
    )
    ap.add_argument(
        "--zoom-frames",
        type=int,
        default=60,
        help="number of frames in zoom animation (default: 60)",
    )
    ap.add_argument(
        "--zoom-scale",
        type=float,
        default=1.03,
        help="zoom factor per frame (default: 1.03)",
    )
    ap.add_argument(
        "--zoom-rot",
        type=float,
        default=0.0,
        help="rotation degrees per frame in zoom animation (default: 0.0)",
    )
    ap.add_argument(
        "--zoom-fps",
        "--fps",
        dest="zoom_fps",
        type=float,
        default=None,
        help="frame rate of zoom animation (default: 30; also used when "
        "encoding a saved frame sequence with --encode-frames; fractional "
        "values like 29.97 are accepted)",
    )
    ap.add_argument(
        "--frames-only",
        action="store_true",
        help="with --zoom/--video: dream the frames and save them as PNGs "
        "to --frames-dir without encoding an .mp4, so the video can be "
        "exported later (any number of times, at any quality) via "
        "--encode-frames or the GUI's Save Video button",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="with --zoom/--video and --frames-dir: if existing consecutive "
        "frames (frame_*.png) are present in the frames directory, resume "
        "generation from the last completed frame up to the target number of frames",
    )
    ap.add_argument(
        "--encode-frames",
        default=None,
        metavar="FRAMES_DIR",
        help="encode an existing saved frame sequence (frame_*.png + "
        "meta.json) to --out without dreaming anything; no input image "
        "needed. Repeat with different --video-quality/--crf for multiple "
        "exports of the same sequence",
    )
    ap.add_argument(
        "--encode-audio-from",
        default=None,
        metavar="SOURCE_VIDEO",
        help="with --encode-frames: encode the audio track from SOURCE_VIDEO "
        "as AAC onto the export (defaults to the sequence meta.json source, if any)",
    )
    ap.add_argument(
        "--zoom-x",
        dest="zoom_cx",
        type=float,
        default=0.5,
        help="horizontal zoom focus in normalized 0..1 coords "
        "(0=left edge, 0.5=center, 1=right edge; default: 0.5). "
        "Click the preview image in the GUI to set it.",
    )
    ap.add_argument(
        "--zoom-y",
        dest="zoom_cy",
        type=float,
        default=0.5,
        help="vertical zoom focus in normalized 0..1 coords "
        "(0=top edge, 0.5=center, 1=bottom edge; default: 0.5).",
    )
    ap.add_argument(
        "--video",
        action="store_true",
        help="process an input video with temporal smoothing",
    )
    ap.add_argument(
        "--video-blend",
        type=float,
        default=0.5,
        help="weight of the previous dreamed frame in the temporal blend "
        "(0 = track the current frame, 1 = keep only the previous dream; default: 0.5)",
    )
    ap.add_argument(
        "--video-quality",
        choices=["small", "balanced", "lossless"],
        default="balanced",
        help="video encoding quality preset: small (CRF 28), "
        "balanced (CRF 17, default), lossless (CRF 12, 2x resolution / 4x pixels)",
    )
    ap.add_argument(
        "--crf",
        type=int,
        default=None,
        help="H.264 Constant Rate Factor "
        "(0=minimum compression to 51=maximum compression; overrides --video-quality CRF only — "
        "geometry stays at the preset's scale)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed for jitter (default: none, non-reproducible). "
        "Seeds Python 'random' and torch so the jitter sequence repeats. "
        "In a batch each image is seeded with --seed plus its position, so a "
        "result does not depend on how many other images are in the run. "
        "GPU kernels may still vary slightly between runs.",
    )
    ap.add_argument(
        "--no-cuda",
        "--cpu",
        dest="no_cuda",
        action="store_true",
        help="force CPU (disables NVIDIA CUDA and Apple MPS GPU acceleration)",
    )
    return ap


def apply_seed(seed: int) -> None:
    """Seed every RNG the engine draws from.

    Jitter uses Python's `random`, so torch seeding alone leaves the output
    unreproducible.
    """
    random.seed(seed)
    torch.manual_seed(seed & 0xFFFFFFFF)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed & 0xFFFFFFFF)
    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        try:
            torch.mps.manual_seed(seed & 0xFFFFFFFF)
        except (RuntimeError, AttributeError):
            pass


def _resolve_and_validate_args(
    args: argparse.Namespace,
) -> tuple[str, list[str], int | None, str | None, int, str, int] | None:
    """Validate arguments and resolve defaults. Returns None if handled directly."""
    if args.seed is not None:
        apply_seed(args.seed)

    model_key = args.model
    model_defaults = MODEL_REGISTRY[model_key].get("defaults", {})
    if args.iterations is None:
        args.iterations = model_defaults.get("iterations", 20)
        if args.zoom or args.video:
            args.iterations = max(1, round(args.iterations / 10))
    if args.lr is None:
        args.lr = model_defaults.get("lr", 0.03)
    if args.octave_scale is None:
        args.octave_scale = model_defaults.get("octave_scale", 1.4)
    if args.octaves is None:
        args.octaves = model_defaults.get("octaves", 0)
    if args.jitter is None:
        args.jitter = model_defaults.get("jitter", 32)
    if args.size is None:
        args.size = model_defaults.get("size", 512)
    class_id, class_name = resolve_target_class(args.target_class)

    if args.layer is None:
        if class_id is not None:
            targets = []
        else:
            args.layer = MODEL_REGISTRY[model_key]["recommended"]
            targets = list(
                dict.fromkeys(
                    resolve_layer(n, model_key) for n in args.layer.split(",") if n.strip()
                )
            )
    else:
        targets = list(
            dict.fromkeys(resolve_layer(n, model_key) for n in args.layer.split(",") if n.strip())
        )

    if args.guide and not targets:
        targets = [MODEL_REGISTRY[model_key]["recommended"]]

    if not targets and class_id is None:
        raise SystemExit("Error: No valid target layers or class specified.")

    if not math.isfinite(args.octave_scale):
        raise SystemExit(f"Error: --octave-scale must be finite, got {args.octave_scale}")
    if args.octave_scale <= 1.0:
        raise SystemExit(f"Error: --octave-scale must be greater than 1.0, got {args.octave_scale}")
    if args.octaves < 0:
        raise SystemExit(f"Error: --octaves must be >= 0, got {args.octaves}")
    if args.iterations < 0:
        raise SystemExit(f"Error: --iterations must be >= 0, got {args.iterations}")
    if not math.isfinite(args.lr):
        raise SystemExit(f"Error: --lr must be finite, got {args.lr}")
    if args.lr <= 0:
        raise SystemExit(f"Error: --lr must be greater than 0, got {args.lr}")
    if args.jitter < 0:
        raise SystemExit(f"Error: --jitter must be >= 0, got {args.jitter}")
    if args.size < 0:
        raise SystemExit(f"Error: --size must be >= 0, got {args.size}")
    if args.encode_frames:
        for _flag in ("zoom", "video", "timelapse", "frames", "frames_only"):
            if getattr(args, _flag):
                raise SystemExit(
                    f"Error: --encode-frames cannot be combined with --{_flag.replace('_', '-')}. "
                    "It only encodes an existing frame sequence."
                )
        if args.image:
            print(
                f"Note: ignoring {len(args.image)} input image(s); "
                "--encode-frames only encodes the saved sequence."
            )
        if args.encode_audio_from and not os.path.isfile(args.encode_audio_from):
            raise SystemExit(f"Error: --encode-audio-from file not found: {args.encode_audio_from}")
    elif not args.image:
        raise SystemExit("Error: the following arguments are required: image")
    if args.frames_only and not (args.zoom or args.video):
        raise SystemExit("Error: --frames-only requires --zoom or --video.")
    if args.clear_frames and not (args.frames or args.timelapse or args.zoom or args.video):
        raise SystemExit(
            "Error: --clear-frames requires --frames, --timelapse, --zoom, or --video."
        )
    if args.clear_frames and args.resume:
        raise SystemExit(
            "Error: --clear-frames cannot be combined with --resume; "
            "--clear-frames starts a new sequence, --resume continues the saved one."
        )
    if (
        args.clear_frames
        and (args.zoom or args.video)
        and not (args.frames_dir or args.frames_only)
    ):
        raise SystemExit(
            "Error: --clear-frames with --zoom or --video needs a frame folder to clear; "
            "add --frames-dir or --frames-only."
        )
    if args.encode_audio_from and not args.encode_frames:
        raise SystemExit("Error: --encode-audio-from requires --encode-frames.")
    fps_explicit = args.zoom_fps is not None
    if args.zoom_fps is None:
        args.zoom_fps = 30
    if not math.isfinite(args.zoom_fps):
        raise SystemExit(f"Error: --zoom-fps must be finite, got {args.zoom_fps}")

    if args.name is not None:
        validate_filename_stem(args.name)
    if not args.encode_frames and len(args.image) == 1 and not os.path.isfile(args.image[0]):
        raise SystemExit(f"Error: Input image not found: {args.image[0]}")

    if args.zoom and args.video:
        raise SystemExit("Error: --zoom and --video cannot be used together.")
    if args.zoom:
        for img_path in args.image:
            if os.path.splitext(img_path)[1].lower() in VIDEO_EXTENSIONS:
                raise SystemExit(
                    f"Error: --zoom requires an image input, got video file '{img_path}'. "
                    "Use --video to process video files."
                )
        if args.zoom_frames <= 0:
            raise SystemExit(f"Error: --zoom-frames must be > 0, got {args.zoom_frames}")
        if args.zoom_fps <= 0:
            raise SystemExit(f"Error: --zoom-fps must be > 0, got {args.zoom_fps}")
        if not math.isfinite(args.zoom_scale) or args.zoom_scale < 1.0:
            raise SystemExit(f"Error: --zoom-scale must be >= 1.0, got {args.zoom_scale}")
        if not math.isfinite(args.zoom_rot):
            raise SystemExit(f"Error: --zoom-rot must be finite, got {args.zoom_rot}")
        for _name, _v in (("--zoom-x", args.zoom_cx), ("--zoom-y", args.zoom_cy)):
            if not math.isfinite(_v) or not (0.0 <= _v <= 1.0):
                raise SystemExit(f"Error: {_name} must be between 0.0 and 1.0, got {_v}")
    if args.video and (not math.isfinite(args.video_blend) or not (0.0 <= args.video_blend <= 1.0)):
        raise SystemExit(
            f"Error: --video-blend must be between 0.0 and 1.0, got {args.video_blend}"
        )
    if args.timelapse and (args.zoom or args.video):
        raise SystemExit("Error: --timelapse only applies to single-image mode.")
    if args.frames and (args.zoom or args.video):
        raise SystemExit("Error: --frames only applies to single-image mode.")
    if args.timelapse and args.iterations == 0:
        raise SystemExit("Error: --timelapse requires --iterations > 0.")
    if args.timelapse and args.timelapse_fps <= 0:
        raise SystemExit(f"Error: --timelapse-fps must be > 0, got {args.timelapse_fps}")

    if args.crf is not None and not (0 <= args.crf <= 51):
        raise SystemExit(f"Error: --crf must be between 0 and 51, got {args.crf}")

    try:
        video_crf, video_preset, video_scale = resolve_video_quality(
            video_quality=args.video_quality, crf=args.crf
        )
    except ValueError as exc:
        raise SystemExit(f"Error: {exc}") from None

    if args.encode_frames:
        _run_encode_frames_mode(
            args,
            video_crf,
            video_preset,
            fps_explicit=fps_explicit,
            video_scale=video_scale,
        )
        return None

    if args.timelapse:
        args.frames = True

    if (
        args.frames_dir
        and not args.frames
        and not args.frames_only
        and not args.zoom
        and not args.video
    ):
        print("Note: --frames-dir given without --frames/--timelapse, so it will be ignored.")

    ensure_output_dir(args.out)
    return model_key, targets, class_id, class_name, video_crf, video_preset, video_scale


def _build_image_plan(
    args: argparse.Namespace,
    model_key: str,
    targets: list[str],
    class_id: int | None,
    class_name: str | None,
) -> tuple[list[dict[str, Any]], int]:
    """Validate and build output plan for all input images."""
    total_images = len(args.image)
    image_plan: list[dict[str, Any]] = []
    plan_errors = 0
    for img_idx, current_image in enumerate(args.image):
        if total_images > 1 and not os.path.isfile(current_image):
            print(f"Error: Input image not found: {current_image}", file=sys.stderr)
            plan_errors += 1
            continue
        if args.video:
            img = None
            was_resized = False
            target_h = target_w = 0
        else:
            try:
                if total_images == 1:
                    img, src_size = load_image_with_size(current_image, args.size)
                else:
                    img = None
                    src_size = load_image_native_size(current_image)
            except (OSError, ValueError, Image.DecompressionBombError) as exc:
                if total_images > 1:
                    print(f"Error: Could not load image '{current_image}': {exc}", file=sys.stderr)
                    plan_errors += 1
                    continue
                raise SystemExit(f"Error: Could not load image '{current_image}': {exc}") from exc

            was_resized = resize_dims(src_size, args.size) is not None
            native_too_small = min(src_size) < MODEL_REGISTRY[model_key]["min_size"]
            size_hint = (
                "Use --size 0 (native resolution) or a larger --size."
                if was_resized and not native_too_small
                else "The image is too small even at native resolution."
            )
            if img is not None:
                target_h, target_w = img.shape[2], img.shape[3]
            else:
                target_w, target_h = resize_dims(src_size, args.size) or src_size
            try:
                validate_input_size(target_h, target_w, model_key, hint=size_hint)
            except SystemExit as exc:
                if total_images > 1:
                    if isinstance(exc.code, str) and exc.code.strip():
                        print(exc.code, file=sys.stderr)
                    plan_errors += 1
                    continue
                raise

        base_stem = os.path.splitext(os.path.basename(current_image))[0]
        if args.name is None:
            size_suffix = f"_{args.size}px" if was_resized else ""
            name = f"{base_stem}{size_suffix}"
        else:
            name = f"{args.name}_{img_idx + 1}" if total_images > 1 else args.name

        target_parts = [t.replace(".", "-") for t in targets]
        if class_id is not None:
            cname = class_name or str(class_id)
            class_slug = re.sub(r"[^a-zA-Z0-9]+", "-", cname).strip("-") or str(class_id)
            target_parts.append(f"class-{class_slug}")
        if args.guide:
            guide_stem = os.path.splitext(os.path.basename(args.guide))[0]
            target_parts.append(f"guide-{guide_stem}")
        target_str = "_".join(target_parts)

        suffix = f"_{target_str}" if target_str else ""
        if args.zoom:
            suffix += "_zoom"
        elif args.video:
            suffix += "_dreamed"

        validate_filename_stem(f"{name}{suffix}", what="output filename", max_len=None)

        ext = ".mp4" if (args.zoom or args.video) else ".png"
        out_path = os.path.join(args.out, f"{name}{suffix}{ext}")
        out_key = os.path.normcase(os.path.abspath(out_path))
        planned_keys = {os.path.normcase(os.path.abspath(item["out_path"])) for item in image_plan}
        if total_images > 1 and out_key in planned_keys:
            base_name = name
            counter = img_idx + 1
            while out_key in planned_keys:
                name = f"{base_name}_{counter}"
                out_path = os.path.join(args.out, f"{name}{suffix}{ext}")
                out_key = os.path.normcase(os.path.abspath(out_path))
                counter += 1
        temp_suffix = (
            f".p{os.getpid()}.remux.temp.mp4"
            if args.video
            else (f".p{os.getpid()}.temp.mp4" if args.zoom else f".p{os.getpid()}.tmp")
        )
        validate_output_path(out_path, extra_suffix=temp_suffix + ".complete")
        if args.timelapse:
            tl_path = os.path.join(args.out, f"{name}{suffix}_timelapse.mp4")
            validate_output_path(tl_path, extra_suffix=f".p{os.getpid()}.temp.mp4.complete")

        if args.frames:
            if args.frames_dir:
                if total_images > 1:
                    raise SystemExit(
                        "Error: --frames-dir cannot be combined with multiple input images."
                    )
                frames_dir = args.frames_dir
            else:
                frames_dir = (
                    os.path.join(args.out, f"frames_{name}")
                    if total_images > 1
                    else os.path.join(args.out, "frames")
                )
        elif (args.zoom or args.video) and (args.frames_dir or args.frames_only):
            if args.frames_dir:
                if total_images > 1:
                    raise SystemExit(
                        "Error: --frames-dir cannot be combined with multiple input images."
                    )
                frames_dir = args.frames_dir
            else:
                frames_dir = os.path.join(args.out, f"frames_{name}")
        else:
            frames_dir = None
        if frames_dir and (args.zoom or args.video):
            check_video_frames_dir_free(frames_dir, bool(args.resume), bool(args.clear_frames))
        if frames_dir and args.iterations > 0:
            if args.zoom or args.video:
                max_frame = args.zoom_frames if args.zoom else 999999
                # save_frame_atomic writes ".frame_NNNNNN.tmp.png" via os.replace.
                validate_output_path(os.path.join(frames_dir, f".frame_{max_frame:06d}.tmp.png"))
            else:
                model_spec = MODEL_REGISTRY.get(model_key)
                model_min = model_spec["min_size"] if model_spec is not None else MIN_OCTAVE_SIZE
                n_octaves = plan_octaves(
                    target_h,
                    target_w,
                    args.octaves,
                    args.octave_scale,
                    min_octave_size=max(MIN_OCTAVE_SIZE, model_min),
                    target_base_size=max(TARGET_BASE_SIZE, model_min),
                )
                # save_image -> publish_media writes the frame temp and then its
                # ".complete" sidecar; the longest path is the sidecar.
                validate_output_path(
                    os.path.join(frames_dir, f"o{n_octaves}_i{args.iterations}.png"),
                    extra_suffix=f".p{os.getpid()}.tmp.complete",
                )

        image_plan.append(
            {
                "current_image": current_image,
                "img": img if total_images == 1 else None,
                "name": name,
                "suffix": suffix,
                "out_path": out_path,
                "frames_dir": frames_dir,
            }
        )
        if total_images > 1:
            img = None

    return image_plan, plan_errors


def _is_oom_error(exc: BaseException) -> bool:
    """True for a CUDA/MPS out-of-memory failure (typed or message-only)."""
    oom_type = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)
    if isinstance(oom_type, type) and isinstance(exc, oom_type):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _release_device_memory(device: str) -> None:
    """Return cached allocator blocks to the device so the next job starts clean.

    Best-effort by design: this runs in cleanup and fallback paths, so a backend
    that refuses the call (torch.mps.empty_cache raises when the MPS backend is
    not actually usable) must not replace the real error with its own.
    """
    try:
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except (RuntimeError, OSError, AttributeError):
        pass


def _oom_message(device: str, model_key: str, working_size: int | None) -> str:
    """Actionable advice for an out-of-memory failure."""
    lines = [f"Error: the {device.upper()} ran out of memory."]
    if working_size and working_size > 0:
        floor = MODEL_MIN_SIZE.get(model_key, 0)
        suggested = max(floor, int(working_size * 0.75) // 32 * 32)
        if suggested < working_size:
            lines.append(
                f"  - Lower the resolution: this run used {working_size}px, try --size {suggested}."
            )
    else:
        lines.append("  - Lower the resolution with --size (e.g. --size 512).")
    lines.append("  - Use fewer target layers, or a lighter model (googlenet).")
    lines.append("  - Close other GPU-heavy apps, or run on the CPU with --cpu (slow).")
    return "\n".join(lines)


def _working_size(args: argparse.Namespace, img: Any) -> int | None:
    shape = getattr(img, "shape", None)
    if shape is not None and len(shape) >= 4:
        return int(max(shape[2], shape[3]))
    size = getattr(args, "size", None)
    return int(size) if size else None


def _device_desc(device: str) -> str:
    """Human-readable device line shown at startup; never raises."""
    if device == "cuda":
        try:
            return f"Device: cuda  ({torch.cuda.get_device_name(0)})"
        except Exception:
            return "Device: cuda"
    if device == "mps":
        return "Device: mps  (Apple Silicon GPU)"
    return "Device: cpu"


def _report_model_load_failure(exc: BaseException, device: str, model_desc: str) -> NoReturn:
    """Explain a failed model load and exit; never returns."""
    if _is_oom_error(exc):
        print(
            f"Error: not enough {device.upper()} memory to load {model_desc}: {exc}\n"
            "Close other GPU-heavy apps, pick a lighter model (googlenet), "
            "or run on the CPU with --cpu.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(
        f"Error: could not download or load the model weights: {exc}\n"
        f"Check your internet connection and try again.",
        file=sys.stderr,
    )
    sys.exit(1)


def _init_device_and_model(
    args: argparse.Namespace,
    model_key: str,
    targets: list[str],
) -> tuple[str, Any, Any]:
    """Detect acceleration hardware, load weights, and extract style guide features."""
    if args.no_cuda:
        device = "cpu"
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(_device_desc(device), flush=True)
    model_desc = MODEL_REGISTRY[model_key]["description"]
    print(f"Loading {model_desc}...", flush=True)
    try:
        model = build_model(model_key, device)
    except (OSError, ValueError, RuntimeError) as exc:
        if device == "cpu":
            _report_model_load_failure(exc, device, model_desc)
            raise SystemExit(1) from exc  # Unreachable: _report exits; guards against mocks.
        # A detected accelerator is not always usable: a driver/runtime mismatch
        # or an unsupported GPU architecture reports is_available() == True and
        # then fails when the weights are moved onto it. The CPU always works, so
        # retry there rather than stranding a first-time user at an error that
        # only mentions a command-line flag they did not ask for.
        print(
            f"Note: {_device_desc(device)} could not be used ({exc}).\n"
            "Falling back to the CPU - it is slower, but it always works.",
            file=sys.stderr,
        )
        _release_device_memory(device)
        device = "cpu"
        print(_device_desc(device), flush=True)
        print(f"Loading {model_desc}...", flush=True)
        try:
            model = build_model(model_key, device)
        except (OSError, ValueError, RuntimeError) as cpu_exc:
            _report_model_load_failure(cpu_exc, device, model_desc)
            raise SystemExit(1) from cpu_exc  # Unreachable: _report exits; guards against mocks.

    guide_features = None
    if args.guide:
        print(f"Extracting guide features from {args.guide}...", flush=True)
        try:
            guide_features = extract_guide_features(
                model,
                model_key,
                args.guide,
                targets,
                device,
                max_size=(args.size if (args.size and args.size > 0) else 512),
            )
        except Exception as exc:
            if _is_oom_error(exc):
                print(_oom_message(device, model_key, args.size), file=sys.stderr)
            else:
                print(f"Error processing guide image: {exc}", file=sys.stderr)
            sys.exit(1)
        if not guide_features:
            raise SystemExit(
                f"Error: No guide features extracted from '{args.guide}' for "
                f"target layer(s) {targets}. The dream would run with no guide "
                "effect; refusing to continue."
            )

    return device, model, guide_features


def _execute_image_plan(
    args: argparse.Namespace,
    model_key: str,
    targets: list[str],
    class_id: int | None,
    class_name: str | None,
    video_crf: int,
    video_preset: str,
    device: str,
    model: Any,
    guide_features: Any,
    image_plan: list[dict[str, Any]],
    plan_errors: int,
    t_all_start: float,
    video_scale: int,
) -> None:
    """Execute dreaming over the planned batch sequence."""
    total_images = len(args.image)
    batch_errors = 0
    result = None

    for img_idx, plan_item in enumerate(image_plan):
        current_image = plan_item["current_image"]
        img = plan_item["img"]
        name = plan_item["name"]
        suffix = plan_item["suffix"]
        out_path = plan_item["out_path"]
        frames_dir = plan_item["frames_dir"]

        # Reseed per image so a batch is reproducible: each result depends
        # only on --seed and its position in the batch, not on RNG state left
        # behind by earlier images (jitter draws from a single global RNG).
        image_seed = args.seed
        if args.seed is not None and total_images > 1:
            image_seed = args.seed + img_idx
            apply_seed(image_seed)

        if total_images > 1:
            print(
                f"\n{'=' * 60}\n"
                f" [{img_idx + 1}/{total_images}] Processing: {current_image}\n"
                f"{'=' * 60}",
                flush=True,
            )

        if frames_dir and args.iterations > 0:
            ensure_output_dir(frames_dir)

        t0 = time.time()
        try:
            if img is None and not args.video:
                img = load_image(current_image, args.size)

            video_config = VideoExportConfig(
                fps=args.zoom_fps,
                preview_file=args.preview_file,
                crf=video_crf,
                preset=video_preset,
                scale=video_scale,
                zoom_cx=args.zoom_cx,
                zoom_cy=args.zoom_cy,
                frames_dir=frames_dir,
                source=current_image,
                size=args.size,
                custom_name=args.name,
                class_name=class_name,
                guide=args.guide,
                resume=bool(args.resume),
                clear_frames=bool(args.clear_frames),
                seed=image_seed,
            )

            if args.zoom:
                dream_zoom_animation(
                    model=model,
                    model_key=model_key,
                    img=img,
                    targets=targets,
                    num_frames=args.zoom_frames,
                    iterations_per_frame=args.iterations,
                    lr=args.lr,
                    jitter=args.jitter,
                    zoom_scale=args.zoom_scale,
                    zoom_rot=args.zoom_rot,
                    device=device,
                    out_path=None if args.frames_only else out_path,
                    guide_features=guide_features,
                    target_class=class_id,
                    config=video_config,
                )
            elif args.video:
                dream_video_file(
                    model=model,
                    model_key=model_key,
                    video_in_path=current_image,
                    video_out_path=None if args.frames_only else out_path,
                    targets=targets,
                    iterations_per_frame=args.iterations,
                    lr=args.lr,
                    jitter=args.jitter,
                    blend_alpha=args.video_blend,
                    device=device,
                    max_size=args.size,
                    guide_features=guide_features,
                    target_class=class_id,
                    config=video_config,
                )
            else:
                print(
                    f"Dreaming... target(s): {targets or [class_name]}  "
                    f"octaves: {'auto' if args.octaves == 0 else args.octaves}  "
                    f"iterations/octave: {args.iterations}",
                    flush=True,
                )
                try:
                    result = deepdream(
                        model,
                        model_key,
                        img,
                        targets,
                        args.iterations,
                        args.lr,
                        args.octave_scale,
                        args.octaves,
                        args.jitter,
                        device,
                        frames_dir,
                        guide_features=guide_features,
                        target_class=class_id,
                        preview_file=args.preview_file,
                        clear_frames=args.clear_frames,
                    )
                except OSError as exc:
                    print(f"Error: could not save frames: {exc}", file=sys.stderr)
                    if total_images > 1:
                        batch_errors += 1
                        continue
                    sys.exit(1)

                try:
                    save_image(result, out_path, warn_on_overwrite=True)
                except OSError as exc:
                    print(
                        f"Error: could not save output to '{out_path}': {exc}",
                        file=sys.stderr,
                    )
                    if total_images > 1:
                        batch_errors += 1
                        continue
                    sys.exit(1)

                if args.timelapse and frames_dir:
                    tl_path = os.path.join(args.out, f"{name}{suffix}_timelapse.mp4")
                    try:
                        compile_timelapse(
                            frames_dir,
                            tl_path,
                            fps=args.timelapse_fps,
                            crf=video_crf,
                            preset=video_preset,
                            scale=video_scale,
                        )
                        print(f"Timelapse video -> {tl_path}", flush=True)
                    except (OSError, RuntimeError, ValueError) as exc:
                        print(
                            f"Error: could not compile timelapse video: {exc}",
                            file=sys.stderr,
                        )
                        if total_images > 1:
                            batch_errors += 1
                            continue
                        sys.exit(1)

        except SystemExit as exc:
            if total_images > 1:
                if isinstance(exc.code, str) and exc.code.strip():
                    print(exc.code, file=sys.stderr)
                batch_errors += 1
                continue
            raise
        except (RuntimeError, ValueError, OSError) as exc:
            if _is_oom_error(exc):
                print(
                    _oom_message(device, model_key, _working_size(args, img)),
                    file=sys.stderr,
                )
            else:
                print(f"Error: {exc}", file=sys.stderr)
            if total_images > 1:
                batch_errors += 1
                continue
            sys.exit(1)
        finally:
            if total_images > 1:
                plan_item["img"] = None
                img = None
                result = None
                _release_device_memory(device)

        if args.frames_only and (args.zoom or args.video):
            print(f"Done in {time.time() - t0:.1f}s -> {frames_dir}", flush=True)
            emit_event(EV_DONE, path=frames_dir)
        else:
            print(f"Done in {time.time() - t0:.1f}s -> {out_path}", flush=True)
            emit_event(EV_DONE, path=out_path)
        if total_images == 1:
            img = None
            result = None
            _release_device_memory(device)

    if total_images > 1:
        queue_errors = plan_errors + batch_errors
        if queue_errors > 0:
            print(
                f"\nQueue completed with {queue_errors} error(s) out of {total_images} generations "
                f"in {time.time() - t_all_start:.1f}s",
                file=sys.stderr,
            )
            sys.exit(1)
        else:
            print(
                f"\nQueue completed: {total_images} generations in {time.time() - t_all_start:.1f}s",
                flush=True,
            )


def _run_encode_frames_cli(args: argparse.Namespace) -> None:
    """Run --encode-frames, announcing a failure on the event stream before it propagates."""
    try:
        _resolve_and_validate_args(args)
    except (SystemExit, Exception):
        emit_event(EV_ERROR)
        raise


def main(argv: list[str] | None = None) -> None:
    """Entry point for the DeepDream CLI."""
    _configure_streams()
    check_runtime_environment()
    ap = build_cli_parser()
    args = ap.parse_args(argv)
    # Temp files are named with this pid. The GUI's handle to the process can be
    # a venv launcher with another pid, so the GUI finds them through this one.
    emit_event(EV_PID, pid=os.getpid())
    if args.encode_frames:
        _run_encode_frames_cli(args)
        return

    resolved = _resolve_and_validate_args(args)
    if resolved is None:
        return
    model_key, targets, class_id, class_name, video_crf, video_preset, video_scale = resolved

    t_plan_start = time.time()
    image_plan, plan_errors = _build_image_plan(args, model_key, targets, class_id, class_name)
    total_images = len(args.image)
    if total_images > 1 and not image_plan:
        print(
            f"\nQueue completed with {plan_errors} error(s) out of {total_images} generations "
            f"in {time.time() - t_plan_start:.1f}s",
            file=sys.stderr,
        )
        sys.exit(1)

    device, model, guide_features = _init_device_and_model(args, model_key, targets)

    if class_id is not None:
        print(f"Target class: #{class_id} ({class_name})", flush=True)

    _execute_image_plan(
        args=args,
        model_key=model_key,
        targets=targets,
        class_id=class_id,
        class_name=class_name,
        video_crf=video_crf,
        video_preset=video_preset,
        video_scale=video_scale,
        device=device,
        model=model,
        guide_features=guide_features,
        image_plan=image_plan,
        plan_errors=plan_errors,
        t_all_start=time.time(),
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Interrupted by user]", file=sys.stderr)
        sys.exit(130)
