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
from dataclasses import dataclass
from typing import Any, NoReturn

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image, ImageOps

from . import __version__
from .model_data import (
    MAX_NAME_LEN,
    MIN_TORCH_VERSION,
    MODEL_DEFAULTS,
    MODEL_DESCRIPTIONS,
    MODEL_LAYERS,
    MODEL_MIN_SIZE,
    MODEL_ORDER,
    MODEL_RECOMMENDED,
    VIDEO_EXTENSIONS,
    VIDEO_QUALITY_PRESETS,
    build_systematic_video_name,
    filename_stem_error,
    read_video_frames_meta,
    resolve_name_conflict,
    resume_mismatches,
    write_video_frames_meta,
)
from .protocol import (
    EV_DONE,
    EV_FRAME,
    EV_ITERATION,
    EV_OCTAVE,
    EV_PREVIEW,
    EV_STATUS,
    format_event,
)


def emit_event(event: str, **fields: Any) -> None:
    """Emit a structured progress event for the GUI, alongside normal output.

    Also printed as text so a human reading the console loses nothing.
    """
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

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def _freeze(m):
    """Put a model in inference mode with gradients off, as dreaming requires."""
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def build_googlenet(device, weights=True):
    """GoogLeNet on *device*.

    weights=False skips the download and uses random initialization: tests need
    the layer structure and forward/backward behaviour, not trained features,
    and should not require network access to run.
    """
    if weights:
        m = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1).to(device)
        m.aux_logits = False
    else:
        m = models.googlenet(weights=None, aux_logits=True, init_weights=True).to(device)
        m.aux_logits = False
    return _freeze(m)


def _forward_trunk(model, x, target_set, is_aux):
    """Forward through a model's direct children, capturing target activations.
    Stops after the deepest target; _transform_input() is re-applied first because
    it is a tensor op, not a child module, so walking children would skip it."""
    if getattr(model, "transform_input", False) and hasattr(model, "_transform_input"):
        x = model._transform_input(x)
    children = list(model.named_children())
    names = [n for n, _ in children]
    missing = target_set - set(names)
    if missing:
        raise RuntimeError(f"Target layer(s) not found in model: {sorted(missing)}")
    aux_targets = sorted(n for n in target_set if is_aux(n))
    if aux_targets:
        raise RuntimeError(
            f"Target layer(s) are auxiliary heads, not dreamable trunk layers: {aux_targets}"
        )
    deepest = max(i for i, n in enumerate(names) if n in target_set)
    stop_at = names[deepest + 1] if deepest + 1 < len(names) else None
    acts = {}
    for name, module in children:
        if name == stop_at:
            break
        # Auxiliary heads are side branches; skip them to stay on the trunk.
        if is_aux(name):
            continue
        x = module(x)
        if name in target_set:
            acts[name] = x
            if len(acts) == len(target_set):
                break
    missing = target_set - set(acts.keys())
    if missing:
        raise RuntimeError(f"Target layer(s) not found in model: {sorted(missing)}")
    return acts


def _forward_googlenet(model, x, target_set):
    """GoogLeNet trunk walk; its auxiliary heads are named aux1/aux2."""
    return _forward_trunk(model, x, target_set, lambda n: n.startswith("aux"))


def _build_inceptionv3(device):
    m = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1).to(device)
    # Auxiliary classifier output is not used for DeepDream feature extraction.
    m.aux_logits = False
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def _forward_inceptionv3(model, x, target_set):
    """InceptionV3 trunk walk; its auxiliary branch is named AuxLogits."""
    return _forward_trunk(model, x, target_set, lambda n: n == "AuxLogits")


def _build_vgg(variant, device):
    """Build a VGG16 or VGG19 model with pretrained ImageNet weights."""
    if variant == "vgg16":
        m = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).to(device)
    else:
        m = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).to(device)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def _forward_vgg(model, x, target_set):
    """Run VGG's features block, stopping at the deepest target.
    Targets are "features.N" indices into model.features; avgpool and classifier
    are never run."""
    idx_to_name = {}
    for name in target_set:
        if not name.startswith("features."):
            raise RuntimeError(
                f"VGG target '{name}' must be in the form 'features.N' (e.g. 'features.28')."
            )
        parts = name.split(".")
        if len(parts) != 2:
            raise RuntimeError(f"Cannot parse layer index from '{name}'.") from None
        try:
            idx = int(parts[1])
        except ValueError:
            raise RuntimeError(f"Cannot parse layer index from '{name}'.") from None
        idx_to_name[idx] = name

    # Reject out-of-range indices before running the forward pass.
    try:
        num_features = len(model.features)
    except TypeError:
        num_features = None
    if num_features is not None:
        bad = sorted(name for idx, name in idx_to_name.items() if idx < 0 or idx >= num_features)
        if bad:
            raise RuntimeError(f"Target layer(s) not found in model.features: {bad}")

    deepest_idx = max(idx_to_name)
    acts = {}
    for i, layer in enumerate(model.features):
        x = layer(x)
        if i in idx_to_name:
            acts[idx_to_name[i]] = x
        if i >= deepest_idx:
            break
    missing = target_set - set(acts.keys())
    if missing:
        raise RuntimeError(f"Target layer(s) not found in model.features: {sorted(missing)}")
    return acts


# Layer labels, defaults, and quality presets live in model_data.py; only the
# torch-dependent pieces (build and truncated forward) are registered here.
_MODEL_IMPLS = {
    "googlenet": (build_googlenet, _forward_googlenet),
    "inceptionv3": (_build_inceptionv3, _forward_inceptionv3),
    "vgg16": (lambda device: _build_vgg("vgg16", device), _forward_vgg),
    "vgg19": (lambda device: _build_vgg("vgg19", device), _forward_vgg),
}

MODEL_REGISTRY: dict[str, dict[str, Any]] = {}
for _model_key in MODEL_ORDER:
    _build_fn, _forward_fn = _MODEL_IMPLS[_model_key]
    MODEL_REGISTRY[_model_key] = {
        "description": MODEL_DESCRIPTIONS[_model_key],
        "layers": list(MODEL_LAYERS[_model_key]),
        "recommended": MODEL_RECOMMENDED[_model_key],
        "build": _build_fn,
        "forward_fn": _forward_fn,
        "min_size": MODEL_MIN_SIZE[_model_key],
        "defaults": dict(MODEL_DEFAULTS[_model_key]),
    }

DEFAULT_MODEL = "vgg16"

# TensorFlow-era layer name aliases (GoogLeNet only, kept for back-compat).
TF_LAYER_MAP = {
    "mixed3a": "inception3a",
    "mixed3b": "inception3b",
    "mixed4a": "inception4a",
    "mixed4b": "inception4b",
    "mixed4c": "inception4c",
    "mixed4d": "inception4d",
    "mixed4e": "inception4e",
    "mixed5a": "inception5a",
    "mixed5b": "inception5b",
}

# Filename-stem validation lives in model_data.py (shared with the GUI).
# Windows classic MAX_PATH is 260, macOS PATH_MAX is 1024, and Linux allows
# 4096. Using the Linux figure everywhere else disables the pre-flight check on
# the platform with the shortest limit, where deeply nested paths are common.
MAX_PATH = 260 if sys.platform == "win32" else (1024 if sys.platform == "darwin" else 4096)

MAX_OCTAVES = 25
# An octave whose short edge would drop below this is not built.
MIN_OCTAVE_SIZE = 32
TARGET_BASE_SIZE = 250

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

# Audio-remux ffmpeg only stream-copies; 600s is a stuck-process ceiling.
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


def resolve_layer(name, model_key):
    """Resolve a layer name (or TF alias) to a canonical key for the given model."""
    if model_key not in MODEL_REGISTRY:
        raise SystemExit(f"Unknown model '{model_key}'. Valid models: {sorted(MODEL_REGISTRY)}")
    name = name.strip()
    # Apply TF alias map (GoogLeNet only).
    canonical = TF_LAYER_MAP.get(name.lower(), name)
    valid_keys = {key for key, _ in MODEL_REGISTRY[model_key]["layers"]}
    if canonical in valid_keys:
        return canonical
    # Lower-case fallback for googlenet inception names.
    for key in valid_keys:
        if key.lower() == canonical.lower():
            return key
    valid_list = sorted(valid_keys)
    raise SystemExit(f"Unknown layer '{name}' for model '{model_key}'. Valid layers: {valid_list}")


def build_model(model_key, device):
    """Load and return the model for the given registry key."""
    if model_key not in MODEL_REGISTRY:
        raise SystemExit(f"Unknown model '{model_key}'. Valid models: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[model_key]["build"](device)


def run_to_layers(model, model_key, x, targets):
    """Forward pass through the model, capturing activations at each target layer."""
    if model_key not in MODEL_REGISTRY:
        raise SystemExit(f"Unknown model '{model_key}'. Valid models: {sorted(MODEL_REGISTRY)}")
    target_set = set(targets)
    if not target_set:
        raise RuntimeError("No target layers specified.")
    forward_fn = MODEL_REGISTRY[model_key]["forward_fn"]
    return forward_fn(model, x, target_set)


def _module_for_target(model, name):
    """Resolve a target name ('inception4c' or 'features.22') to its module."""
    module = model
    for part in name.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def forward_with_activations(model, x, targets):
    """Run one full forward pass, capturing target activations with hooks."""
    acts = {}
    handles = []

    def capture(name):
        def hook(_module, _inputs, output):
            acts[name] = output

        return hook

    unresolved = []
    try:
        for name in dict.fromkeys(targets):
            try:
                module = _module_for_target(model, name)
            except (AttributeError, IndexError, KeyError, TypeError, ValueError):
                unresolved.append(name)
                continue
            handles.append(module.register_forward_hook(capture(name)))
        if unresolved:
            raise RuntimeError(f"Target layer(s) not found in model: {sorted(unresolved)}")
        logits = model(x)
    finally:
        for handle in handles:
            handle.remove()
    missing = set(targets) - set(acts)
    if missing:
        raise RuntimeError(f"Target layer(s) not found in model: {sorted(missing)}")
    return acts, logits


def _activation_loss(acts, guide_features):
    """Feature-amplification or guide-matching loss over captured activations.
    Unguided: 0.5 * sum(a**2). Guided: cosine similarity between per-location
    normalized image and guide features, each scored at its nearest guide feature."""
    if not acts:
        raise RuntimeError("No activations captured: no target layers were run.")
    if guide_features:
        total = None
        for name, a in acts.items():
            if name not in guide_features:
                continue
            fx = a[0].reshape(a.shape[1], -1)  # (C, Nx)
            fx_norm = fx / (fx.norm(p=2, dim=0, keepdim=True) + 1e-7)
            sim = torch.mm(fx_norm.t(), guide_features[name])  # (Nx, Ng)
            best_sim, _ = sim.max(dim=1)
            total = best_sim.sum() if total is None else total + best_sim.sum()
        if total is not None:
            return total
        raise RuntimeError(
            "Guide features were supplied but none overlap the captured "
            f"layer(s) {sorted(acts)} (guide feature layers: {sorted(guide_features)}). "
            "A guide dream would otherwise run silently with no guide effect."
        )
    return 0.5 * sum((a**2).sum() for a in acts.values())


def resize_dims(size, max_size):
    """Target (w, h) when downscaling to max_size is needed, else None."""
    w, h = size
    if max_size and max_size > 0:
        scale = max_size / max(w, h)
        if scale < 1.0:
            return (max(1, round(w * scale)), max(1, round(h * scale)))
    return None


def load_pil_image(path):
    """Open an image, apply EXIF orientation, and convert it to RGB."""
    with Image.open(path) as raw_img:
        return ImageOps.exif_transpose(raw_img).convert("RGB")


def load_image_native_size(path):
    """Native (w, h) from the image header, without decoding pixel data.
    EXIF rotation preserves min/max edge length, so validation and naming need
    no second decode."""
    with Image.open(path) as raw_img:
        return raw_img.size


def load_image_with_size(path, max_size):
    """Load an image plus its native (w, h), from a single file open."""
    img = load_pil_image(path)
    native_size = img.size
    new_size = resize_dims(img.size, max_size)
    if new_size:
        img = img.resize(new_size, getattr(Image, "Resampling", Image).LANCZOS)
    return pil_to_tensor(img), native_size


def load_image(path, max_size):
    return load_image_with_size(path, max_size)[0]


def tensor_to_np(t):
    t = t.detach().clone().cpu().float()
    if t.dim() == 4:
        if t.shape[0] != 1:
            raise ValueError(
                f"tensor_to_np expects a single image (batch==1), got shape {tuple(t.shape)}"
            )
        t = t[0]
    for c in range(3):
        t[c] = t[c] * STD[c] + MEAN[c]
    t = t.clamp(0, 1).mul(255).round().byte()
    return t.permute(1, 2, 0).numpy()


def pil_to_tensor(img):
    arr = np.asarray(img, dtype="float32") / 255.0
    mean = np.array(MEAN, dtype="float32")
    std = np.array(STD, dtype="float32")
    arr = (arr - mean) / std
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def tensor_to_pil(t):
    return Image.fromarray(tensor_to_np(t))


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


def _atomic_tmp_path(path, suffix=".tmp"):
    """Pid-suffixed sibling temp path, so concurrent writers never share a temp file."""
    return f"{path}.p{os.getpid()}{suffix}"


def _remove_stale_temps(*paths):
    """Best-effort removal of temp files left by a previous (crashed) run."""
    for path in paths:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


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


def _append_saved_video_frames(writer, frames_dir, count):
    """Seed a resumed MP4 encode with the frames that were already generated."""
    for frame_idx in range(1, count + 1):
        frame_path = os.path.join(frames_dir, f"frame_{frame_idx:06d}.png")
        with Image.open(frame_path) as frame:
            writer.append_data(np.asarray(frame.convert("RGB")))


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
        macro_block_size=1,
        **kwargs,
    )


def _publish_video(temp_out, out_path, what="video"):
    """Atomically move a completed encode from its temp file to *out_path*.
    A locked destination is reported (the temp file is preserved) and re-raised,
    so the caller can keep the temp and skip cleanup."""
    try:
        os.replace(temp_out, out_path)
    except PermissionError:
        print(
            f"Warning: Destination file '{out_path}' is locked by another program. "
            f"Completed {what} preserved at '{temp_out}'.",
            file=sys.stderr,
            flush=True,
        )
        raise


def _save_frame_atomic(frame_np, frames_dir, index):
    """Save one sequence frame as frame_NNNNNN.png via temp file + rename."""
    temp_frame = os.path.join(frames_dir, f".frame_{index:06d}.tmp.png")
    Image.fromarray(frame_np).save(temp_frame, format="PNG")
    os.replace(temp_frame, os.path.join(frames_dir, f"frame_{index:06d}.png"))


def save_image(t, path, warn_on_overwrite=False):
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    if warn_on_overwrite and os.path.exists(path):
        print(
            f"Warning: output file '{path}' already exists and will be "
            "overwritten. Use --name or --out to keep the previous result.",
            file=sys.stderr,
            flush=True,
        )
    arr = tensor_to_np(t)
    img = Image.fromarray(arr)
    # Encode to a sibling temp file and rename on success so a failure (disk
    # full, interrupt) never truncates a pre-existing image at `path`.
    fmt = Image.registered_extensions().get(os.path.splitext(path)[1].lower())
    if not fmt:
        fmt = "PNG"
    tmp_path = _atomic_tmp_path(path, ".tmp")
    keep_temp = False
    try:
        img.save(tmp_path, format=fmt)
        try:
            os.replace(tmp_path, path)
        except PermissionError:
            keep_temp = True
            print(
                f"Warning: Destination file '{path}' is locked by another process. "
                f"Rendered image preserved at '{tmp_path}'.",
                file=sys.stderr,
                flush=True,
            )
            raise
    except BaseException:
        if not keep_temp:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
        raise


def save_preview_frame(frame_np, preview_file):
    """Atomically write a uint8 (H,W,3) preview frame so a concurrent GUI
    reader never sees a torn file. Best-effort: returns False on recoverable
    failure, never raises (Interrupt/SystemExit still propagate)."""
    tmp_path = _atomic_tmp_path(preview_file, ".tmp")
    try:
        dirname = os.path.dirname(preview_file)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        img = Image.fromarray(frame_np)
        fmt = Image.registered_extensions().get(os.path.splitext(preview_file)[1].lower())
        if not fmt:
            fmt = "PNG"
        img.save(tmp_path, format=fmt)
        os.replace(tmp_path, preview_file)
        return True
    except (OSError, ValueError):
        return False
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass


# Iteration frames written by single-image --frames/--timelapse runs.
_STALE_FRAME_RE = re.compile(r"^o\d+_i\d+\.png$", re.IGNORECASE)


def clear_stale_frames(frames_dir: str | os.PathLike | None, allow_delete: bool = False) -> None:
    """Clear old iteration frames only when explicitly requested."""
    if not frames_dir or not os.path.isdir(frames_dir):
        return
    try:
        entries = os.listdir(frames_dir)
    except OSError as exc:
        raise SystemExit(f"Error: cannot inspect --frames-dir '{frames_dir}': {exc}") from exc
    foreign = [name for name in entries if not _STALE_FRAME_RE.match(name)]
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


def resize(t, size):
    return F.interpolate(t, size=size, mode="bilinear", align_corners=False, antialias=True)


def gradient_ascent(
    model,
    model_key,
    img,
    targets,
    lr,
    jitter,
    guide_features=None,
    target_class=None,
):
    if not targets and target_class is None:
        raise SystemExit("Error: No layer targets or class target specified.")
    sx = sy = 0
    if jitter:
        sx, sy = random.randint(-jitter, jitter), random.randint(-jitter, jitter)
        img = torch.roll(img, shifts=(sx, sy), dims=(2, 3))

    img = img.detach().clone().requires_grad_(True)
    loss = torch.tensor(0.0, device=img.device)

    if targets and target_class is not None:
        # Single full forward: the classifier runs while hooks capture the
        # target activations, so the trunk isn't forwarded twice when layer
        # targets and a class target are combined.
        acts, logits = forward_with_activations(model, img, targets)
        layer_loss = _activation_loss(acts, guide_features)
        class_loss = logits[0, target_class]
        # The activation sum (~10^5 elements) dwarfs a single class logit by
        # ~10^3x; normalize each gradient to unit mean-absolute magnitude so
        # both terms steer equally. Costs one extra backward, not a second forward.
        grad_layer = torch.autograd.grad(layer_loss, img, retain_graph=True)[0]
        grad_class = torch.autograd.grad(class_loss, img)[0]
        grad = grad_layer / (grad_layer.abs().mean() + 1e-8) + grad_class / (
            grad_class.abs().mean() + 1e-8
        )
        loss = loss + layer_loss.detach() + class_loss.detach()
    elif targets:
        acts = run_to_layers(model, model_key, img, targets)
        loss = loss + _activation_loss(acts, guide_features)
        loss.backward()
        grad = img.grad
    elif target_class is not None:
        logits = model(img)
        loss = loss + logits[0, target_class]
        loss.backward()
        grad = img.grad

    with torch.no_grad():
        if grad is None:
            raise RuntimeError(
                "Gradient was None during ascent: the computational "
                "graph is disconnected (no path from loss to input)."
            )
        if not bool(torch.isfinite(grad).all()):
            raise RuntimeError("Gradient was NaN/Inf during ascent: try a lower --lr.")
        grad = grad / (grad.abs().mean() + 1e-8)
        img = img + lr * grad
        for c in range(3):
            img[:, c] = img[:, c].clamp(-MEAN[c] / STD[c], (1 - MEAN[c]) / STD[c])
        img = img.detach()
        if jitter:
            img = torch.roll(img, shifts=(-sx, -sy), dims=(2, 3))
    return img, float(loss.detach())


def auto_octave_count(min_dim, octave_scale, target_base_size=TARGET_BASE_SIZE):
    """Octaves auto mode asks for, before the MAX_OCTAVES cap is applied."""
    if not math.isfinite(octave_scale) or octave_scale <= 1.0:
        raise ValueError(f"octave_scale must be > 1.0, got {octave_scale}")
    if min_dim > target_base_size:
        return max(1, round(math.log(min_dim / target_base_size, octave_scale)) + 1)
    return 1


def plan_octaves(
    h,
    w,
    num_octaves,
    octave_scale,
    min_octave_size=MIN_OCTAVE_SIZE,
    target_base_size=TARGET_BASE_SIZE,
):
    """Number of octaves deepdream() will actually run for an hxw image."""
    if not math.isfinite(octave_scale) or octave_scale <= 1.0:
        raise ValueError(f"octave_scale must be > 1.0, got {octave_scale}")
    if num_octaves == 0:
        num_octaves = auto_octave_count(min(h, w), octave_scale, target_base_size)
    num_octaves = min(num_octaves, MAX_OCTAVES)
    count = 1
    hh, ww = h, w
    for _ in range(num_octaves - 1):
        hh = round(hh / octave_scale)
        ww = round(ww / octave_scale)
        if min(hh, ww) < min_octave_size:
            break
        count += 1
    return count


def octave_truncation_note(requested, actual, octave_scale, auto_capped=False):
    """Why fewer octaves than requested will run, or None if all will run."""
    if auto_capped:
        if actual < MAX_OCTAVES:
            return (
                f"auto octaves would exceed the {MAX_OCTAVES}-octave cap and the image "
                f"can't fit {MAX_OCTAVES} octaves at octave-scale {octave_scale}; "
                f"running {actual} instead."
            )
        return (
            f"auto octaves capped at {MAX_OCTAVES}: octave-scale {octave_scale} is "
            f"so close to 1.0 that reaching the ~{TARGET_BASE_SIZE}px base octave "
            f"would take more than {MAX_OCTAVES} levels; running {actual} instead."
        )
    if requested <= 0 or actual >= requested:
        return None
    cap_bound = requested > MAX_OCTAVES
    size_bound = actual < min(requested, MAX_OCTAVES)
    if cap_bound and size_bound:
        return (
            f"--octaves {requested} exceeds the {MAX_OCTAVES}-octave cap and the "
            f"image can't fit {MAX_OCTAVES} octaves at octave-scale {octave_scale}; "
            f"running {actual} instead."
        )
    if cap_bound:
        return f"--octaves {requested} capped at {MAX_OCTAVES} octaves; running {actual} instead."
    return (
        f"image too small for {requested} octaves at octave-scale {octave_scale}; "
        f"running {actual} instead."
    )


def validate_input_size(h, w, model_key, what="Input image", hint=None):
    """Reject an input too small for the model's trunk to run without collapsing."""
    if model_key not in MODEL_REGISTRY:
        raise SystemExit(f"Unknown model '{model_key}'. Valid models: {sorted(MODEL_REGISTRY)}")
    need = MODEL_REGISTRY[model_key]["min_size"]
    if need and min(h, w) < need:
        msg = (
            f"Error: {what} is too small ({w}x{h}); model '{model_key}' "
            f"needs at least {need}px on the short edge."
        )
        if hint:
            msg += f" {hint}"
        raise SystemExit(msg)


def deepdream(
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
    frames_dir=None,
    guide_features=None,
    target_class=None,
    preview_file=None,
    clear_frames: bool = False,
):
    if not targets and target_class is None:
        raise SystemExit("Error: No layer targets or class target specified.")
    if not math.isfinite(octave_scale):
        raise SystemExit(f"Error: octave_scale must be finite, got {octave_scale}")
    if octave_scale <= 1.0:
        raise SystemExit(f"Error: octave_scale must be > 1.0, got {octave_scale}")
    if num_octaves < 0:
        raise SystemExit(f"Error: num_octaves must be >= 0, got {num_octaves}")
    if iterations < 0:
        raise SystemExit(f"Error: iterations must be >= 0, got {iterations}")
    if not math.isfinite(lr):
        raise SystemExit(f"Error: lr must be finite, got {lr}")
    if lr <= 0:
        raise SystemExit(f"Error: lr must be > 0, got {lr}")
    if jitter < 0:
        raise SystemExit(f"Error: jitter must be >= 0, got {jitter}")

    validate_input_size(img.shape[2], img.shape[3], model_key)

    if frames_dir and iterations > 0:
        clear_stale_frames(frames_dir, allow_delete=clear_frames)
    elif frames_dir:
        print(
            "Note: --iterations is 0, so no frames will be written or changed.",
            flush=True,
        )

    # No full-image .to(device) here: the pyramid below is built and kept on
    # CPU, and only one octave at a time moves to the device in the loop.

    requested_octaves = num_octaves
    model_min = MODEL_REGISTRY.get(model_key, {}).get("min_size", MIN_OCTAVE_SIZE)
    effective_min_octave = max(MIN_OCTAVE_SIZE, model_min)
    effective_target_base = max(TARGET_BASE_SIZE, model_min)

    num_octaves = plan_octaves(
        img.shape[2],
        img.shape[3],
        num_octaves,
        octave_scale,
        min_octave_size=effective_min_octave,
        target_base_size=effective_target_base,
    )
    auto_capped = (
        requested_octaves == 0
        and auto_octave_count(
            min(img.shape[2], img.shape[3]),
            octave_scale,
            target_base_size=effective_target_base,
        )
        > MAX_OCTAVES
    )

    # Pyramid lives on CPU; only one octave moves to `device` at a time,
    # keeping peak VRAM at ~3x the current level rather than the full pyramid.
    base_cpu = img.detach().to("cpu")
    octaves_cpu = [base_cpu]
    for _ in range(num_octaves - 1):
        cur = octaves_cpu[-1]
        h_scaled = round(cur.shape[2] / octave_scale)
        w_scaled = round(cur.shape[3] / octave_scale)
        if min(h_scaled, w_scaled) < effective_min_octave:
            break
        octaves_cpu.append(resize(cur, [h_scaled, w_scaled]))
    octaves_cpu.reverse()
    note = octave_truncation_note(
        requested_octaves, len(octaves_cpu), octave_scale, auto_capped=auto_capped
    )
    if note:
        print(f"Note: {note}", flush=True)

    detail_cpu = torch.zeros_like(octaves_cpu[0])
    img_w = None
    for oi, octave_cpu in enumerate(octaves_cpu):
        if detail_cpu.shape[2:] != octave_cpu.shape[2:]:
            detail_cpu = resize(detail_cpu, [octave_cpu.shape[2], octave_cpu.shape[3]])
        octave = octave_cpu.to(device)
        detail_d = detail_cpu.to(device)
        img_w = (octave + detail_d).detach()
        del detail_d
        print(
            f"Octave {oi + 1}/{len(octaves_cpu)}  image {img_w.shape[3]}x{img_w.shape[2]}",
            flush=True,
        )
        emit_event(EV_OCTAVE, current=oi + 1, total=len(octaves_cpu))
        for it in range(iterations):
            img_w, loss = gradient_ascent(
                model,
                model_key,
                img_w,
                targets,
                lr,
                jitter,
                guide_features=guide_features,
                target_class=target_class,
            )
            print(
                f"  iteration {it + 1}/{iterations}   loss {loss:9.1f}",
                flush=True,
            )
            emit_event(EV_ITERATION, current=it + 1, total=iterations)
            if frames_dir:
                save_image(img_w, os.path.join(frames_dir, f"o{oi + 1}_i{it + 1}.png"))

        detail_cpu = (img_w - octave).detach().to("cpu")
        del octave
        if preview_file:
            try:
                if save_preview_frame(tensor_to_np(img_w), preview_file):
                    print(f"[PREVIEW] {preview_file}", flush=True)
                    emit_event(EV_PREVIEW, path=preview_file)
            except (OSError, ValueError):
                pass
        if oi < len(octaves_cpu) - 1:
            # Free before allocating the next (larger) octave; detail carries forward on CPU.
            del img_w
            img_w = None

    if img_w is None:
        img_w = base_cpu.to(device)
    return img_w


def resolve_video_quality(video_quality="balanced", crf=None, preset=None):
    """Resolve H.264 CRF and encoding speed preset from quality name or explicit values."""
    vq_key = (video_quality or "balanced").lower().strip()
    if vq_key in VIDEO_QUALITY_PRESETS:
        target_crf = VIDEO_QUALITY_PRESETS[vq_key]["crf"]
        target_preset = VIDEO_QUALITY_PRESETS[vq_key]["preset"]
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

    return target_crf, target_preset


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
):
    if config is not None:
        fps = config.fps
        preview_file = config.preview_file if preview_file is None else preview_file
        crf = config.crf
        preset = config.preset
        zoom_cx = config.zoom_cx
        zoom_cy = config.zoom_cy
        frames_dir = config.frames_dir if frames_dir is None else frames_dir
        source = config.source if source is None else source
        size = config.size if size is None else size
        custom_name = config.custom_name if custom_name is None else custom_name
        class_name = config.class_name if class_name is None else class_name
        guide = config.guide if guide is None else guide
        resume = resume if resume else getattr(config, "resume", False)
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
    img = img.to(device)
    # Ensure dimensions are even for H.264 encoder compatibility
    ih, iw = img.shape[2], img.shape[3]
    even_w = iw if (iw % 2 == 0) else iw + 1
    even_h = ih if (ih % 2 == 0) else ih + 1
    if (even_w, even_h) != (iw, ih):
        img = resize(img, (even_h, even_w))
    validate_input_size(img.shape[2], img.shape[3], model_key)

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
    print(
        f"Generating zoom animation: {num_frames} frames @ {fps} fps "
        f"({even_w}x{even_h}, zoom_scale={zoom_scale}, rot={zoom_rot}\u00b0"
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
                _append_saved_video_frames(writer, frames_dir, start_frame)
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
                    writer.append_data(frame_np)
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
            _publish_video(temp_out, out_path)
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
):
    if config is not None:
        preview_file = config.preview_file if preview_file is None else preview_file
        crf = config.crf
        preset = config.preset
        frames_dir = config.frames_dir if frames_dir is None else frames_dir
        custom_name = config.custom_name if custom_name is None else custom_name
        class_name = config.class_name if class_name is None else class_name
        guide = config.guide if guide is None else guide
        resume = resume if resume else getattr(config, "resume", False)
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
    if frames_dir:
        ensure_output_dir(frames_dir)

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
        canvas_w, canvas_h = _video_canvas_size(_first_video_frame_size(video_in_path), max_size)
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
    try:
        # Created inside the outer try (like dream_zoom_animation) so a
        # get_writer() failure still runs the temp-file cleanup below
        # instead of leaking a partial *.temp.mp4.
        if encode and temp_out:
            writer = _open_video_writer(temp_out, fps, output_params)
            if start_frame > 0 and frames_dir:
                _append_saved_video_frames(writer, frames_dir, start_frame)
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
                    if frames_dir:
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
                    writer.append_data(frame_np)
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


# Sequential per-frame filenames written by zoom/video generation when a
# frames dir is given. Kept distinct from the single-image `o{oct}_i{iter}`
# timelapse pattern so the two encoders can tell their inputs apart.
VIDEO_FRAME_PATTERN = re.compile(r"^frame_(\d+)\.png$", re.IGNORECASE)


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
        and not n.lower().startswith("_gui_preview")
        and not n.lower().endswith(".tmp")
    ]
    generic.sort(
        key=lambda text: [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", text)]
    )
    return [os.path.join(frames_dir, n) for n in generic]


def _encode_png_sequence(files, out_path, fps=30, crf=17, preset="medium", what="video"):
    """Encode sorted PNG *files* to an H.264 .mp4 at *out_path* (atomic)."""
    ensure_output_dir(os.path.dirname(out_path) or ".")
    with Image.open(files[-1]) as last_img:
        fw, fh = last_img.size
    target_w = max(2, fw - (fw % 2))
    target_h = max(2, fh - (fh % 2))
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
                if img.size != (target_w, target_h):
                    img = img.resize(
                        (target_w, target_h), getattr(Image, "Resampling", Image).LANCZOS
                    )
                writer.append_data(np.asarray(img))
                pct = int(idx / total_frames * 100)
                print(f"Encoding frame {idx}/{total_frames} ({pct}%)", flush=True)
                emit_event(EV_FRAME, current=idx, total=total_frames)
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
    """Remux *audio_source*'s audio onto *silent_mp4* to *final_out*, atomically.
    Stream-copies through a temp file and returns the final path; a failed remux
    warns and falls back to the silent video."""
    remux_out = _atomic_tmp_path(final_out, ".remux.temp.mp4")
    _remove_stale_temps(remux_out, final_out + ".remux.temp.mp4")
    target = silent_mp4
    try:
        print("Remuxing audio track...", flush=True)
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
    try:
        os.replace(target, final_out)
        if target is remux_out and os.path.exists(silent_mp4):
            try:
                os.remove(silent_mp4)
            except OSError:
                pass
    except PermissionError:
        print(
            f"Warning: Destination file '{final_out}' is locked by another program. "
            f"Completed video preserved at '{target}'.",
            file=sys.stderr,
            flush=True,
        )
        raise
    return final_out


def compile_video_frames(frames_dir, out_path, fps=30, crf=17, preset="medium", audio_from=None):
    """Encode a saved frame sequence (frame_*.png) to .mp4.
    The save half of the generate/save split, so frames export at any quality
    without re-running the dream; audio from *audio_from* or the meta is remuxed."""
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
    temp_out = _encode_png_sequence(files, out_path, fps=fps, crf=crf, preset=preset)
    keep_temp = False
    try:
        if audio_from and os.path.isfile(audio_from):
            try:
                return _remux_audio(temp_out, audio_from, out_path)
            except PermissionError:
                keep_temp = True
                raise
        try:
            _publish_video(temp_out, out_path)
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


def compile_timelapse(frames_dir, out_path, fps=30, crf=17, preset="medium"):
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

    temp_out = _encode_png_sequence(
        files, out_path, fps=fps, crf=crf, preset=preset, what="timelapse"
    )
    try:
        _publish_video(temp_out, out_path, "timelapse")
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
    args: argparse.Namespace, video_crf: int, video_preset: str, fps_explicit: bool = False
) -> None:
    """Encode-only mode: --encode-frames DIR --out OUT.mp4 (no dreaming).
    Reads fps/audio defaults from the sequence meta.json unless overridden."""
    frames_dir = args.encode_frames
    if not os.path.isdir(frames_dir):
        raise SystemExit(f"Error: frames directory '{frames_dir}' does not exist.")
    files = list_video_frames(frames_dir)
    if not files:
        raise SystemExit(f"Error: No PNG frames found in '{frames_dir}'.")
    meta = read_video_frames_meta(frames_dir)
    out = args.out
    if os.path.isdir(out):
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
    if not out.lower().endswith(".mp4"):
        raise SystemExit(f"Error: --out must be an .mp4 path in encode mode, got '{args.out}'.")
    audio_from = args.encode_audio_from or meta.get("source")
    if not has_audio_stream(audio_from):
        audio_from = None
    temp_suffix = (
        f".p{os.getpid()}.remux.temp.mp4"
        if (audio_from and os.path.isfile(audio_from))
        else f".p{os.getpid()}.temp.mp4"
    )
    validate_output_path(out, extra_suffix=temp_suffix)
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
        help="with --frames/--timelapse, replace existing iteration frames in --frames-dir",
    )
    ap.add_argument(
        "--frames-dir",
        default=None,
        help="directory for frames (default: <out>/frames or "
        "<out>/frames_<name> for multi-image runs; used by the GUI to isolate queued jobs). "
        "With --zoom/--video it persists the dreamed video frames as "
        "frame_*.png (+ meta.json) for later --encode-frames exports",
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
        help="with --encode-frames: remux the audio track from SOURCE_VIDEO "
        "onto the export (defaults to the sequence meta.json source, if any)",
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
        "balanced (CRF 17, default), lossless (CRF 0)",
    )
    ap.add_argument(
        "--crf",
        type=int,
        default=None,
        help="H.264 Constant Rate Factor "
        "(0=lossless to 51=max compression, overrides --video-quality)",
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
) -> tuple[str, list[str], int | None, str | None, int, str] | None:
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
    if args.clear_frames and not (args.frames or args.timelapse):
        raise SystemExit("Error: --clear-frames requires --frames or --timelapse.")
    if args.encode_audio_from and not args.encode_frames:
        raise SystemExit("Error: --encode-audio-from requires --encode-frames.")
    fps_explicit = args.zoom_fps is not None
    if args.zoom_fps is None:
        args.zoom_fps = 30

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
        video_crf, video_preset = resolve_video_quality(
            video_quality=args.video_quality, crf=args.crf
        )
    except ValueError as exc:
        raise SystemExit(f"Error: {exc}") from None

    if args.encode_frames:
        _run_encode_frames_mode(args, video_crf, video_preset, fps_explicit)
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
    return model_key, targets, class_id, class_name, video_crf, video_preset


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
        validate_output_path(out_path, extra_suffix=temp_suffix)
        if args.timelapse:
            tl_path = os.path.join(args.out, f"{name}{suffix}_timelapse.mp4")
            validate_output_path(tl_path, extra_suffix=f".p{os.getpid()}.temp.mp4")

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
        if frames_dir and args.iterations > 0:
            if args.zoom or args.video:
                max_frame = args.zoom_frames if args.zoom else 999999
                validate_output_path(
                    os.path.join(frames_dir, f"frame_{max_frame:06d}.png"),
                    extra_suffix=f".p{os.getpid()}.tmp",
                )
            else:
                model_min = MODEL_REGISTRY.get(model_key, {}).get("min_size", MIN_OCTAVE_SIZE)
                n_octaves = plan_octaves(
                    target_h,
                    target_w,
                    args.octaves,
                    args.octave_scale,
                    min_octave_size=max(MIN_OCTAVE_SIZE, model_min),
                    target_base_size=max(TARGET_BASE_SIZE, model_min),
                )
                validate_output_path(
                    os.path.join(frames_dir, f"o{n_octaves}_i{args.iterations}.png"),
                    extra_suffix=f".p{os.getpid()}.tmp",
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
                zoom_cx=args.zoom_cx,
                zoom_cy=args.zoom_cy,
                frames_dir=frames_dir,
                source=current_image,
                size=args.size,
                custom_name=args.name,
                class_name=class_name,
                guide=args.guide,
                resume=bool(args.resume),
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


def main(argv: list[str] | None = None) -> None:
    """Entry point for the DeepDream CLI."""
    _configure_streams()
    check_runtime_environment()
    ap = build_cli_parser()
    args = ap.parse_args(argv)

    resolved = _resolve_and_validate_args(args)
    if resolved is None:
        return
    model_key, targets, class_id, class_name, video_crf, video_preset = resolved

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
