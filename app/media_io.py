"""Image decoding, normalized tensor conversion, and atomic media publication.

This module owns file I/O, not generation, encoding, or frame-sequence recovery.
Completed outputs replace their destinations only after successful writes.
"""

import os
import sys
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image, ImageOps
from torch import Tensor

MediaPath = str | os.PathLike[str]
ImageSize = tuple[int, int]
RGBFrame = NDArray[np.uint8]

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def resize_dims(size: ImageSize, max_size: int | None) -> ImageSize | None:
    """Return downscaled (w, h), or None when no downscaling is needed."""
    w, h = size
    if max_size and max_size > 0:
        scale = max_size / max(w, h)
        if scale < 1.0:
            return (max(1, round(w * scale)), max(1, round(h * scale)))
    return None


def load_pil_image(path: MediaPath) -> Image.Image:
    """Decode RGB pixels with EXIF orientation applied; close the source file."""
    with Image.open(path) as raw_img:
        return ImageOps.exif_transpose(raw_img).convert("RGB")


def load_image_native_size(path: MediaPath) -> ImageSize:
    """Read header dimensions without decoding pixels or applying EXIF rotation.

    Rotation preserves minimum/maximum edge lengths, which is sufficient for
    preflight validation and naming. load_image_with_size returns oriented size.
    """
    with Image.open(path) as raw_img:
        return raw_img.size


def load_image_with_size(path: MediaPath, max_size: int | None) -> tuple[Tensor, ImageSize]:
    """Load a normalized tensor and oriented native dimensions in one file open."""
    img = load_pil_image(path)
    native_size = img.size
    new_size = resize_dims(img.size, max_size)
    if new_size:
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    return pil_to_tensor(img), native_size


def load_image(path: MediaPath, max_size: int | None) -> Tensor:
    return load_image_with_size(path, max_size)[0]


def tensor_to_np(t: Tensor) -> RGBFrame:
    """Convert a single normalized CHW/NCHW image without modifying its tensor."""
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
    return cast(RGBFrame, t.permute(1, 2, 0).numpy())


def pil_to_tensor(img: Image.Image) -> Tensor:
    """Convert an RGB image to an ImageNet-normalized 1x3xHxW tensor."""
    arr = np.asarray(img, dtype="float32") / 255.0
    mean = np.array(MEAN, dtype="float32")
    std = np.array(STD, dtype="float32")
    arr = (arr - mean) / std
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def tensor_to_pil(t: Tensor) -> Image.Image:
    return Image.fromarray(tensor_to_np(t))


def atomic_tmp_path(path: MediaPath, suffix: str = ".tmp") -> str:
    """Use a PID-suffixed sibling temp path, avoiding cross-process collisions."""
    return f"{os.fspath(path)}.p{os.getpid()}{suffix}"


def remove_stale_temps(*paths: MediaPath | None) -> None:
    """Best-effort removal of explicitly supplied temporary files."""
    for path in paths:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


def publish_video(temp_out: MediaPath, out_path: MediaPath, what: str = "video") -> None:
    """Publish a completed encode; preserve it if the destination is locked."""
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


def save_frame_atomic(frame_np: RGBFrame, frames_dir: MediaPath, index: int) -> None:
    """Publish one sequence frame via sibling temporary file and replacement."""
    temp_frame = os.path.join(frames_dir, f".frame_{index:06d}.tmp.png")
    Image.fromarray(frame_np).save(temp_frame, format="PNG")
    os.replace(temp_frame, os.path.join(frames_dir, f"frame_{index:06d}.png"))


def save_image(t: Tensor, path: MediaPath, warn_on_overwrite: bool = False) -> None:
    """Save atomically; preserve completed temp output on a locked destination."""
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
    img = Image.fromarray(tensor_to_np(t))
    fmt = Image.registered_extensions().get(os.path.splitext(path)[1].lower()) or "PNG"
    tmp_path = atomic_tmp_path(path)
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
            remove_stale_temps(tmp_path)
        raise


def save_preview_frame(frame_np: RGBFrame, preview_file: MediaPath) -> bool:
    """Publish a preview best-effort; interrupts propagate and temps are cleaned."""
    tmp_path = atomic_tmp_path(preview_file)
    try:
        dirname = os.path.dirname(preview_file)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        img = Image.fromarray(frame_np)
        fmt = Image.registered_extensions().get(os.path.splitext(preview_file)[1].lower()) or "PNG"
        img.save(tmp_path, format=fmt)
        os.replace(tmp_path, preview_file)
        return True
    except (OSError, ValueError):
        return False
    finally:
        remove_stale_temps(tmp_path)
