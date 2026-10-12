"""Video export planning without dialogs, widgets, or subprocess lifecycle state."""

from __future__ import annotations

import os
from dataclasses import dataclass

from .gui_helpers import venv_python
from .gui_job import GenerationJob
from .model_data import (
    build_systematic_video_name,
    read_video_frames_meta,
    resolve_name_conflict,
)


@dataclass(frozen=True)
class VideoExportRequest:
    """Snapshot export settings before launching a background encode."""

    frames_dir: str
    out_path: str
    fps: float
    quality: str
    crf: int
    audio_from: str | None = None

    def build_cmd(self) -> list[str]:
        cmd = [
            venv_python(),
            "-m",
            "app.deepdream",
            "--encode-frames",
            self.frames_dir,
            "--out",
            self.out_path,
            "--fps",
            f"{self.fps:.2f}",
        ]
        quality = self.quality.lower()
        if quality in ("small", "balanced", "lossless"):
            cmd += ["--video-quality", quality]
        cmd += ["--crf", str(self.crf)]
        if self.audio_from and os.path.isfile(self.audio_from):
            cmd += ["--encode-audio-from", self.audio_from]
        return cmd


def suggest_video_name(
    *,
    job: GenerationJob | None,
    frames_dir: str | None,
    source: str | None,
    mode: str | None,
    crf: int,
    out_dir: str,
) -> str:
    """Combine the generating job and persisted metadata into a non-conflicting name."""
    meta = read_video_frames_meta(frames_dir) if frames_dir else {}
    candidate = build_systematic_video_name(
        source=getattr(job, "image_path", None) or source or meta.get("source"),
        custom_name=getattr(job, "custom_name", None) or meta.get("custom_name"),
        size=getattr(job, "size", None) if job is not None else meta.get("size"),
        width=meta.get("width"),
        height=meta.get("height"),
        model=getattr(job, "model_key", None) or meta.get("model"),
        layers=getattr(job, "layer_spec", None) or meta.get("layers"),
        iterations=(
            getattr(job, "iterations", None) if job is not None else meta.get("iterations")
        ),
        target_class=getattr(job, "target_class", None) or meta.get("target_class"),
        class_name=meta.get("class_name"),
        guide=getattr(job, "guide_path", None) or meta.get("guide"),
        mode=getattr(job, "mode", None) or mode or meta.get("mode") or "zoom",
        crf=crf,
    )
    return resolve_name_conflict(out_dir, candidate)
