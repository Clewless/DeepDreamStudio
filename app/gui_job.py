"""Job queue representations and subprocess progress parsing for DeepDream Studio."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

from .gui_helpers import venv_python
from .model_data import VIDEO_QUALITY_PRESETS
from .protocol import EV_FRAME, EV_ITERATION, EV_OCTAVE, EV_STATUS, parse_event


def _as_int(value: Any) -> int:
    """Best-effort int from an event field; 0 when unusable."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


# Display copy per quality preset; CRF values come from the shared table
_VIDEO_QUALITY_DESCRIPTIONS = {
    "Small": "Small: compact file, medium compression (CRF {crf}, fast)",
    "Balanced": "Balanced: high quality (CRF {crf}, recommended)",
    "Lossless": "Lossless (visually): maximum detail (CRF {crf}, 2× resolution)",
}


class _VideoQualityInfo(TypedDict):
    crf: int
    scale: int
    desc: str


VIDEO_QUALITY_INFO: dict[str, _VideoQualityInfo] = {
    name: {
        "crf": VIDEO_QUALITY_PRESETS[name.lower()]["crf"],
        "scale": VIDEO_QUALITY_PRESETS[name.lower()]["scale"],
        "desc": template.format(crf=VIDEO_QUALITY_PRESETS[name.lower()]["crf"]),
    }
    for name, template in _VIDEO_QUALITY_DESCRIPTIONS.items()
}

# Reverse lookup: CRF value -> quality preset name. Only 1x presets: a 2x preset
# (Lossless) is chosen by name, so a bare CRF value never changes resolution.
_CRF_TO_QUALITY: dict[int, str] = {}
for _name, _info in VIDEO_QUALITY_INFO.items():
    if _info["scale"] == 1:
        _CRF_TO_QUALITY.setdefault(_info["crf"], _name)


@dataclass
class GenerationJob:
    """Snapshot of user parameters representing a queued generation task."""

    id: int
    image_path: str
    model_key: str
    layer_spec: str
    iterations: int
    octaves: int
    octave_scale: float
    lr: float
    size: int
    jitter: int
    out_dir: str
    mode: Literal["Single Image", "Dream Zoom", "Dream Video"] = "Single Image"
    timelapse: bool = False
    zoom_frames: int = 60
    zoom_scale: float = 1.03
    zoom_rot: float = 0.0
    zoom_fps: int = 30
    zoom_cx: float = 0.5
    zoom_cy: float = 0.5
    video_blend: float = 0.5
    video_quality: Literal["Small", "Balanced", "Lossless", "Custom"] = "Balanced"
    video_crf: int = 17
    # Geometry preset retained when the CRF slider marks the quality "Custom".
    video_preset: Literal["Small", "Balanced", "Lossless"] | None = None
    guidance_mode: Literal["Unguided", "Guide Image", "Target Class"] = "Unguided"
    guide_path: str | None = None
    target_class: str | None = None
    custom_name: str | None = None
    save_frames: bool = False
    no_cuda: bool = False
    status: Literal["Pending", "Running", "Done", "Stopped", "Failed"] = "Pending"
    output_path: str | None = None
    timelapse_path: str | None = None
    cmd: list[str] = field(default_factory=list)
    frames_dir: str | None = None
    resume: bool = False
    seed: int | None = None

    def build_cmd(self, preview_tmp_path: str | None = None) -> list[str]:
        """Construct the CLI command arguments to execute this job."""
        cmd = [
            venv_python(),
            "-m",
            "app.deepdream",
            self.image_path,
            "--model",
            self.model_key,
        ]
        if self.guidance_mode != "Target Class" or self.layer_spec:
            cmd += ["--layer", self.layer_spec]

        cmd += [
            "--iterations",
            str(self.iterations),
            "--octaves",
            str(self.octaves),
            "--octave-scale",
            f"{self.octave_scale:.3f}",
            "--lr",
            f"{self.lr:.4f}",
            "--size",
            str(self.size),
            "--jitter",
            str(self.jitter),
            "--out",
            self.out_dir,
        ]
        if preview_tmp_path:
            cmd += ["--preview-file", preview_tmp_path]

        separated_save = self.mode in ("Dream Zoom", "Dream Video") and bool(self.frames_dir)
        if self.mode == "Dream Zoom":
            cmd += [
                "--zoom",
                "--zoom-frames",
                str(self.zoom_frames),
                "--zoom-scale",
                f"{self.zoom_scale:.3f}",
                "--zoom-rot",
                f"{self.zoom_rot:.2f}",
                "--zoom-fps",
                str(self.zoom_fps),
                "--zoom-x",
                f"{self.zoom_cx:.4f}",
                "--zoom-y",
                f"{self.zoom_cy:.4f}",
            ]
        elif self.mode == "Dream Video":
            cmd += [
                "--video",
                "--video-blend",
                f"{self.video_blend:.2f}",
            ]
        else:
            if self.timelapse:
                cmd.append("--timelapse")

        if separated_save and self.frames_dir:
            cmd += ["--frames-dir", self.frames_dir, "--frames-only"]
        elif self.mode in ("Dream Zoom", "Dream Video") or self.timelapse:
            vq_lower = (self.video_preset or self.video_quality or "").lower()
            if vq_lower in ("small", "balanced", "lossless"):
                cmd += ["--video-quality", vq_lower]
            if self.video_crf is not None:
                cmd += ["--crf", str(self.video_crf)]

        if self.guide_path:
            cmd += ["--guide", self.guide_path]
        elif self.target_class:
            cmd += ["--class", self.target_class]

        if self.custom_name:
            cmd += ["--name", self.custom_name]
        if self.mode == "Single Image" and self.save_frames:
            cmd.append("--frames")
        if self.mode == "Single Image" and self.frames_dir and (self.save_frames or self.timelapse):
            # The frames dir is owned by this job (frames_job<id>_p<pid>), so a
            # retry may safely replace its own iteration frames. Foreign files
            # still make clear_stale_frames() refuse.
            cmd += ["--frames-dir", self.frames_dir, "--clear-frames"]
        if self.no_cuda:
            cmd.append("--no-cuda")
        if self.resume:
            cmd.append("--resume")
        if self.seed is not None:
            cmd += ["--seed", str(self.seed)]

        return cmd


class ProgressTracker:
    """Turns deepdream CLI output into (fraction, status).

    Prefers the structured events emitted by app.protocol; falls back to parsing
    log text so an older CLI (or a line with no event) still reports progress.
    """

    def __init__(self) -> None:
        self.octave = 0
        self.octaves = 0

    def update(self, line: str) -> tuple[float | None, str | None]:
        event = parse_event(line)
        if event is not None:
            result = self.update_event(event)
            if result != (None, None):
                return result
            return None, None
        return self._update_from_text(line)

    def update_event(self, event: dict) -> tuple[float | None, str | None]:
        """Handle one structured event. (None, None) means 'nothing to report'."""
        kind = event.get("event")
        current = event.get("current")
        total = event.get("total")
        if kind in (EV_OCTAVE, EV_ITERATION):
            n, m = _as_int(current), _as_int(total)
            if not n or not m:
                return None, None
            if kind == EV_OCTAVE:
                self.octave, self.octaves = n, m
                return (n - 1) / m, f"Octave {n}/{m}"
            if not self.octaves:
                return None, None
            frac = (self.octave - 1) / self.octaves + (n / m) / self.octaves
            return min(1.0, max(0.0, frac)), None
        if kind == EV_FRAME:
            if not total:
                # Frame count unknown (live video): show progress without a total.
                return (None, f"Frame {current}") if current else (None, None)
            if not current or not total:
                return None, None
            return (
                min(1.0, max(0.0, float(current) / float(total))),
                f"Frame {current}/{total}",
            )
        if kind == EV_STATUS:
            text = event.get("text")
            return None, str(text) if text else None
        return None, None

    def _update_from_text(self, line: str) -> tuple[float | None, str | None]:
        line = line.strip()
        if line.startswith("Octave "):
            try:
                n, m = map(int, line.split()[1].split("/"))
                if n <= 0 or m <= 0:
                    return None, None
            except (ValueError, IndexError):
                return None, None
            self.octave, self.octaves = n, m
            frac: float | None = (n - 1) / m
            return frac, f"Octave {n}/{m}"

        if "iteration" in line and "loss" in line:
            if not self.octaves:
                return None, None
            try:
                i, n = map(int, line.split()[1].split("/"))
                frac = (
                    (self.octave - 1) / self.octaves + (i / n) / self.octaves
                    if (self.octaves and n)
                    else None
                )
            except (ValueError, IndexError, ZeroDivisionError):
                return None, None
            return frac, None

        if line.startswith("Encoding frame "):
            try:
                parts = line.split()
                if len(parts) >= 3 and "/" in parts[2]:
                    n, m = map(int, parts[2].split("/"))
                    if n > 0 and m > 0:
                        return min(1.0, max(0.0, n / m)), f"Encoding {n}/{m}"
            except (ValueError, IndexError, ZeroDivisionError):
                return None, None

        if line.startswith("Remuxing audio"):
            return 0.98, "Remuxing audio..."

        if line.startswith("Compiling "):
            return 0.0, "Compiling..."

        if "Frame" in line:
            try:
                parts = line.split()
                if "Frame" in parts:
                    idx = parts.index("Frame") + 1
                    if idx < len(parts):
                        token = parts[idx]
                        if "/" in token:
                            n, m = map(int, token.split("/"))
                            if n > 0 and m > 0:
                                return min(1.0, n / m), f"Frame {n}/{m}"
                        elif token.isdigit():
                            return None, f"Frame {int(token)}"
            except (ValueError, IndexError, ZeroDivisionError):
                return None, None
        return None, None
