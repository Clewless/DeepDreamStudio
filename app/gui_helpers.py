from __future__ import annotations

import math
import os
import queue
import re
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .model_data import VIDEO_EXTENSIONS

# This package's directory, and the repository root that holds .venv, out/, and
# the launchers. Both are resolved from this file so they stay correct no matter
# what the current working directory is.
APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent

# Zoom bounds for the integrated viewer
ZOOM_MIN = 0.05
ZOOM_MAX = 32.0

# Viewer letterbox background RGB (matches BG_CARD #181818)
VIEWER_BG_RGB = (24, 24, 24)

# Media extensions supported in the GUI
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")
MEDIA_EXTENSIONS = IMAGE_EXTENSIONS + VIDEO_EXTENSIONS

# Subprocess completion prefix for inter-thread message queues
PROC_DONE_PREFIX = "\x00procdone:"

# Cap on log lines applied per UI tick, so a burst of output cannot starve the
# event loop. Leftovers are picked up on the next tick.
MAX_LINES_PER_TICK = 400


def venv_python() -> str:
    """Locate the virtual environment Python executable at the repository root."""
    candidates = [
        ROOT / ".venv" / "Scripts" / "python.exe",
        ROOT / ".venv" / "bin" / "python",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return sys.executable


def clamp_zoom(z: Any) -> float:
    try:
        f = float(z)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(f):
        return 1.0
    return max(ZOOM_MIN, min(ZOOM_MAX, f))


def fit_zoom_for(src_w: int, src_h: int, view_w: int, view_h: int) -> float:
    if src_w <= 0 or src_h <= 0 or view_w <= 0 or view_h <= 0:
        return 1.0
    return clamp_zoom(min(view_w / src_w, view_h / src_h))


def compute_zoom_view(
    src_w: int,
    src_h: int,
    view_w: int,
    view_h: int,
    zoom: float,
    center_x: float,
    center_y: float,
) -> dict[str, Any]:
    """Calculate crop and paste offsets for viewport rendering at a given zoom and focal point."""
    if src_w <= 0 or src_h <= 0 or view_w <= 0 or view_h <= 0:
        return {"crop": (0, 0, 1, 1), "size": (1, 1), "offset": (0, 0), "resample": "lanczos"}
    z = clamp_zoom(zoom)
    cx, cy = float(center_x), float(center_y)
    if math.isnan(cx) or math.isnan(cy):
        cx, cy = src_w / 2.0, src_h / 2.0

    vis_w, vis_h = view_w / z, view_h / z
    if vis_w >= src_w and vis_h >= src_h:
        dw = min(view_w, max(1, int(round(src_w * z))))
        dh = min(view_h, max(1, int(round(src_h * z))))
        return {
            "crop": (0, 0, src_w, src_h),
            "size": (dw, dh),
            "offset": ((view_w - dw) // 2, (view_h - dh) // 2),
            "resample": "nearest" if z > 1.0 else "lanczos",
        }

    cw = min(float(src_w), max(1.0, vis_w))
    ch = min(float(src_h), max(1.0, vis_h))
    cx = min(src_w - cw / 2.0, max(cw / 2.0, cx))
    cy = min(src_h - ch / 2.0, max(ch / 2.0, cy))
    left = max(0, min(src_w - max(1, int(round(cw))), int(round(cx - cw / 2.0))))
    top = max(0, min(src_h - max(1, int(round(ch))), int(round(cy - ch / 2.0))))
    right = min(src_w, max(left + 1, left + max(1, int(round(cw)))))
    bottom = min(src_h, max(top + 1, top + max(1, int(round(ch)))))

    scale = min(view_w / max(1, right - left), view_h / max(1, bottom - top))
    dw = max(1, min(view_w, int(round((right - left) * scale))))
    dh = max(1, min(view_h, int(round((bottom - top) * scale))))
    return {
        "crop": (left, top, right, bottom),
        "size": (dw, dh),
        "offset": ((view_w - dw) // 2, (view_h - dh) // 2),
        "resample": "nearest" if z > 1.0 else "lanczos",
    }


def render_zoomed_preview(
    src_pil: Image.Image | None,
    view_w: int,
    view_h: int,
    zoom: float,
    center_x: float,
    center_y: float,
) -> Image.Image:
    """Render a zoomed and cropped view of a PIL image into a fixed viewport size."""
    vw, vh = max(1, int(view_w)), max(1, int(view_h))
    canvas = Image.new("RGB", (vw, vh), VIEWER_BG_RGB)
    if src_pil is None:
        return canvas

    try:
        sw, sh = src_pil.size
        view = compute_zoom_view(sw, sh, vw, vh, zoom, center_x, center_y)
        left, top, right, bottom = view["crop"]
        dw, dh = view["size"]
        ox, oy = view["offset"]
        resample = (
            getattr(Image, "Resampling", Image).NEAREST
            if view["resample"] == "nearest"
            else getattr(Image, "Resampling", Image).LANCZOS
        )
        crop = src_pil.crop((left, top, right, bottom))
        if crop.size != (dw, dh):
            crop = crop.resize((dw, dh), resample)
        canvas.paste(crop, (ox, oy))
    except (OSError, ValueError):
        pass
    return canvas


def timeline_index_to_t(index: int, total: int) -> float:
    """Map a discrete frame index to a normalized [0.0, 1.0] slider position."""
    if total <= 1:
        return 0.0
    return max(0.0, min(1.0, index / (total - 1)))


def timeline_t_to_index(t: float, total: int) -> int:
    """Map a normalized slider position to an integer frame index."""
    if total <= 0 or not math.isfinite(t):
        return 0
    f = max(0.0, min(1.0, float(t)))
    return max(0, min(total - 1, int(round(f * (total - 1)))))


def format_frame_label(index: int, total: int, fps: float | None = None) -> str:
    """Format frame index and optional frame rate into a user-facing string."""
    if total <= 0:
        return "No frames"
    i = max(1, min(total, index + 1))
    if fps is not None and fps > 0:
        return f"Frame {i}/{total} @ {fps:g} fps"
    return f"Frame {i}/{total}"


def build_reveal_command(path: str | Path, platform: str | None = None) -> list[str] | None:
    """Build the OS-specific command to reveal a target file in the native file manager."""
    plat = platform if platform is not None else sys.platform
    p = os.path.abspath(os.path.normpath(str(path)))
    if not p:
        return None
    if plat == "win32":
        return ["explorer", f"/select,{p}"]
    if plat == "darwin":
        return ["open", "-R", p]
    return None


def clamp01(v: float) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.5
    if not math.isfinite(f):
        return 0.5
    return max(0.0, min(1.0, f))


def draw_zoom_focus_marker(
    img: Image.Image,
    fx: float,
    fy: float,
    marker_size: int = 11,
    color: tuple[int, int, int] = (60, 130, 250),
) -> Image.Image:
    """Overlay a focus reticle on a preview image at normalized coordinates (fx, fy)."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    cx = int(round(fx * w))
    cy = int(round(fy * h))
    r = marker_size // 2

    draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=color, width=2)
    gap = 2
    draw.line((cx - r - 3, cy, cx - gap, cy), fill=color, width=2)
    draw.line((cx + gap, cy, cx + r + 3, cy), fill=color, width=2)
    draw.line((cx, cy - r - 3, cx, cy - gap), fill=color, width=2)
    draw.line((cx, cy + gap, cx, cy + r + 3), fill=color, width=2)
    return out


def create_split_image(
    before: Image.Image,
    after: Image.Image,
    split_ratio: float,
    divider_width: int = 2,
) -> Image.Image:
    """Create a side-by-side split comparison of before and after images."""
    w, h = after.size
    split_ratio = max(0.0, min(1.0, float(split_ratio)))
    split_x = int(round(w * split_ratio))

    before_matched = (
        before if before.size == (w, h) else before.resize((w, h), Image.Resampling.BILINEAR)
    )
    composite = after.copy()

    if split_x > 0:
        left_crop = before_matched.crop((0, 0, split_x, h))
        composite.paste(left_crop, (0, 0))

    if 0 < split_x < w and divider_width > 0:
        draw = ImageDraw.Draw(composite)
        draw.line([(split_x, 0), (split_x, h)], fill=(255, 255, 255), width=divider_width)

    return composite


def slider_t_to_val(
    t: float,
    from_: float,
    to: float,
    curve: float = 1.0,
    is_int: bool = False,
    step: float | None = None,
) -> float | int:
    """Map normalized slider position t in [0.0, 1.0] to a parameter value using a power curve."""
    t_val = max(0.0, min(1.0, float(t)))
    c = max(0.01, float(curve))
    min_val = min(from_, to)
    max_val = max(from_, to)
    frac = t_val**c
    val = from_ + frac * (to - from_)
    if is_int:
        int_val = int(round(val))
        if step is not None and step > 1:
            int_val = int(from_ + int(round((int_val - from_) / step)) * step)
        return max(int(min_val), min(int(max_val), int_val))
    if step is not None and step > 0:
        val = from_ + round((val - from_) / step) * step
    return float(max(min_val, min(max_val, round(val, 4))))


def slider_val_to_t(val: float, from_: float, to: float, curve: float = 1.0) -> float:
    """Map parameter value to normalized slider position t in [0.0, 1.0] using a power curve."""
    if to == from_:
        return 0.0
    c = max(0.01, float(curve))
    frac = max(0.0, min(1.0, (float(val) - from_) / (to - from_)))
    return float(frac ** (1.0 / c))


def _fmt_slider_value(v: float, is_int: bool) -> str:
    return str(int(round(v))) if is_int else f"{v:.3f}"


def output_base_dir() -> Path:
    """Keep launcher outputs local; installed wheels store artwork in the user's home."""
    has_launcher = (ROOT / "Launch.sh").is_file() or (ROOT / "Launch.bat").is_file()
    if (ROOT / "pyproject.toml").is_file() and has_launcher:
        return ROOT
    return Path.home() / "DeepDream Studio"


def resolve_out_dir(value: str, base: str | Path | None = None) -> str:
    """Resolve outputs against the source root or an installed app's user-owned base."""
    val = value.strip().strip("\"'")
    if not val:
        anchor = base if base is not None else output_base_dir()
        return str(Path(anchor) / "out")
    if os.path.isabs(val):
        return val
    anchor = output_base_dir() if base is None else Path(base)
    _drive, tail = os.path.splitdrive(val)
    if _drive:
        if not tail or tail.strip(" \\/") == "":
            return os.path.abspath(str(anchor))
        val = tail.lstrip("\\/")
    return os.path.abspath(os.path.normpath(os.path.join(str(anchor), val)))


def parse_dropped_paths(data: Any) -> list[str]:
    """Parse dropped file paths from TkDND event data."""
    if not data:
        return []
    if isinstance(data, (list, tuple)):
        return [str(p) for p in data]
    matches = re.findall(r"\{([^}]+)\}|\"([^\"]+)\"|'([^']+)'|(\S+)", str(data).strip())
    return [next(filter(None, m)) for m in matches]


def _dialog_pattern_for_ext(ext: str) -> str:
    """Create a case-insensitive file dialog pattern for an extension."""
    if sys.platform == "win32":
        return f"*{ext.lower()}"
    stem = ext.lstrip(".")
    return "*." + "".join(f"[{c.lower()}{c.upper()}]" if c.isalpha() else c for c in stem)


def extract_output_path(line: str) -> str | None:
    """Extract output path from deepdream CLI 'Done in ... -> path' progress line."""
    if not line.strip().startswith("Done in"):
        return None
    m = re.search(r"->\s*(.+?)\s*$", line)
    if not m:
        return None
    path = m.group(1).strip()
    return path or None


def terminate_process(proc: Any, force: bool = False) -> None:
    """Terminate a running subprocess and its child tree."""
    if proc is None:
        return
    # Real Popen pids are plain ints; anything else (None, a Mock in tests,
    # a closed wrapper) means "no process tree to signal" rather than a
    # crash inside os.getpgid/str() below.
    raw_pid = getattr(proc, "pid", None)
    pid: int | None = raw_pid if isinstance(raw_pid, int) else None
    if sys.platform == "win32" and pid:
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError:
            pass
    elif sys.platform != "win32" and pid:
        try:
            import signal

            os.killpg(os.getpgid(pid), signal.SIGKILL if force else signal.SIGTERM)
        except OSError:
            pass

    try:
        if force:
            proc.kill()
        else:
            proc.terminate()
    except (OSError, AttributeError):
        pass


def _newest_matching(dir_path: str | Path, exts: tuple[str, ...]) -> str | None:
    """Find the newest file matching a tuple of extensions in dir_path."""
    out_dir = Path(dir_path)
    try:
        if not out_dir.is_dir():
            return None
        entries = list(out_dir.iterdir())
    except OSError:
        return None

    newest = None
    newest_mtime = -1.0
    for p in entries:
        name_low = p.name.lower()
        if not any(name_low.endswith(e) for e in exts):
            continue
        if name_low.startswith("_gui_preview") or name_low.endswith(".temp.mp4"):
            continue
        try:
            if not p.is_file():
                continue
            mtime = p.stat().st_mtime
        except OSError:
            continue
        if mtime > newest_mtime:
            newest_mtime = mtime
            newest = p
    return str(newest) if newest else None


def newest_png(dir_path: str | Path) -> str | None:
    return _newest_matching(dir_path, (".png",))


def newest_media(dir_path: str | Path) -> str | None:
    return _newest_matching(dir_path, (".png", ".jpg", ".jpeg", ".mp4"))


def list_saved_video_frames(frames_dir: str | Path | None) -> list[str]:
    """Return sorted list of saved video frame paths within frames_dir."""
    if not frames_dir:
        return []
    try:
        names = os.listdir(frames_dir)
    except OSError:
        return []

    numbered: list[tuple[int, str]] = []
    for name in names:
        m = re.match(r"^frame_(\d+)\.png$", name, re.IGNORECASE)
        if m:
            try:
                numbered.append((int(m.group(1)), name))
            except ValueError:
                continue
    if numbered:
        numbered.sort()
        return [os.path.join(str(frames_dir), name) for _, name in numbered]

    octave_frames: list[tuple[int, int, str]] = []
    for name in names:
        m = re.match(r"^o(\d+)_i(\d+)\.png$", name, re.IGNORECASE)
        if m:
            try:
                octave_frames.append((int(m.group(1)), int(m.group(2)), name))
            except ValueError:
                continue
    if octave_frames:
        octave_frames.sort()
        return [os.path.join(str(frames_dir), name) for _, _, name in octave_frames]

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
    return [os.path.join(str(frames_dir), n) for n in generic]


def count_saved_video_frames(frames_dir: str | Path | None) -> int:
    if not frames_dir:
        return 0
    return len(list_saved_video_frames(frames_dir))


class VideoFrameLoader:
    """Decodes video frames on a worker thread, newest request wins.

    Probing an .mp4 (frame count) and seeking to an index both block for as long
    as the decoder takes, so doing them on the Tk thread freezes the window. This
    runs one decode at a time on a daemon thread and drops stale results: while
    the user keeps scrubbing, only the frame they finally asked for is used.

    Requests and results cross threads as tuples; the caller applies them on the
    Tk thread. Nothing here touches Tk.
    """

    def __init__(self, on_result: Callable[[tuple], None], max_pending: int = 1) -> None:
        self._requests: queue.Queue = queue.Queue()
        self._results: queue.Queue = queue.Queue()
        self._max_pending = max(1, max_pending)
        self._on_result = on_result
        self._generation = 0
        # Guards _generation and _submitted. _submitted counts work that is
        # queued, in flight, or finished-but-not-yet-polled, so wait_idle() can
        # tell exactly when the worker has nothing left to do.
        self._lock = threading.Lock()
        self._submitted = 0
        self._last_kind: str = "frame"
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="video-loader", daemon=True)
        self._thread.start()

    def _bump_generation(self) -> int:
        """Invalidate in-flight results so late arrivals are ignored."""
        with self._lock:
            self._generation += 1
            return self._generation

    def generation(self) -> int:
        with self._lock:
            return self._generation

    def invalidate(self) -> None:
        """Drop queued work and ignore anything already in flight."""
        self._bump_generation()
        dropped = 0
        while True:
            try:
                self._requests.get_nowait()
                dropped += 1
            except queue.Empty:
                break
        if dropped:
            with self._lock:
                self._submitted = max(0, self._submitted - dropped)

    def _submit(self, request: tuple) -> None:
        with self._lock:
            if self._closed:
                return
            self._submitted += 1
        self._requests.put(request)

    def request_frame(self, kind: str, source: str, index: int, token: Any = None) -> None:
        """Ask for one frame. Returns immediately; the result arrives later."""
        self._submit(("frame", kind, source, int(index), token))

    def request_probe(self, path: str, token: Any = None) -> None:
        """Ask for (total_frames, fps) for an .mp4, off the Tk thread."""
        self._submit(("probe", path, token))

    def has_pending(self) -> bool:
        with self._lock:
            return self._submitted > 0

    def _trim_queue(self) -> None:
        """Drop superseded work so scrubbing does not build a backlog.

        Only same-kind requests compete: a probe is issued once per source and
        must not evict the frame request that was submitted beside it.
        """
        if self._requests.qsize() <= self._max_pending:
            return
        pending: list[tuple] = []
        while True:
            try:
                pending.append(self._requests.get_nowait())
            except queue.Empty:
                break
        # Keep the newest _max_pending of this job's kind, and every other kind
        # (probes are cheap and rare; frames are the ones that pile up).
        kind = self._last_kind
        same = [req for req in pending if req[0] == kind]
        others = [req for req in pending if req[0] != kind]
        keep = same[-self._max_pending :]
        dropped = len(same) - len(keep)
        for req in [*others, *keep]:
            self._requests.put(req)
        if dropped:
            with self._lock:
                self._submitted = max(0, self._submitted - dropped)

    def _run(self) -> None:
        while True:
            try:
                job = self._requests.get()
            except (EOFError, OSError):
                return
            if job is None:
                return
            self._last_kind = job[0]
            self._trim_queue()
            try:
                self._dispatch(job)
            except Exception as exc:  # one bad decode must not kill the worker
                self._on_result(("error", repr(exc)))
            finally:
                with self._lock:
                    self._submitted = max(0, self._submitted - 1)

    def _dispatch(self, job: tuple) -> None:
        kind = job[0]
        if kind == "frame":
            _, frame_kind, source, index, token = job
            generation = self.generation()
            frame = _decode_frame(frame_kind, source, index)
            self._results.put(("frame", frame_kind, source, index, frame, token, generation))
        elif kind == "probe":
            _, path, token = job
            generation = self.generation()
            total, fps = probe_video(path)
            self._results.put(("probe", path, total, fps, token, generation))

    def poll(self) -> bool:
        """Deliver finished results to the callback on the calling (Tk) thread.

        Returns True while work is still outstanding."""
        while True:
            try:
                result = self._results.get_nowait()
            except queue.Empty:
                break
            try:
                self._on_result(result)
            except Exception:  # never let a UI callback kill the worker
                pass
        return self.has_pending() or not self._results.empty()

    def close(self, join_timeout: float = 2.0) -> None:
        """Stop the worker. Joins briefly so an in-flight decode does not keep
        the source file open after the caller is done with it (Windows refuses
        to delete or overwrite a file that is still open)."""
        if self._closed:
            return
        self._closed = True
        try:
            self._requests.put_nowait(None)
        except queue.Empty:
            pass
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=join_timeout)

    def wait_idle(self, timeout: float = 2.0) -> bool:
        """Block until the worker has no queued or in-flight work left.

        Waiting only for the *decode* side (not for poll() to drain results)
        means a caller can stop the worker and safely reuse or delete the source
        file. Results already decoded stay queued for poll().
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._requests.empty():
                with self._lock:
                    if self._submitted == 0:
                        return True
            time.sleep(0.005)
        return False


def _decode_frame(kind: str, source: str, index: int):
    """Decode one frame to a PIL RGB image. Returns None on failure."""
    if kind == "frames":
        try:
            with Image.open(source) as raw:
                return raw.convert("RGB")
        except (OSError, ValueError):
            return None
    if kind == "mp4":
        import imageio
        import imageio.v3 as iio

        try:
            return Image.fromarray(iio.imread(source, index=int(index))).convert("RGB")
        except (ImportError, OSError, ValueError, IndexError):
            pass
        try:
            reader = imageio.get_reader(source)
            try:
                return Image.fromarray(reader.get_data(int(index))).convert("RGB")
            finally:
                try:
                    reader.close()
                except (OSError, RuntimeError, AttributeError):
                    pass
        except (ImportError, OSError, ValueError, IndexError, RuntimeError):
            return None
    return None


def probe_video(path: str) -> tuple[int, float]:
    """(total_frames, fps) for a video, or (0, 30.0). Decodes as a last resort."""
    try:
        import imageio

        reader = imageio.get_reader(path)
        try:
            try:
                total = reader.count_frames()  # type: ignore[attr-defined]
            except (ValueError, OSError, RuntimeError, AttributeError):
                total = 0
            try:
                meta = reader.get_meta_data()
                fps = float(meta.get("fps", 30.0) or 30.0)
            except (ValueError, OSError, RuntimeError, AttributeError):
                fps = 30.0
            if not total or total <= 0:
                # Bounded sequential scan: a header without a frame count
                # would otherwise have us decode an arbitrarily long file.
                try:
                    n = 0
                    for _ in reader:  # type: ignore[attr-defined]
                        n += 1
                        if n > 2000:
                            break
                    total = n
                except (ValueError, OSError, RuntimeError, AttributeError):
                    total = 0
            return int(total or 0), float(fps or 30.0)
        finally:
            try:
                reader.close()
            except (ValueError, OSError, RuntimeError, AttributeError):
                pass
    except (ImportError, OSError, ValueError, RuntimeError):
        pass
    try:
        import imageio.v3 as iio

        try:
            meta = iio.immeta(path)
            fps = float(meta.get("fps", 30.0) or 30.0)
        except (ValueError, OSError, RuntimeError, AttributeError):
            fps = 30.0
        return 0, fps
    except (ImportError, OSError, ValueError, RuntimeError):
        return 0, 30.0


class FrameCache:
    """Bounded index -> frame cache for video playback."""

    def __init__(self, max_entries: int = 12) -> None:
        self._entries: OrderedDict[int, Any] = OrderedDict()
        self._max = max(1, max_entries)

    def get(self, key):
        frame = self._entries.get(key)
        if frame is not None:
            self._entries.move_to_end(key)
        return frame

    def put(self, key, frame) -> None:
        if frame is None:
            return
        self._entries[key] = frame
        self._entries.move_to_end(key)
        while len(self._entries) > self._max:
            self._entries.popitem(last=False)

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key) -> bool:
        return key in self._entries
