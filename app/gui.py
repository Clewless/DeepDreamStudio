"""DeepDream Studio GUI: standard-Tkinter frontend for the DeepDream CLI.

Drag & drop or browse for an image, tweak parameters, watch live progress, and
preview the output without the command line. Uses Tkinter, Pillow and imageio;
it never imports torch itself, since each job runs the CLI in a subprocess.

Usage (from the repository root):
    .venv\\Scripts\\python.exe -m app.gui
"""

import os
import queue
import shutil
import subprocess
import sys
import threading
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Any, Literal, cast

import imageio
import imageio.v3 as iio
from PIL import Image, ImageOps

from . import __version__

try:
    import tkinterdnd2
    from tkinterdnd2 import DND_FILES

    _DND_AVAILABLE = True
except ImportError:
    _DND_AVAILABLE = False
    DND_FILES = "DND_Files"

from .gui_helpers import (
    IMAGE_EXTENSIONS,
    MAX_LINES_PER_TICK,
    MEDIA_EXTENSIONS,
    PROC_DONE_PREFIX,
    ROOT,
    FrameCache,
    VideoFrameLoader,
    _dialog_pattern_for_ext,
    build_reveal_command,
    clamp01,
    clamp_zoom,
    compute_zoom_view,
    count_saved_video_frames,
    create_split_image,
    draw_zoom_focus_marker,
    extract_output_path,
    fit_zoom_for,
    format_frame_label,
    list_saved_video_frames,
    newest_media,
    newest_png,
    parse_dropped_paths,
    render_zoomed_preview,
    resolve_out_dir,
    terminate_process,
    timeline_index_to_t,
    timeline_t_to_index,
    venv_python,
)
from .gui_job import (
    _CRF_TO_QUALITY,
    VIDEO_QUALITY_INFO,
    GenerationJob,
    ProgressTracker,
)
from .gui_widgets import (
    ACCENT,
    ACCENT_HOVER,
    BG_CARD,
    BG_DEEP,
    BG_PANEL,
    BORDER,
    ERROR_COL,
    MAX_LOG_LINES,
    SUCCESS,
    TEXT_DIM,
    TEXT_LABEL,
    TEXT_MAIN,
    WARNING,
    ModernButton,
    ModernCheckBox,
    ModernEntry,
    ModernFont,
    ModernFrame,
    ModernImage,
    ModernLabel,
    ModernOptionMenu,
    ModernProgressBar,
    ModernScrollableFrame,
    ModernScrollbar,
    ModernSegmentedButton,
    ModernSlider,
    ModernTextbox,
    Tooltip,
    _apply_modern_style,
    _SliderRow,
    resolve_font_family,
)
from .imagenet_categories import lookup_category, search_categories
from .model_data import (
    MAX_NAME_LEN,
    MODEL_DEFAULTS,
    MODEL_GUI_DESCRIPTIONS,
    MODEL_LAYERS,
    MODEL_ORDER,
    MODEL_RECOMMENDED,
    VIDEO_EXTENSIONS,
    build_systematic_video_name,
    filename_stem_error,
    read_video_frames_meta,
    resolve_name_conflict,
)
from .model_data import (
    MODEL_LABELS as MODEL_LABELS_MAP,
)
from .protocol import EV_DONE, EV_PREVIEW, parse_event

__all__ = ["DeepDreamGUI", "main"]

# deepdream.py runs as a subprocess, so model metadata comes from the
# torch-free model_data module instead of importing it here.

MODEL_META: dict[str, dict[str, Any]] = {
    key: {
        "label": MODEL_LABELS_MAP[key],
        "description": MODEL_GUI_DESCRIPTIONS[key],
        "layers": list(MODEL_LAYERS[key]),
        "recommended": dict(MODEL_LAYERS[key])[MODEL_RECOMMENDED[key]],
        "defaults": dict(MODEL_DEFAULTS[key]),
    }
    for key in MODEL_ORDER
}

# Categorized layer depth tiers for visual layer selection
LAYER_TIERS = {
    "googlenet": [
        (
            "Shallow Textures",
            "Fine textures, lines & ripples",
            ["inception3a", "inception3b"],
        ),
        (
            "Shapes & Objects",
            "Eyes, faces, dogs & organic shapes",
            ["inception4a", "inception4b", "inception4c", "inception4d", "inception4e"],
        ),
        (
            "Deep Abstraction",
            "Large structures & heavy hallucination",
            ["inception5a", "inception5b"],
        ),
    ],
    "inceptionv3": [
        (
            "Shallow Textures",
            "Fine patterns & soft textures",
            ["Mixed_5b", "Mixed_5c", "Mixed_5d"],
        ),
        (
            "Shapes & Objects",
            "Objects and shapes forming",
            ["Mixed_6a", "Mixed_6b", "Mixed_6c", "Mixed_6d", "Mixed_6e"],
        ),
        (
            "Deep Abstraction",
            "Bold, complex objects & structures",
            ["Mixed_7a", "Mixed_7b", "Mixed_7c"],
        ),
    ],
    "vgg16": [
        (
            "Shallow Textures",
            "Fine textures & brush patterns",
            ["features.3", "features.8"],
        ),
        (
            "Shapes & Objects",
            "Bold shapes & emerging objects",
            ["features.15", "features.22"],
        ),
        ("Deep Abstraction", "Abstract forms & heavy hallucination", ["features.29"]),
    ],
    "vgg19": [
        (
            "Shallow Textures",
            "Micro-textures & ripples",
            ["features.3", "features.8"],
        ),
        (
            "Shapes & Objects",
            "Geometric patterns & objects forming",
            ["features.17", "features.26"],
        ),
        ("Deep Abstraction", "Deep structures & abstraction", ["features.35"]),
    ],
}

RECOMMENDED_MULTI_LAYERS = {
    "googlenet": ["inception4c", "inception5b"],
    "inceptionv3": ["Mixed_6d", "Mixed_7b"],
    "vgg16": ["features.15", "features.22"],
    "vgg19": ["features.26", "features.35"],
}

# Ordered model keys for the dropdown.
MODEL_KEYS = list(MODEL_ORDER)
MODEL_LABELS = [MODEL_META[k]["label"] for k in MODEL_KEYS]
MODEL_BY_LABEL = {MODEL_META[k]["label"]: k for k in MODEL_KEYS}

DEFAULT_MODEL_KEY = "vgg16"
DEFAULT_MODEL_LABEL = MODEL_META[DEFAULT_MODEL_KEY]["label"]

PRESETS: dict[str, dict[str, Any] | None] = {
    "Select a preset...": None,
    # GoogLeNet (InceptionV1)
    "GoogLeNet Low": {
        "model": "googlenet",
        "layer": "inception3b",
        "custom_layers": "",
        "iterations": 12,
        "octaves": 0,
        "octave_scale": 1.3,
        "lr": 0.02,
        "jitter": 24,
        "size": 512,
        "desc": "Subtle textures, quick render (inception3b)",
    },
    "GoogLeNet Medium": {
        "model": "googlenet",
        "layer": "inception4c",
        "custom_layers": "",
        "iterations": 20,
        "octaves": 0,
        "octave_scale": 1.4,
        "lr": 0.03,
        "jitter": 32,
        "size": 512,
        "desc": "Classic balanced hallucination (inception4c)",
    },
    "GoogLeNet High": {
        "model": "googlenet",
        "layer": "inception5b",
        "custom_layers": "",
        "iterations": 30,
        "octaves": 0,
        "octave_scale": 1.5,
        "lr": 0.04,
        "jitter": 40,
        "size": 512,
        "desc": "High-level abstraction, slower render (inception5b)",
    },
    # InceptionV3
    "InceptionV3 Low": {
        "model": "inceptionv3",
        "layer": "Mixed_5c",
        "custom_layers": "",
        "iterations": 12,
        "octaves": 0,
        "octave_scale": 1.25,
        "lr": 0.015,
        "jitter": 24,
        "size": 512,
        "desc": "Fine early patterns, quick render (Mixed_5c)",
    },
    "InceptionV3 Medium": {
        "model": "inceptionv3",
        "layer": "Mixed_6d",
        "custom_layers": "",
        "iterations": 24,
        "octaves": 0,
        "octave_scale": 1.35,
        "lr": 0.025,
        "jitter": 32,
        "size": 512,
        "desc": "Balanced complex patterns (Mixed_6d)",
    },
    "InceptionV3 High": {
        "model": "inceptionv3",
        "layer": "Mixed_7c",
        "custom_layers": "",
        "iterations": 36,
        "octaves": 0,
        "octave_scale": 1.4,
        "lr": 0.035,
        "jitter": 40,
        "size": 512,
        "desc": "High-level abstraction, slower render (Mixed_7c)",
    },
    # VGG16
    "VGG16 Low": {
        "model": "vgg16",
        "layer": "features.8",
        "custom_layers": "",
        "iterations": 12,
        "octaves": 0,
        "octave_scale": 1.25,
        "lr": 0.012,
        "jitter": 24,
        "size": 512,
        "desc": "Soft patterns, quick render (Block 2)",
    },
    "VGG16 Medium": {
        "model": "vgg16",
        "layer": "features.22",
        "custom_layers": "",
        "iterations": 20,
        "octaves": 0,
        "octave_scale": 1.32,
        "lr": 0.018,
        "jitter": 32,
        "size": 512,
        "desc": "Balanced objects forming (Block 4)",
    },
    "VGG16 High": {
        "model": "vgg16",
        "layer": "features.29",
        "custom_layers": "",
        "iterations": 30,
        "octaves": 0,
        "octave_scale": 1.4,
        "lr": 0.024,
        "jitter": 40,
        "size": 512,
        "desc": "Deep hallucination, slower render (Block 5)",
    },
    # VGG19
    "VGG19 Low": {
        "model": "vgg19",
        "layer": "features.8",
        "custom_layers": "",
        "iterations": 14,
        "octaves": 0,
        "octave_scale": 1.25,
        "lr": 0.012,
        "jitter": 24,
        "size": 512,
        "desc": "Soft patterns, quick render (Block 2)",
    },
    "VGG19 Medium": {
        "model": "vgg19",
        "layer": "features.26",
        "custom_layers": "",
        "iterations": 25,
        "octaves": 0,
        "octave_scale": 1.32,
        "lr": 0.018,
        "jitter": 32,
        "size": 512,
        "desc": "Balanced detailed texture (Block 4)",
    },
    "VGG19 High": {
        "model": "vgg19",
        "layer": "features.35",
        "custom_layers": "",
        "iterations": 34,
        "octaves": 0,
        "octave_scale": 1.4,
        "lr": 0.024,
        "jitter": 40,
        "size": 512,
        "desc": "Deep structures, slower render (Block 5)",
    },
}


def presets_for_model(model_key):
    """Return list of preset names applicable to the given model."""
    return ["Select a preset..."] + [
        name for name, p in PRESETS.items() if p and p.get("model") == model_key
    ]


def _layer_display_list(model_key):
    """Return the list of display labels for layers in the given model."""
    return [label for _, label in MODEL_META[model_key]["layers"]]


def _layer_value_map(model_key):
    """Return a {display_label: layer_key} dict for the given model."""
    return {label: key for key, label in MODEL_META[model_key]["layers"]}


# Main window


class DeepDreamGUI(tk.Tk):
    def __init__(self, initial_image=None):
        super().__init__()

        # UI widget handles populated during _build_ui
        self._zoom_focus_label: Any = None
        self._status_label: Any = None
        self._preset_menu: Any = None
        self._video_quality_seg: Any = None
        self._video_quality_info_lbl: Any = None
        self._video_quality_frame: Any = None
        self._class_selected_label: Any = None
        self._class_dropdown: Any = None
        self._class_listbox: Any = None
        self._layer_chips_container: Any = None
        self._layer_mode_seg: Any = None
        self._single_layer_frame: Any = None
        self._multi_layer_frame: Any = None
        self._layer_badge_count: Any = None
        self._main_resume_btn: Any = None
        self._save_video_btn: Any = None
        self._sidebar_save_video_btn: Any = None
        self._zoom_bar: Any = None
        self._zoom_label: Any = None
        self._video_bar: Any = None
        self._video_pos_label: Any = None
        self._video_slider: Any = None
        self._video_play_btn: Any = None
        self._compare_seg: Any = None
        self._queue_scroll: Any = None
        self._queue_inner: Any = None
        self._video_frames_label: Any = None

        self.title(f"DeepDream Studio {__version__}")
        self.geometry("1180x780")
        self.minsize(900, 640)
        _apply_modern_style(self)
        self.configure(bg=BG_DEEP)

        self._image_path = tk.StringVar()
        self._out_dir = tk.StringVar(value=str(ROOT / "out"))
        self._model_var = tk.StringVar(value=DEFAULT_MODEL_LABEL)
        self._layer_var = tk.StringVar(value=MODEL_META[DEFAULT_MODEL_KEY]["recommended"])
        _defs = MODEL_META[DEFAULT_MODEL_KEY]["defaults"]
        self._iterations = tk.IntVar(value=_defs["iterations"])
        self._octaves = tk.IntVar(value=_defs["octaves"])
        self._octave_scale = tk.DoubleVar(value=_defs["octave_scale"])
        self._lr = tk.DoubleVar(value=_defs["lr"])
        self._size = tk.IntVar(value=_defs["size"])
        self._jitter = tk.IntVar(value=_defs["jitter"])
        self._save_frames = tk.BooleanVar(value=False)
        self._no_cuda = tk.BooleanVar(value=False)
        self._custom_name = tk.StringVar()

        # Mode & presets
        self._mode_var = tk.StringVar(value="Single Image")
        self._previous_mode = "Single Image"
        self._saved_single_image_iters = self._iterations.get()
        self._preset_var = tk.StringVar(value="Select a preset...")

        # Guidance
        self._guidance_mode_var = tk.StringVar(value="Unguided")
        self._guide_path = tk.StringVar(value="")
        self._target_class_var = tk.StringVar(value="")
        self._target_class_var.trace_add("write", self._on_target_class_var_changed)
        self._class_search_var = tk.StringVar(value="")
        self._class_search_var.trace_add("write", self._on_class_search_changed)
        self._class_filter_name = "Popular"

        # Multi-layer
        self._layer_mode_var = tk.StringVar(value="Single Layer")
        self._custom_layers_var = tk.StringVar(value="")
        self._custom_layers_var.trace_add("write", self._on_custom_layers_var_changed)

        # Video & animation options
        self._timelapse = tk.BooleanVar(value=False)
        self._zoom_frames = tk.IntVar(value=60)
        self._zoom_scale = tk.DoubleVar(value=1.03)
        self._zoom_rot = tk.DoubleVar(value=0.0)
        self._zoom_fps = tk.IntVar(value=30)
        self._zoom_cx = tk.DoubleVar(value=0.5)
        self._zoom_cy = tk.DoubleVar(value=0.5)
        self._video_blend = tk.DoubleVar(value=0.5)
        self._video_quality_var = tk.StringVar(value="Balanced")
        self._video_crf = tk.IntVar(value=17)

        # Before / After comparison state
        self._input_pil = None
        self._input_pil_path = None
        self._input_pil_mtime = None
        self._dream_pil = None
        self._aligned_input_pil = None
        self._compare_mode = "Dream"  # "Dream", "Slide Reveal", or "Original"
        self._reveal_split = 0.5
        self._reveal_dragging = False
        self._is_video_output = False
        # True while the main preview is showing the input image with the
        # zoom-focus reticle (clicks then aim the zoom instead of opening
        # the output). Any live/dream/video preview flips it back to False.
        self._showing_zoom_focus = False
        self._slider_registry = []
        self._active_hover_nudge = None
        self._preview_tmp_path = None
        # Integrated viewer state (zoom/pan + in-window video)
        # _viewer_zoom is absolute (1.0 == 100%); _viewer_fit True means the
        # image is auto-fitted to the preview area. _viewer_cx/cy is the focal
        # point in *source* pixels shown at the viewport center.
        self._viewer_zoom = 1.0
        self._viewer_fit = True
        self._viewer_cx = None
        self._viewer_cy = None
        self._viewer_drag = None  # (start_x, start_y, cx0, cy0, moved)
        self._video_kind = None  # None | "frames" | "mp4"
        self._video_paths: list[str] = []
        self._video_mp4: str | None = None
        self._video_index = 0
        self._video_total = 0
        self._video_fps = 30.0
        self._video_playing = False
        self._video_after_id = None
        self._video_slider_sync = False
        self._video_cache: FrameCache = FrameCache()
        # Decodes and frame counts run on a worker thread so a long mp4 cannot
        # freeze the window; results are applied by _poll_video_loader.
        self._video_loader: VideoFrameLoader | None = None
        self._video_poll_after_id = None
        self._video_requested_index = 0
        self._video_probe_token = 0
        self._video_pending_title = None

        def _on_video_result(result):
            if result and result[0] == "frame":
                self._on_video_frame_result(*result[1:])
            elif result and result[0] == "probe":
                self._on_video_probe_result(*result[1:])

        self._video_loader = VideoFrameLoader(_on_video_result)
        self._timelapse_mp4: str | None = None
        self._viewer_mode = "image"  # "image" | "video" | "timelapse"

        self._is_closing = False
        self._reported_callback_errors: set = set()
        self._proc = None
        self._encode_proc = None
        self._poll_after_id = None
        self._next_job_after_id = None
        self._log_queue: queue.Queue[str] = queue.Queue()
        # Encode-subprocess events (log lines and completion) as tuples, drained
        # on the Tk thread by _poll_log so workers never touch Tk.
        self._encode_queue: queue.Queue[tuple] = queue.Queue()
        self._output_path = None  # last successfully saved image
        self._preview_pil = None  # PIL image for the input preview
        self._tracker = ProgressTracker()
        self._expected_output = None  # path from the "Done in ... ->" log line
        self._aborted = False  # True when the user pressed Stop
        self._aborted_jobs: set[int] = set()
        self._frames_dir = None  # frames/ dir for the current run
        self._frames_preexisting = False
        # Separated video-save state: the most recent dreamed frame sequence
        # awaiting export. It stays re-exportable until its frames are removed.
        self._video_frames_dir = None
        self._video_frames_count = 0
        self._video_frames_fps = 30.0
        self._video_frames_mode = None
        self._video_frames_source = None
        self._video_frames_job_id = None
        self._video_frames_job = None
        self._video_save_in_progress = False
        self._video_export_target = None

        # Generation Queue state
        self._job_queue = []
        self._current_job = None
        self._job_counter = 0
        self._queue_running = False
        self._resumable_job = None

        # Additional UI and viewer state
        self._video_last_save = None
        self._video_last_crf = None
        self._last_render_view_size = None
        self._class_listbox_height = 8
        self._class_listbox_min_height = 3
        self._class_listbox_max_height = 20
        self._class_list_resize_state = None
        # Load TkDND into this window's tcl interpreter BEFORE building the UI
        global _DND_AVAILABLE
        if _DND_AVAILABLE:
            try:
                tkinterdnd2.TkinterDnD._require(self)
            except (tk.TclError, RuntimeError, AttributeError):
                _DND_AVAILABLE = False  # fall back to click-only gracefully

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build_ui()

        self.bind("<Left>", lambda e: self._on_arrow_nudge(-1, e))
        self.bind("<Down>", lambda e: self._on_arrow_nudge(-1, e))
        self.bind("<Right>", lambda e: self._on_arrow_nudge(1, e))
        self.bind("<Up>", lambda e: self._on_arrow_nudge(1, e))
        self.bind("<Leave>", self._on_window_leave, add="+")

        if initial_image and os.path.exists(initial_image):
            self.after_idle(lambda: self._load_image_preview(initial_image))

    def _on_close(self):
        """Cleanly terminate running subprocess when window is closed."""
        self._is_closing = True
        self._stop_video_playback()
        loader = getattr(self, "_video_loader", None)
        if loader is not None:
            loader.close()
        for after_id in (self._poll_after_id, self._video_poll_after_id):
            if after_id is not None:
                try:
                    self.after_cancel(after_id)
                except tk.TclError:
                    pass
        self._poll_after_id = None
        self._video_poll_after_id = None
        self._cancel_next_job_timer()

        encode_proc = self._encode_proc
        active_child = (self._proc and self._proc.poll() is None) or (
            encode_proc and encode_proc.poll() is None
        )
        if active_child:
            # Withdraw window immediately so the GUI disappears without delay
            # while the child processes and temporary files are reaped.
            try:
                self.withdraw()
                self.update_idletasks()
            except tk.TclError:
                pass

            self._aborted = True
            job = self._current_job
            if job is not None and getattr(job, "id", None) is not None:
                self._aborted_jobs.add(job.id)
            if self._proc and self._proc.poll() is None:
                terminate_process(self._proc)
            if encode_proc and encode_proc.poll() is None:
                terminate_process(encode_proc)

            # Reaping: let a daemon thread wait on the children and sweep temps.
            # Perform a bounded join so normal terminations clean up reliably
            # before Python teardown destroys daemon threads.
            proc_ref = self._proc
            encode_proc_ref = encode_proc
            job_ref = job
            pid_ref = getattr(proc_ref, "pid", None)
            encode_pid = getattr(encode_proc_ref, "pid", None)

            def _reap():
                for p in (proc_ref, encode_proc_ref):
                    if p is not None:
                        try:
                            p.wait(timeout=5.0)
                        except subprocess.TimeoutExpired:
                            terminate_process(p, force=True)
                            try:
                                p.wait(timeout=2.0)
                            except subprocess.TimeoutExpired:
                                pass
                # Sweep only after the child is reaped, otherwise it can
                # recreate a partial frame/temp after the cleanup runs.
                # _cleanup_partial_frames() (and its _log_msg calls)
                # short-circuits on _is_closing, so this is safe after destroy().
                self._cleanup_partial_frames()
                if job_ref is not None:
                    self._cleanup_job_temps(job_ref.out_dir, pid_ref)
                vfd = self._video_frames_dir
                if encode_pid is not None and vfd:
                    self._cleanup_job_temps(os.path.dirname(vfd), encode_pid)

            t = threading.Thread(target=_reap, daemon=True)
            t.start()
            t.join(timeout=1.0)
        self._cleanup_preview_temps()
        for row in self._slider_registry:
            row.dispose()
        self.destroy()

    def report_callback_exception(self, exc, val, tb):
        """Surface unhandled Tk callback errors to the console pane and a dialog.
        The launcher starts pythonw.exe (no console), so Tkinter's default stderr
        handler is invisible."""
        if self._is_closing:
            return
        err_text = "".join(traceback.format_exception(exc, val, tb))
        self._log_msg(f"\n[Unexpected error] {val}\n{err_text}\n", color=ERROR_COL)
        if self._status_label is not None:
            try:
                self._status_label.configure(text="Error", text_color=ERROR_COL)
            except tk.TclError:
                pass
        key = (exc, str(val))
        if key in self._reported_callback_errors:
            return
        self._reported_callback_errors.add(key)
        try:
            messagebox.showerror("DeepDream Studio - Unexpected Error", err_text)
        except tk.TclError:
            pass

    # UI construction

    def _build_ui(self):
        try:
            self.unbind_class("TCombobox", "<MouseWheel>")
            self.unbind_class("TCombobox", "<Button-4>")
            self.unbind_class("TCombobox", "<Button-5>")
            self.unbind_class("TSpinbox", "<MouseWheel>")
            self.unbind_class("TSpinbox", "<Button-4>")
            self.unbind_class("TSpinbox", "<Button-5>")
        except tk.TclError:
            pass
        self.grid_columnconfigure(0, weight=0, minsize=340)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_main()

    # Sidebar

    def _build_sidebar(self):
        sb_outer = ModernScrollableFrame(
            self,
            width=340,
            fg_color=BG_PANEL,
            scrollbar_button_color=BG_CARD,
            scrollbar_button_hover_color=BORDER,
        )
        sb_outer.grid(row=0, column=0, sticky="nsew")
        sb_outer.grid_columnconfigure(0, weight=1)
        sb_outer.inner.grid_columnconfigure(0, weight=1)
        self._sidebar_frame = sb_outer
        # Content goes into the scrollable inner frame so it actually scrolls.
        sb = sb_outer.inner

        self._build_sidebar_header(sb)
        self._build_sidebar_mode(sb)
        self._build_sidebar_presets(sb)
        self._build_sidebar_media(sb)
        self._build_sidebar_guidance(sb)
        self._build_sidebar_model(sb)
        self._build_sidebar_layers(sb)
        self._build_sidebar_parameters(sb)
        self._build_sidebar_dynamic_parameters(sb)

    def _build_sidebar_header(self, sb):
        # Header
        ModernLabel(
            sb,
            text="DeepDream Studio",
            font=ModernFont(family="Segoe UI", size=11, weight="bold"),
            text_color=TEXT_DIM,
            anchor="w",
        ).grid(row=0, column=0, sticky="w", padx=20, pady=(12, 4))

    def _build_sidebar_mode(self, sb):
        # Mode Selector
        self._section(sb, row=1, label="Mode")
        self._mode_seg = ModernSegmentedButton(
            sb,
            values=["Single Image", "Dream Zoom", "Dream Video"],
            variable=self._mode_var,
            selected_color=BORDER,
            selected_hover_color=ACCENT,
            font=ModernFont(family="Segoe UI", size=11),
            command=self._on_mode_change,
        )
        self._mode_seg.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 8))
        Tooltip(
            self._mode_seg,
            "Workflow mode:\n"
            "• Single Image: Multi-scale dream on a still photo.\n"
            "• Dream Zoom: From a still image, creates a tunnel-dive dream animation.\n"
            "• Dream Video: From a video file, stylizes each frame with dream blending.",
        )

    def _build_sidebar_presets(self, sb):
        # Presets
        self._section(sb, row=3, label="Preset")
        self._preset_menu = ModernOptionMenu(
            sb,
            variable=self._preset_var,
            values=presets_for_model(DEFAULT_MODEL_KEY),
            fg_color=BG_CARD,
            button_color=ACCENT,
            button_hover_color=ACCENT_HOVER,
            dropdown_fg_color=BG_CARD,
            dropdown_hover_color=BORDER,
            font=ModernFont(family="Segoe UI", size=11),
            text_color=TEXT_MAIN,
            height=32,
            command=self._on_preset_change,
        )
        self._preset_menu.grid(row=4, column=0, sticky="ew", padx=16, pady=(0, 8))
        Tooltip(
            self._preset_menu,
            "Preset parameters for the active model. "
            "Selecting one overwrites iterations, octaves, scale, learning rate, jitter, and size.",
        )

    def _build_sidebar_media(self, sb):
        # Input Image / Video
        self._section(sb, row=5, label="Input Media")
        img_frame = ModernFrame(sb, fg_color=BG_CARD)
        img_frame.grid(row=6, column=0, sticky="ew", padx=16, pady=(0, 6))
        img_frame.grid_columnconfigure(0, weight=1)

        self._drop_zone = ModernLabel(
            img_frame,
            text="Click to browse\nor drop an image here",
            font=ModernFont(family="Segoe UI", size=12),
            text_color=TEXT_DIM,
            height=110,
            fg_color=BG_PANEL,
            cursor="hand2",
        )
        self._drop_zone.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        self._drop_zone.bind("<Button-1>", lambda e: self._browse_image())
        self._drop_zone.bind("<Enter>", lambda e: self._drop_zone.configure(fg_color=BORDER))
        self._drop_zone.bind("<Leave>", lambda e: self._drop_zone.configure(fg_color=BG_PANEL))
        if _DND_AVAILABLE:
            try:
                self._drop_zone.drop_target_register(DND_FILES)
                self._drop_zone.dnd_bind("<<Drop>>", self._on_drop)
                self._drop_zone.dnd_bind(
                    "<<DragEnter>>", lambda e: self._drop_zone.configure(fg_color=BORDER)
                )
                self._drop_zone.dnd_bind(
                    "<<DragLeave>>", lambda e: self._drop_zone.configure(fg_color=BG_PANEL)
                )
            except (tk.TclError, RuntimeError, AttributeError):
                # tkdnd loaded but registration failed: click-to-browse still
                # works, so don't abort GUI startup over drag & drop.
                pass
        Tooltip(
            self._drop_zone,
            "Click to browse or drop an image/video file. "
            "Multiple dropped images are queued as separate jobs.",
        )

        self._img_path_label = ModernLabel(
            img_frame,
            textvariable=self._image_path,
            font=ModernFont(family="Segoe UI", size=10),
            text_color=TEXT_DIM,
            wraplength=290,
            anchor="w",
        )
        self._img_path_label.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 8))

    def _build_sidebar_guidance(self, sb):
        # Guidance
        self._section(sb, row=7, label="Guidance")
        self._guidance_seg = ModernSegmentedButton(
            sb,
            values=["Unguided", "Guide Image", "Target Class"],
            variable=self._guidance_mode_var,
            selected_color=BORDER,
            selected_hover_color=ACCENT,
            font=ModernFont(family="Segoe UI", size=11),
            command=self._on_guidance_change,
        )
        self._guidance_seg.grid(row=8, column=0, sticky="ew", padx=16, pady=(0, 6))
        Tooltip(
            self._guidance_seg,
            "Dream guidance:\n"
            "• Unguided: Amplifies detected patterns without an external reference.\n"
            "• Guide Image: Matches textures and motifs from a reference image.\n"
            "• Target Class: Steers the output toward a specific object (e.g. flamingo, pizza).",
        )

        # Guide Image container
        self._guide_frame = ModernFrame(sb, fg_color=BG_CARD)
        self._guide_frame.grid(row=9, column=0, sticky="ew", padx=16, pady=(0, 8))
        self._guide_frame.grid_columnconfigure(0, weight=1)

        self._guide_drop_zone = ModernLabel(
            self._guide_frame,
            text="Click to select Guide Image\n(transfers styles & motifs)",
            font=ModernFont(family="Segoe UI", size=11),
            text_color=TEXT_DIM,
            height=70,
            fg_color=BG_PANEL,
            cursor="hand2",
        )
        self._guide_drop_zone.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 4))
        self._guide_drop_zone.bind("<Button-1>", lambda e: self._browse_guide_image())
        self._guide_drop_zone.bind(
            "<Enter>", lambda e: self._guide_drop_zone.configure(fg_color=BORDER)
        )
        self._guide_drop_zone.bind(
            "<Leave>", lambda e: self._guide_drop_zone.configure(fg_color=BG_PANEL)
        )
        Tooltip(
            self._guide_drop_zone,
            "Reference image. Activations from this image are used as the optimization target.",
        )

        self._guide_path_label = ModernLabel(
            self._guide_frame,
            textvariable=self._guide_path,
            font=ModernFont(family="Segoe UI", size=9),
            text_color=TEXT_DIM,
            wraplength=290,
            anchor="w",
        )
        self._guide_path_label.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 6))
        self._guide_frame.grid_remove()  # Hidden initially

        # Target Class container
        self._class_frame = ModernFrame(sb, fg_color=BG_CARD)
        self._class_frame.grid(row=9, column=0, sticky="ew", padx=16, pady=(0, 8))
        self._class_frame.grid_columnconfigure(0, weight=1)

        # Active Target Class Badge / Status
        self._class_status_card = ModernFrame(self._class_frame, fg_color=BG_PANEL)
        self._class_status_card.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 4))
        self._class_status_card.grid_columnconfigure(0, weight=1)

        self._class_selected_label = ModernLabel(
            self._class_status_card,
            text="No target selected (search or click below)",
            font=ModernFont(family="Segoe UI", size=10, weight="bold"),
            text_color=TEXT_LABEL,
            anchor="w",
        )
        self._class_selected_label.grid(row=0, column=0, sticky="w", padx=8, pady=4)

        self._class_clear_btn = ModernButton(
            self._class_status_card,
            text="✕ Clear",
            width=50,
            height=22,
            fg_color=BG_CARD,
            hover_color=BORDER,
            text_color=TEXT_DIM,
            font=ModernFont(family="Segoe UI", size=9),
            command=self._clear_target_class,
        )
        self._class_clear_btn.grid(row=0, column=1, padx=6, pady=4)
        Tooltip(self._class_clear_btn, "Clear current target class.")

        # Search Box with Clear Button
        search_box_frame = ModernFrame(self._class_frame, fg_color="transparent")
        search_box_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 4))
        search_box_frame.grid_columnconfigure(0, weight=1)

        self._class_search_entry = ModernEntry(
            search_box_frame,
            textvariable=self._class_search_var,
            placeholder_text="Search category or index (e.g. dog, 130)...",
            fg_color=BG_PANEL,
            border_color=BORDER,
            text_color=TEXT_MAIN,
            font=ModernFont(family="Segoe UI", size=10),
            height=28,
        )
        self._class_search_entry.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self._class_search_entry.bind("<Return>", self._on_class_search_enter)
        Tooltip(
            self._class_search_entry,
            "Type any category name or numeric index (0-999) to filter the dropdown in real time.",
        )

        clear_search_btn = ModernButton(
            search_box_frame,
            text="✕",
            width=24,
            height=28,
            fg_color=BG_PANEL,
            hover_color=BORDER,
            text_color=TEXT_DIM,
            font=ModernFont(family="Segoe UI", size=9, weight="bold"),
            command=self._clear_class_search,
        )
        clear_search_btn.grid(row=0, column=1)
        Tooltip(clear_search_btn, "Clear search text.")

        # Mouse-only Quick Category Filters
        self._class_filter_seg = ModernSegmentedButton(
            self._class_frame,
            values=["Popular", "Animals", "Food", "Vehicles", "All"],
            command=self._on_class_category_filter_change,
            selected_color=BORDER,
            selected_hover_color=ACCENT,
            font=ModernFont(family="Segoe UI", size=9),
            height=24,
        )
        self._class_filter_seg.set("Popular")
        self._class_filter_seg.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 4))
        Tooltip(
            self._class_filter_seg, "Click to filter categories by theme with only mouse input."
        )

        # Initial popular choices for dropdown
        initial_matches = search_categories("", category_filter="★ Popular")
        init_values = [m[2] for m in initial_matches] or ["#130: flamingo"]

        # Dropdown Menu
        self._class_dropdown = ModernOptionMenu(
            self._class_frame,
            values=init_values,
            command=self._on_class_dropdown_select,
            fg_color=BG_PANEL,
            button_color=ACCENT,
            button_hover_color=ACCENT_HOVER,
            dropdown_fg_color=BG_CARD,
            dropdown_hover_color=BORDER,
            font=ModernFont(family="Segoe UI", size=11),
            text_color=TEXT_MAIN,
            height=30,
        )
        self._class_dropdown.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 4))
        Tooltip(
            self._class_dropdown,
            "Dropdown menu: click with mouse to browse and select ImageNet categories.",
        )

        # Integrated Dropdown Listbox (scrollable with mouse wheel, 1-click select)
        # Height is 8 rows by default; the grip below resizes it (3-20 rows).
        list_container = ModernFrame(self._class_frame, fg_color=BG_PANEL)
        list_container.grid(row=4, column=0, sticky="ew", padx=10, pady=(0, 2))
        list_container.grid_columnconfigure(0, weight=1)

        self._class_listbox_height = 8
        self._class_listbox_min_height = 3
        self._class_listbox_max_height = 20
        self._class_list_resize_state = None
        self._class_listbox = tk.Listbox(
            list_container,
            bg=BG_PANEL,
            fg=TEXT_MAIN,
            selectbackground=ACCENT,
            selectforeground=BG_DEEP,
            activestyle="none",
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            font=(resolve_font_family("Segoe UI"), 9),
            height=self._class_listbox_height,
        )
        self._class_listbox.grid(row=0, column=0, sticky="nsew", padx=(6, 0), pady=4)

        lb_scroll = ModernScrollbar(list_container, command=self._class_listbox.yview, width=12)
        lb_scroll.grid(row=0, column=1, sticky="ns", padx=(0, 4), pady=4)
        self._class_listbox.configure(yscrollcommand=lb_scroll.set)

        self._class_listbox.bind("<<ListboxSelect>>", self._on_class_listbox_select)
        self._class_listbox.bind("<ButtonRelease-1>", self._on_class_listbox_select)
        Tooltip(
            list_container,
            "Scrollable category box: click any item with mouse to select as target.",
        )

        # Drag-to-resize grip for the listbox above
        self._class_list_resize_grip = ModernLabel(
            self._class_frame,
            text="⋯  ⇕ drag to resize list  ⋯",
            font=ModernFont(family="Segoe UI", size=8),
            text_color=TEXT_DIM,
            fg_color=BG_PANEL,
            cursor="sb_v_double_arrow",
            anchor="center",
        )
        self._class_list_resize_grip.grid(row=5, column=0, sticky="ew", padx=10, pady=(0, 8))
        Tooltip(
            self._class_list_resize_grip,
            "Click and drag up/down to resize the category list to fit your needs.",
        )
        self._class_list_resize_grip.bind("<Button-1>", self._on_class_list_resize_start)
        self._class_list_resize_grip.bind("<B1-Motion>", self._on_class_list_resize_drag)
        self._class_list_resize_grip.bind("<ButtonRelease-1>", self._on_class_list_resize_end)

        # Populate initial listbox items
        for v in init_values:
            self._class_listbox.insert(tk.END, f"  {v}")

        self._class_frame.grid_remove()  # Hidden initially

    def _build_sidebar_model(self, sb):
        # Model selector
        self._section(sb, row=10, label="Model")
        model_card = ModernFrame(sb, fg_color=BG_CARD)
        model_card.grid(row=11, column=0, sticky="ew", padx=16, pady=(0, 4))
        model_card.grid_columnconfigure(0, weight=1)

        self._model_menu = ModernOptionMenu(
            model_card,
            variable=self._model_var,
            values=MODEL_LABELS,
            fg_color=BG_PANEL,
            button_color=ACCENT,
            button_hover_color=ACCENT_HOVER,
            dropdown_fg_color=BG_CARD,
            dropdown_hover_color=BORDER,
            font=ModernFont(family="Segoe UI", size=12),
            text_color=TEXT_MAIN,
            height=36,
            command=self._on_model_change,
        )
        self._model_menu.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 4))
        Tooltip(
            self._model_menu,
            "Neural network backbone:\n"
            "• GoogLeNet: The 2015 original (dogs, eyes, faces).\n"
            "• InceptionV3: Deeper; fine textures and geometry.\n"
            "• VGG16: Strong object shapes and textures.\n"
            "• VGG19: Deeper still; detailed textures.",
        )

        self._model_desc_label = ModernLabel(
            model_card,
            text=MODEL_META[DEFAULT_MODEL_KEY]["description"],
            font=ModernFont(family="Segoe UI", size=10),
            text_color=TEXT_DIM,
            anchor="w",
            wraplength=280,
        )
        self._model_desc_label.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 10))

    def _build_sidebar_layers(self, sb):
        # Layer selector
        self._section(sb, row=12, label="Dream Layer")
        layer_card = ModernFrame(sb, fg_color=BG_CARD)
        layer_card.grid(row=13, column=0, sticky="ew", padx=16, pady=(0, 8))
        layer_card.grid_columnconfigure(0, weight=1)

        # Mode toggle: Single Layer vs Multi-Layer
        self._layer_mode_seg = ModernSegmentedButton(
            layer_card,
            values=["Single Layer", "Multi-Layer"],
            variable=self._layer_mode_var,
            selected_color=BORDER,
            selected_hover_color=ACCENT,
            font=ModernFont(family="Segoe UI", size=11),
            command=self._on_layer_mode_change,
        )
        self._layer_mode_seg.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 6))
        Tooltip(
            self._layer_mode_seg,
            "Layer mode:\n"
            "• Single Layer: Focuses all dream iterations on one specific network depth.\n"
            "• Multi-Layer: Combines and balances features across multiple layers at once.",
        )

        # Single Layer container
        self._single_layer_frame = ModernFrame(layer_card, fg_color="transparent")
        self._single_layer_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 8))
        self._single_layer_frame.grid_columnconfigure(0, weight=1)

        self._layer_menu = ModernOptionMenu(
            self._single_layer_frame,
            variable=self._layer_var,
            values=_layer_display_list(DEFAULT_MODEL_KEY),
            command=self._on_single_layer_select,
            fg_color=BG_PANEL,
            button_color=ACCENT,
            button_hover_color=ACCENT_HOVER,
            dropdown_fg_color=BG_CARD,
            dropdown_hover_color=BORDER,
            font=ModernFont(family="Segoe UI", size=12),
            text_color=TEXT_MAIN,
            height=34,
        )
        self._layer_menu.grid(row=0, column=0, sticky="ew")
        Tooltip(
            self._layer_menu,
            "Which layer to optimize:\n"
            "• Early layers: Fine lines, ripples, textures.\n"
            "• Middle layers: Geometric forms, eyes, shapes.\n"
            "• Deep layers: Large high-level structures.",
        )

        # Multi-Layer container
        self._multi_layer_frame = ModernFrame(layer_card, fg_color="transparent")
        self._multi_layer_frame.grid_columnconfigure(0, weight=1)

        ml_header = ModernFrame(self._multi_layer_frame, fg_color="transparent")
        ml_header.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        ml_header.grid_columnconfigure(0, weight=1)

        self._layer_badge_count = ModernLabel(
            ml_header,
            text="Active (0)",
            font=ModernFont(family="Segoe UI", size=11, weight="bold"),
            text_color=TEXT_LABEL,
        )
        self._layer_badge_count.grid(row=0, column=0, sticky="w")

        ml_rec_btn = ModernButton(
            ml_header,
            text="★ Rec",
            width=50,
            height=24,
            fg_color=BG_PANEL,
            hover_color=BORDER,
            text_color=TEXT_MAIN,
            font=ModernFont(family="Segoe UI", size=10, weight="bold"),
            command=self._select_recommended_layers,
        )
        ml_rec_btn.grid(row=0, column=1, padx=(0, 4))
        Tooltip(ml_rec_btn, "Activate recommended layer combination for this model.")

        ml_all_btn = ModernButton(
            ml_header,
            text="All",
            width=36,
            height=24,
            fg_color=BG_PANEL,
            hover_color=BORDER,
            text_color=TEXT_MAIN,
            font=ModernFont(family="Segoe UI", size=10),
            command=self._select_all_layers,
        )
        ml_all_btn.grid(row=0, column=2, padx=(0, 4))
        Tooltip(ml_all_btn, "Activate all available layers.")

        ml_clear_btn = ModernButton(
            ml_header,
            text="Clear",
            width=44,
            height=24,
            fg_color=BG_PANEL,
            hover_color=BORDER,
            text_color=TEXT_DIM,
            font=ModernFont(family="Segoe UI", size=10),
            command=self._clear_multi_layers,
        )
        ml_clear_btn.grid(row=0, column=3)
        Tooltip(ml_clear_btn, "Deactivate all layers.")

        # Interactive chips container
        self._layer_chips_container = ModernFrame(
            self._multi_layer_frame,
            fg_color=BG_PANEL,
        )
        self._layer_chips_container.grid(row=1, column=0, sticky="ew", pady=(0, 6))

        # Direct text entry for custom layer names.
        self._custom_layers_entry = ModernEntry(
            self._multi_layer_frame,
            textvariable=self._custom_layers_var,
            placeholder_text="e.g. inception4c,inception5b",
            fg_color=BG_PANEL,
            border_color=BORDER,
            text_color=TEXT_MAIN,
            font=ModernFont(family="Segoe UI", size=10),
            height=26,
        )
        self._custom_layers_entry.grid(row=2, column=0, sticky="ew", pady=(0, 4))
        Tooltip(
            self._custom_layers_entry,
            "Direct layer input: comma-separated list of layer names. Auto-syncs with badges above.",
        )

    def _build_sidebar_parameters(self, sb):
        # Parameters
        self._section(sb, row=14, label="Parameters")
        params = ModernFrame(sb, fg_color=BG_CARD)
        params.grid(row=15, column=0, sticky="ew", padx=16, pady=(0, 8))
        params.grid_columnconfigure(0, weight=1)

        self._slider(
            params,
            0,
            "Iterations",
            self._iterations,
            1,
            250,
            True,
            tip="Steps per octave. More = deeper, more vivid hallucination (too high = burnt/oversaturated). "
            "Normal: 15-35. Videos auto-cut to 1/10th (1-4) to prevent feedback blowout.",
            curve=2.2,
            step=1,
        )
        self._slider(
            params,
            1,
            "Octaves (0=auto)",
            self._octaves,
            0,
            10,
            True,
            tip="Multi-scale passes. 0 = auto-detects down to ~250px. Low octaves create large structures; "
            "high octaves refine fine details. Set 1 for surface texture only, 3-6 for strong distortion.",
            curve=1.0,
            step=1,
        )
        self._slider(
            params,
            2,
            "Octave Scale",
            self._octave_scale,
            1.05,
            2.0,
            False,
            tip="Size jump between octaves (e.g. 1.4 = 40% bigger each pass). "
            "1.2-1.3 = smooth, gradual transitions. 1.4-1.6 = dramatic, bold structural shifts.",
            curve=1.0,
            step=0.01,
        )
        self._slider(
            params,
            3,
            "Learning Rate",
            self._lr,
            0.001,
            0.2,
            False,
            tip="Dream intensity per step (default 0.015-0.035). "
            "Higher = bolder patterns fast (too high causes noise/glow). "
            "Lower = crisp, delicate line art.",
            curve=2.0,
            step=0.001,
        )
        self._slider(
            params,
            4,
            "Max Size (px)",
            self._size,
            0,
            2048,
            True,
            tip="Max image size (0 = native resolution). Network features are a fixed pixel size: "
            "higher res = tiny, dense details; lower res = big chunky shapes.",
            curve=1.8,
            step=32,
        )
        self._slider(
            params,
            5,
            "Jitter",
            self._jitter,
            0,
            64,
            True,
            tip="Random pixel offset per step to reduce tiling artifacts. 32 is a good default.",
            curve=1.0,
            step=1,
        )

    def _build_sidebar_dynamic_parameters(self, sb):
        # Dynamic Mode Parameters
        # Zoom parameters
        self._zoom_frame = ModernFrame(sb, fg_color=BG_CARD)
        self._zoom_frame.grid(row=16, column=0, sticky="ew", padx=16, pady=(0, 8))
        self._zoom_frame.grid_columnconfigure(0, weight=1)
        ModernLabel(
            self._zoom_frame,
            text="Dream Zoom Settings",
            font=ModernFont(family="Segoe UI", size=11, weight="bold"),
            text_color=TEXT_LABEL,
            anchor="w",
        ).grid(row=0, column=0, sticky="w", padx=12, pady=(8, 2))
        self._slider(
            self._zoom_frame,
            1,
            "Frames",
            self._zoom_frames,
            10,
            1800,
            True,
            tip="Total video frames. 60 frames = 2 sec at 30fps, 1800 frames = 60 sec at 30fps. Because dreams feed into the next frame, "
            "longer runs evolve into deeper morphing visuals.",
            curve=1.5,
            step=1,
        )
        self._slider(
            self._zoom_frame,
            2,
            "Zoom Factor",
            self._zoom_scale,
            1.01,
            1.10,
            False,
            tip="Scale per frame. 1.03 = 3% zoom inward each step. 1.01 = slow zoom; "
            "1.05+ = fast inward zoom.",
            curve=1.0,
            step=0.005,
        )
        self._slider(
            self._zoom_frame,
            3,
            "Rotation (\u00b0)",
            self._zoom_rot,
            -2.0,
            2.0,
            False,
            tip="Spin per frame (+ clockwise, - counter-clockwise). "
            "0.2-0.5 creates visible rotation; 0 = straight zoom.",
            curve=1.0,
            step=0.1,
        )
        self._slider(
            self._zoom_frame,
            4,
            "FPS",
            self._zoom_fps,
            10,
            60,
            True,
            tip="Video playback speed. 30 = standard motion; 60 = smoother motion at larger file size.",
            curve=1.0,
            step=1,
        )
        # Zoom focus picker: click the large preview to aim the tunnel.
        self._zoom_focus_label = ModernLabel(
            self._zoom_frame,
            text="Focus: Center (0.50, 0.50)",
            font=ModernFont(family="Segoe UI", size=10),
            text_color=TEXT_LABEL,
            anchor="w",
            wraplength=280,
        )
        self._zoom_focus_label.grid(row=5, column=0, sticky="ew", padx=12, pady=(2, 2))
        Tooltip(
            self._zoom_focus_label,
            "Where the zoom dives toward, in normalized coords:\n"
            "(0, 0) = top-left, (0.5, 0.5) = center, (1, 1) = bottom-right.\n"
            "Click the large preview image to set it with the mouse.",
        )
        zoom_focus_btns = ModernFrame(self._zoom_frame, fg_color="transparent")
        zoom_focus_btns.grid(row=6, column=0, sticky="ew", padx=12, pady=(0, 8))
        zoom_focus_btns.grid_columnconfigure(0, weight=1)
        zoom_focus_btns.grid_columnconfigure(1, weight=1)
        self._zoom_focus_reset_btn = ModernButton(
            zoom_focus_btns,
            text="Reset Center",
            font=ModernFont(family="Segoe UI", size=10),
            fg_color=BG_PANEL,
            hover_color=BORDER,
            text_color=TEXT_MAIN,
            height=26,
            command=self._reset_zoom_focus,
        )
        self._zoom_focus_reset_btn.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        Tooltip(self._zoom_focus_reset_btn, "Reset the zoom focus to the image center.")
        self._zoom_focus_aim_btn = ModernButton(
            zoom_focus_btns,
            text="Show Input to Aim",
            font=ModernFont(family="Segoe UI", size=10),
            fg_color=BG_PANEL,
            hover_color=BORDER,
            text_color=TEXT_MAIN,
            height=26,
            command=self._show_zoom_focus_for_aiming,
        )
        self._zoom_focus_aim_btn.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        Tooltip(
            self._zoom_focus_aim_btn,
            "Show the input image with the focus reticle so you can click to re-aim "
            "(e.g. after a run replaced the preview with the video result).",
        )
        self._zoom_frame.grid_remove()

        # Video parameters
        self._video_frame = ModernFrame(sb, fg_color=BG_CARD)
        self._video_frame.grid(row=16, column=0, sticky="ew", padx=16, pady=(0, 8))
        self._video_frame.grid_columnconfigure(0, weight=1)
        ModernLabel(
            self._video_frame,
            text="Dream Video Settings",
            font=ModernFont(family="Segoe UI", size=11, weight="bold"),
            text_color=TEXT_LABEL,
            anchor="w",
        ).grid(row=0, column=0, sticky="w", padx=12, pady=(8, 2))
        self._slider(
            self._video_frame,
            1,
            "Blend Factor",
            self._video_blend,
            0.1,
            0.9,
            False,
            tip="Dream memory across video frames (0 = none, 1 = fully previous frame). "
            "0.6-0.8 = smooth, persistent dream trails. "
            "0.2-0.4 = tracks live video action tighter with less ghosting.",
            curve=1.0,
            step=0.05,
        )
        self._video_frame.grid_remove()

        # Video Export parameters: quality preset + explicit Save Video.
        # Generation only dreams the frame sequence; exporting the .mp4 is a
        # separate user action so the same frames can be saved repeatedly at
        # different quality levels without re-running the dream.
        self._video_quality_frame = ModernFrame(sb, fg_color=BG_CARD)
        self._video_quality_frame.grid(row=17, column=0, sticky="ew", padx=16, pady=(0, 8))
        self._video_quality_frame.grid_columnconfigure(0, weight=1)
        ModernLabel(
            self._video_quality_frame,
            text="Video Export",
            font=ModernFont(family="Segoe UI", size=11, weight="bold"),
            text_color=TEXT_LABEL,
            anchor="w",
        ).grid(row=0, column=0, sticky="w", padx=12, pady=(8, 4))

        self._video_quality_seg = ModernSegmentedButton(
            self._video_quality_frame,
            values=["Small", "Balanced", "Lossless"],
            variable=self._video_quality_var,
            selected_color=BORDER,
            selected_hover_color=ACCENT,
            font=ModernFont(family="Segoe UI", size=11),
            command=self._on_video_quality_seg_change,
        )
        self._video_quality_seg.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 4))
        Tooltip(
            self._video_quality_seg,
            "Video encoding preset:\n"
            "• Small: Fast export, compact file (CRF 28).\n"
            "• Balanced: High quality, crisp detail (CRF 17, default).\n"
            "• Lossless: No compression loss (CRF 0).",
        )

        self._video_quality_info_lbl = ModernLabel(
            self._video_quality_frame,
            text=VIDEO_QUALITY_INFO["Balanced"]["desc"],
            font=ModernFont(family="Segoe UI", size=10),
            text_color=TEXT_LABEL,
            anchor="w",
            wraplength=220,
            justify="left",
        )
        self._video_quality_info_lbl.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 4))

        self._slider(
            self._video_quality_frame,
            3,
            "CRF (Quality)",
            self._video_crf,
            0,
            51,
            True,
            tip="H.264 video quality (0-51; lower = better). 0 = lossless (exact pixels), "
            "17 = visually perfect (recommended), 28 = compact file for easy sharing.",
            curve=1.0,
            step=1,
        )
        self._video_crf.trace_add("write", self._on_video_crf_changed)

        self._video_frames_label = ModernLabel(
            self._video_quality_frame,
            text="No video frames yet. Run a Zoom/Video generation first.",
            font=ModernFont(family="Segoe UI", size=10),
            text_color=TEXT_DIM,
            anchor="w",
            wraplength=260,
            justify="left",
        )
        self._video_frames_label.grid(row=4, column=0, sticky="ew", padx=12, pady=(0, 4))

        self._sidebar_save_video_btn = ModernButton(
            self._video_quality_frame,
            text="Save Video",
            font=ModernFont(family="Segoe UI", size=11, weight="bold"),
            fg_color=BORDER,
            hover_color=ACCENT,
            text_color=TEXT_MAIN,
            height=30,
            state="disabled",
            command=self._on_save_video_click,
        )
        self._sidebar_save_video_btn.grid(row=5, column=0, sticky="ew", padx=12, pady=(0, 10))
        Tooltip(
            self._sidebar_save_video_btn,
            "Export the current dreamed frame sequence to an .mp4 at the "
            "quality selected above.\n"
            "Repeat as often as you like (once Small for sharing, once "
            "Lossless for archival) without re-running the dream.",
        )
        self._video_quality_frame.grid_remove()

        # Output options
        self._section(sb, row=18, label="Output")
        out_frame = ModernFrame(sb, fg_color=BG_CARD)
        out_frame.grid(row=19, column=0, sticky="ew", padx=16, pady=(0, 8))
        out_frame.grid_columnconfigure(0, weight=1)

        ModernLabel(
            out_frame,
            text="Output folder",
            font=ModernFont(family="Segoe UI", size=11),
            text_color=TEXT_LABEL,
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 2))

        dir_row = ModernFrame(out_frame, fg_color="transparent")
        dir_row.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 4))
        dir_row.grid_columnconfigure(0, weight=1)

        out_dir_entry = ModernEntry(
            dir_row,
            textvariable=self._out_dir,
            fg_color=BG_PANEL,
            border_color=BORDER,
            text_color=TEXT_MAIN,
            font=ModernFont(family="Segoe UI", size=11),
            height=30,
        )
        out_dir_entry.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        Tooltip(out_dir_entry, "Folder where dreamed images, videos, and timelapses are saved.")

        out_dir_btn = ModernButton(
            dir_row,
            text="\u2026",
            width=32,
            height=30,
            fg_color=BG_PANEL,
            hover_color=BORDER,
            text_color=TEXT_MAIN,
            command=self._browse_outdir,
        )
        out_dir_btn.grid(row=0, column=1)
        Tooltip(out_dir_btn, "Browse for destination folder.")

        ModernLabel(
            out_frame,
            text="Filename stem (optional)",
            font=ModernFont(family="Segoe UI", size=11),
            text_color=TEXT_LABEL,
            anchor="w",
        ).grid(row=2, column=0, sticky="ew", padx=12, pady=(8, 2))
        custom_name_entry = ModernEntry(
            out_frame,
            textvariable=self._custom_name,
            placeholder_text="auto (from input filename)",
            fg_color=BG_PANEL,
            border_color=BORDER,
            text_color=TEXT_MAIN,
            font=ModernFont(family="Segoe UI", size=11),
            height=30,
        )
        custom_name_entry.grid(row=3, column=0, sticky="ew", padx=12, pady=(0, 8))
        Tooltip(
            custom_name_entry,
            "Custom filename prefix. Leave blank to auto-name based on input file and settings.",
        )

        self._timelapse_cb = ModernCheckBox(
            out_frame,
            text="Create Timelapse Video (.mp4)",
            variable=self._timelapse,
            text_color=TEXT_LABEL,
            font=ModernFont(family="Segoe UI", size=12),
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            checkmark_color=TEXT_MAIN,
            command=self._on_timelapse_toggle,
        )
        self._timelapse_cb.grid(row=4, column=0, sticky="w", padx=12, pady=(0, 4))
        Tooltip(
            self._timelapse_cb, "Records the dream assembling octave-by-octave into an MP4 video."
        )

        self._save_frames_cb = ModernCheckBox(
            out_frame,
            text="Save iteration frames",
            variable=self._save_frames,
            text_color=TEXT_LABEL,
            font=ModernFont(family="Segoe UI", size=12),
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            checkmark_color=TEXT_MAIN,
        )
        self._save_frames_cb.grid(row=5, column=0, sticky="w", padx=12, pady=(0, 4))
        Tooltip(
            self._save_frames_cb,
            "Saves PNG images of every single optimization step into a subfolder. Great for custom animations.",
        )

        no_cuda_cb = ModernCheckBox(
            out_frame,
            text="Force CPU (no GPU)",
            variable=self._no_cuda,
            text_color=TEXT_LABEL,
            font=ModernFont(family="Segoe UI", size=12),
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            checkmark_color=TEXT_MAIN,
        )
        no_cuda_cb.grid(row=6, column=0, sticky="w", padx=12, pady=(0, 12))
        Tooltip(
            no_cuda_cb,
            "Forces CPU processing, disabling GPU acceleration (NVIDIA CUDA or Apple MPS). Slower, but avoids GPU memory limits on massive images.",
        )

    # Model change handler

    def _on_model_change(self, label, from_preset=False):
        """Called when the user picks a new model. Refreshes layer dropdown and applies per-model defaults."""
        model_key = MODEL_BY_LABEL.get(label, DEFAULT_MODEL_KEY)
        meta = MODEL_META[model_key]
        self._model_desc_label.configure(text=meta["description"])
        new_layers = _layer_display_list(model_key)
        self._layer_menu.configure(values=new_layers)
        self._layer_var.set(meta["recommended"])

        # Update preset dropdown to match the selected model
        if self._preset_menu is not None:
            model_presets = presets_for_model(model_key)
            self._preset_menu.configure(values=model_presets)

        if not from_preset:
            defaults = meta.get("defaults", {})
            if "iterations" in defaults:
                base_iters = defaults["iterations"]
                self._saved_single_image_iters = base_iters
                if self._mode_var.get() in ("Dream Zoom", "Dream Video"):
                    self._iterations.set(max(1, int(round(base_iters / 10.0))))
                else:
                    self._iterations.set(base_iters)
            if "octaves" in defaults:
                self._octaves.set(defaults["octaves"])
            if "octave_scale" in defaults:
                self._octave_scale.set(defaults["octave_scale"])
            if "lr" in defaults:
                self._lr.set(defaults["lr"])
            if "jitter" in defaults:
                self._jitter.set(defaults["jitter"])
            if "size" in defaults:
                self._size.set(defaults["size"])
            self._layer_mode_var.set("Single Layer")
            self._custom_layers_var.set("")
            self._update_layer_ui()
            self._preset_var.set("Select a preset...")
            self._refresh_slider_labels()
            self._log_msg(
                f"Model [{meta['label']}]: Loaded calibrated defaults "
                f"({defaults.get('iterations', 20)} iters, lr {defaults.get('lr', 0.03):.3f}, "
                f"scale {defaults.get('octave_scale', 1.4):.2f})\n",
                color=ACCENT,
            )

    # Main panel

    def _build_main(self):
        main = ModernFrame(self, fg_color=BG_DEEP)
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(0, weight=6)  # preview area expands to take vast majority of space
        main.grid_rowconfigure(1, weight=0)  # dream/stop buttons
        main.grid_rowconfigure(2, weight=0)  # progress bar
        main.grid_rowconfigure(3, weight=1)  # log area compact

        self._build_main_preview(main)
        self._build_main_actions(main)
        self._build_main_progress(main)
        self._build_main_console_and_queue(main)

    def _build_main_preview(self, main):
        # Preview
        preview_card = ModernFrame(main, fg_color=BG_CARD)
        preview_card.grid(row=0, column=0, sticky="nsew", padx=10, pady=(10, 4))
        preview_card.grid_columnconfigure(0, weight=1)
        preview_card.grid_rowconfigure(0, weight=0)
        preview_card.grid_rowconfigure(1, weight=0)
        preview_card.grid_rowconfigure(2, weight=0)
        preview_card.grid_rowconfigure(3, weight=1)
        preview_card.grid_rowconfigure(4, weight=0)
        preview_card.grid_rowconfigure(5, weight=0)
        preview_card.grid_propagate(False)
        self._preview_card = preview_card

        header = ModernFrame(preview_card, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=10, pady=(6, 2))
        header.grid_columnconfigure(0, weight=1)
        header.grid_columnconfigure(1, weight=0)

        self._preview_title = ModernLabel(
            header,
            text="Output Preview",
            font=ModernFont(family="Segoe UI", size=13, weight="bold"),
            text_color=TEXT_DIM,
            anchor="w",
        )
        self._preview_title.grid(row=0, column=0, sticky="w")

        # Comparison controls container in header
        compare_frame = ModernFrame(header, fg_color="transparent")
        compare_frame.grid(row=0, column=1, sticky="e")
        self._compare_frame = compare_frame

        self._toggle_btn = ModernButton(
            compare_frame,
            text="⇄ Toggle",
            font=ModernFont(family="Segoe UI", size=11),
            fg_color=BG_PANEL,
            hover_color=BORDER,
            text_color=TEXT_MAIN,
            height=26,
            width=68,
            command=self._toggle_compare,
        )
        self._toggle_btn.grid(row=0, column=0, padx=(0, 6))
        Tooltip(
            self._toggle_btn,
            "Instant before/after toggle between original media and dreamed result (Shortcut: Spacebar).",
        )

        self._compare_seg = ModernSegmentedButton(
            compare_frame,
            values=["Dream", "Slide Reveal", "Original"],
            command=self._on_compare_mode_change,
            font=ModernFont(family="Segoe UI", size=11),
            height=26,
            selected_color=BORDER,
            selected_hover_color=BORDER,
        )
        self._compare_seg.set("Dream")
        self._compare_seg.grid(row=0, column=1)
        Tooltip(
            self._compare_seg,
            "Comparison view:\n"
            "• Dream: View full dream output.\n"
            "• Slide Reveal: Drag the divider line on the image to wipe before/after.\n"
            "• Original: View unmodified source media.",
        )
        self._compare_frame.grid_remove()

        self._build_main_viewer_toolbar(preview_card)
        self._build_main_video_bar(preview_card)

    def _build_main_viewer_toolbar(self, preview_card):
        # Integrated viewer toolbar: zoom/pan for full-quality inspection
        zoom_bar = ModernFrame(preview_card, fg_color="transparent")
        zoom_bar.grid(row=1, column=0, sticky="ew", padx=10, pady=(2, 0))
        zoom_bar.grid_columnconfigure(5, weight=1)
        self._zoom_bar = zoom_bar
        self._zoom_out_btn = ModernButton(
            zoom_bar,
            text="−",
            font=ModernFont(family="Segoe UI", size=11, weight="bold"),
            fg_color=BG_PANEL,
            hover_color=BORDER,
            text_color=TEXT_MAIN,
            height=24,
            width=32,
            command=self._on_zoom_out,
        )
        self._zoom_out_btn.grid(row=0, column=0, padx=(0, 4))
        Tooltip(self._zoom_out_btn, "Zoom out (or mouse-wheel down on the preview).")
        self._zoom_label = ModernLabel(
            zoom_bar,
            text="Fit",
            font=ModernFont(family="Consolas", size=11),
            text_color=TEXT_MAIN,
            width=56,
        )
        self._zoom_label.grid(row=0, column=1, padx=(0, 4))
        Tooltip(self._zoom_label, "Current zoom (100% = one image pixel per screen pixel).")
        self._zoom_in_btn = ModernButton(
            zoom_bar,
            text="+",
            font=ModernFont(family="Segoe UI", size=11, weight="bold"),
            fg_color=BG_PANEL,
            hover_color=BORDER,
            text_color=TEXT_MAIN,
            height=24,
            width=32,
            command=self._on_zoom_in,
        )
        self._zoom_in_btn.grid(row=0, column=2, padx=(0, 4))
        Tooltip(self._zoom_in_btn, "Zoom in (or mouse-wheel up on the preview).")
        self._zoom_fit_btn = ModernButton(
            zoom_bar,
            text="Fit",
            font=ModernFont(family="Segoe UI", size=10),
            fg_color=BG_PANEL,
            hover_color=BORDER,
            text_color=TEXT_DIM,
            height=24,
            width=44,
            command=self._on_zoom_fit,
        )
        self._zoom_fit_btn.grid(row=0, column=3, padx=(0, 4))
        Tooltip(self._zoom_fit_btn, "Fit the whole image into the preview area.")
        self._zoom_100_btn = ModernButton(
            zoom_bar,
            text="100%",
            font=ModernFont(family="Segoe UI", size=10),
            fg_color=BG_PANEL,
            hover_color=BORDER,
            text_color=TEXT_DIM,
            height=24,
            width=48,
            command=self._on_zoom_100,
        )
        self._zoom_100_btn.grid(row=0, column=4, padx=(0, 4))
        Tooltip(self._zoom_100_btn, "Show at 100% (one image pixel per screen pixel).")
        ModernLabel(
            zoom_bar,
            text="Wheel: zoom • Drag: pan",
            font=ModernFont(family="Segoe UI", size=10),
            text_color=TEXT_DIM,
            anchor="e",
        ).grid(row=0, column=5, sticky="e")
        self._zoom_bar.grid_remove()

        self._preview_label = ModernLabel(
            preview_card,
            text="Run a dream to see the output here",
            font=ModernFont(family="Segoe UI", size=13),
            text_color=TEXT_DIM,
            fg_color="transparent",
            cursor="hand2",
        )
        self._preview_label.grid(row=3, column=0, sticky="nsew", padx=2, pady=(2, 2))
        self._preview_label.bind("<ButtonPress-1>", self._on_viewer_press)
        self._preview_label.bind("<B1-Motion>", self._on_viewer_drag)
        self._preview_label.bind("<ButtonRelease-1>", self._on_viewer_release)
        self._preview_label.bind("<MouseWheel>", self._on_viewer_wheel)
        self._preview_label.bind("<Button-4>", self._on_viewer_wheel)
        self._preview_label.bind("<Button-5>", self._on_viewer_wheel)
        self._preview_label.bind("<Motion>", self._on_viewer_motion, add="+")
        self._preview_label.bind("<Leave>", self._on_viewer_leave, add="+")
        self._preview_card.bind("<Configure>", self._on_preview_resize)
        # Re-render when the label's allocated size changes (e.g. a toolbar
        # row appears/disappears) so the image always fills the exact space.
        self._preview_label.bind("<Configure>", self._on_viewer_configure, add="+")
        self.bind("<space>", self._on_space_pressed)

    def _build_main_video_bar(self, preview_card):
        # Integrated video bar: timeline scrubber + play/pause
        # Frame-sequence preview (no re-encode): dreamed PNGs are shown in
        # order at their recorded fps. Scrubbing is instant and costs no
        # extra encode; saving via Save Video still produces the final .mp4.
        video_bar = ModernFrame(preview_card, fg_color=BG_PANEL, height=30)
        video_bar.grid_columnconfigure(1, weight=1)
        self._video_bar = video_bar
        self._video_play_btn = ModernButton(
            video_bar,
            text="▶ Play",
            font=ModernFont(family="Segoe UI", size=11, weight="bold"),
            fg_color=BORDER,
            hover_color=ACCENT,
            text_color=TEXT_MAIN,
            height=24,
            width=68,
            command=self._on_video_play_pause,
        )
        self._video_play_btn.grid(row=0, column=0, padx=(8, 4), pady=3)
        Tooltip(self._video_play_btn, "Play / pause the preview (plays dreamed frames in order).")
        self._video_slider = ModernSlider(
            video_bar,
            from_=0.0,
            to=1.0,
            command=self._on_video_slider_change,
            height=12,
        )
        self._video_slider.set(0.0)
        self._video_slider.grid(row=0, column=1, sticky="ew", padx=4, pady=3)
        Tooltip(self._video_slider, "Scrub through dreamed frames.")
        self._video_pos_label = ModernLabel(
            video_bar,
            text="No frames",
            font=ModernFont(family="Consolas", size=10),
            text_color=TEXT_DIM,
            width=160,
        )
        self._video_pos_label.grid(row=0, column=2, padx=(4, 8), pady=3)
        self._video_bar.grid(row=4, column=0, sticky="ew", padx=10, pady=(2, 2))
        self._video_bar.grid_remove()

        # Preview actions and output location.
        open_row = ModernFrame(preview_card, fg_color="transparent")
        open_row.grid(row=5, column=0, sticky="e", padx=10, pady=(2, 6))

        self._view_image_btn = ModernButton(
            open_row,
            text="View Image",
            fg_color=BG_PANEL,
            hover_color=BORDER,
            text_color=TEXT_MAIN,
            font=ModernFont(family="Segoe UI", size=11),
            height=28,
            width=95,
            command=self._on_view_image_toggle,
        )
        self._view_image_btn.grid(row=0, column=0, padx=(0, 6))
        self._view_image_btn.grid_remove()
        Tooltip(self._view_image_btn, "Switch the preview back to the still image.")

        self._view_timelapse_btn = ModernButton(
            open_row,
            text="▶ Timelapse",
            fg_color=BG_PANEL,
            hover_color=BORDER,
            text_color=TEXT_MAIN,
            font=ModernFont(family="Segoe UI", size=11),
            height=28,
            width=95,
            command=self._on_view_timelapse_toggle,
        )
        self._view_timelapse_btn.grid(row=0, column=1, padx=(0, 6))
        self._view_timelapse_btn.grid_remove()
        Tooltip(self._view_timelapse_btn, "Preview the timelapse video inside this window.")

        self._open_dir_btn = ModernButton(
            open_row,
            text="Show in Folder",
            fg_color=BG_PANEL,
            hover_color=BORDER,
            text_color=TEXT_MAIN,
            font=ModernFont(family="Segoe UI", size=11),
            height=28,
            width=110,
            command=self._show_in_folder,
        )
        self._open_dir_btn.grid(row=0, column=2)
        self._open_dir_btn.grid_remove()
        Tooltip(self._open_dir_btn, "Reveal the output file in File Explorer (highlighted).")

    def _build_main_actions(self, main):
        # Dream / Save Video / Queue / Stop buttons
        btn_frame = ModernFrame(main, fg_color="transparent")
        btn_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(4, 2))
        btn_frame.grid_columnconfigure(0, weight=2)
        btn_frame.grid_columnconfigure(1, weight=0)
        btn_frame.grid_columnconfigure(2, weight=1)
        btn_frame.grid_columnconfigure(3, weight=1)
        btn_frame.grid_columnconfigure(4, weight=0)

        self._run_btn = ModernButton(
            btn_frame,
            text="\u25b6 Dream",
            fg_color=BORDER,
            hover_color=ACCENT,
            text_color=TEXT_MAIN,
            font=ModernFont(family="Segoe UI", size=13, weight="bold"),
            height=36,
            command=self._run,
        )
        self._run_btn.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        Tooltip(self._run_btn, "Start dreaming immediately with current settings.")

        self._main_resume_btn = ModernButton(
            btn_frame,
            text="\u25b6 Resume",
            fg_color=BG_PANEL,
            hover_color=ACCENT,
            text_color=TEXT_MAIN,
            font=ModernFont(family="Segoe UI", size=11, weight="bold"),
            height=36,
            width=86,
            command=self._on_main_resume_click,
        )
        self._main_resume_btn.grid(row=0, column=1, sticky="ew", padx=(0, 6))
        self._main_resume_btn.grid_remove()
        Tooltip(self._main_resume_btn, "Resume generation from the last saved frame.")

        self._save_video_btn = ModernButton(
            btn_frame,
            text="Save Video",
            fg_color=BG_PANEL,
            hover_color=BORDER,
            text_color=TEXT_MAIN,
            font=ModernFont(family="Segoe UI", size=12, weight="bold"),
            height=36,
            state="disabled",
            command=self._on_save_video_click,
        )
        self._save_video_btn.grid(row=0, column=2, sticky="ew", padx=(0, 6))
        Tooltip(
            self._save_video_btn,
            "Save the most recent video generation as an .mp4.\n"
            "Active whenever a zoom or video generation has finished and frames are ready to export.",
        )

        self._add_queue_btn = ModernButton(
            btn_frame,
            text="+ Add to Queue",
            fg_color=BG_PANEL,
            hover_color=BORDER,
            text_color=TEXT_MAIN,
            font=ModernFont(family="Segoe UI", size=12, weight="bold"),
            height=36,
            command=self._on_add_to_queue_click,
        )
        self._add_queue_btn.grid(row=0, column=3, sticky="ew", padx=(0, 6))
        Tooltip(
            self._add_queue_btn,
            "Snapshot current settings and add to queue. Jobs run one-by-one automatically so you can keep working.",
        )

        self._stop_btn = ModernButton(
            btn_frame,
            text="Stop",
            fg_color=BG_CARD,
            hover_color=BORDER,
            text_color=TEXT_DIM,
            font=ModernFont(family="Segoe UI", size=13, weight="bold"),
            height=36,
            width=80,
            state="disabled",
            command=self._stop,
        )
        self._stop_btn.grid(row=0, column=4)
        Tooltip(self._stop_btn, "Safely stop the active generation and clean up partial files.")

    def _build_main_progress(self, main):
        # Progress bar
        prog_row = ModernFrame(main, fg_color="transparent")
        prog_row.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 2))
        prog_row.grid_columnconfigure(1, weight=1)

        self._status_label = ModernLabel(
            prog_row,
            text="Idle",
            font=ModernFont(family="Segoe UI", size=11),
            text_color=TEXT_DIM,
            anchor="w",
            width=90,
        )
        self._status_label.grid(row=0, column=0, padx=(0, 6))

        self._progress = ModernProgressBar(
            prog_row,
            fg_color=BG_CARD,
            progress_color=ACCENT,
            height=6,
        )
        self._progress.grid(row=0, column=1, sticky="ew")
        self._progress.set(0)

    def _build_main_console_and_queue(self, main):
        # Console Output / Queue Tabs
        log_card = ModernFrame(main, fg_color=BG_CARD)
        log_card.grid(row=3, column=0, sticky="nsew", padx=10, pady=(2, 10))
        log_card.grid_columnconfigure(0, weight=1)
        log_card.grid_rowconfigure(1, weight=1)

        log_header = ModernFrame(log_card, fg_color="transparent")
        log_header.grid(row=0, column=0, sticky="ew", padx=10, pady=(4, 2))
        log_header.grid_columnconfigure(0, weight=1)

        self._bottom_tab_seg = ModernSegmentedButton(
            log_header,
            values=["Console Output", "Queue (0)"],
            command=self._on_bottom_tab_change,
            font=ModernFont(family="Segoe UI", size=11, weight="bold"),
            height=26,
            selected_color=BORDER,
            selected_hover_color=BORDER,
        )
        self._bottom_tab_seg.set("Console Output")
        self._bottom_tab_seg.grid(row=0, column=0, sticky="w")
        Tooltip(
            self._bottom_tab_seg, "Switch between live console logs and the batch queue manager."
        )

        self._clear_log_btn = ModernButton(
            log_header,
            text="Clear",
            width=48,
            height=22,
            fg_color=BG_PANEL,
            hover_color=BORDER,
            text_color=TEXT_DIM,
            font=ModernFont(family="Segoe UI", size=10),
            command=self._clear_log,
        )
        self._clear_log_btn.grid(row=0, column=1, sticky="e")
        Tooltip(self._clear_log_btn, "Clear console log text.")

        self._queue_action_frame = ModernFrame(log_header, fg_color="transparent")

        self._resume_btn = ModernButton(
            self._queue_action_frame,
            text="\u25b6 Resume",
            width=68,
            height=22,
            fg_color=BORDER,
            hover_color=ACCENT,
            text_color=TEXT_MAIN,
            font=ModernFont(family="Segoe UI", size=10, weight="bold"),
            command=self._resume_queue,
        )
        Tooltip(self._resume_btn, "Start or pause sequential queue processing.")

        self._clear_pending_btn = ModernButton(
            self._queue_action_frame,
            text="Clear Pending",
            width=84,
            height=22,
            fg_color=BG_PANEL,
            hover_color=BORDER,
            text_color=TEXT_DIM,
            font=ModernFont(family="Segoe UI", size=10),
            command=self._clear_pending_jobs,
        )
        self._clear_pending_btn.pack(side="right", padx=(4, 0))
        Tooltip(self._clear_pending_btn, "Remove all waiting (pending) jobs from the queue.")

        self._clear_done_btn = ModernButton(
            self._queue_action_frame,
            text="Clear Done",
            width=70,
            height=22,
            fg_color=BG_PANEL,
            hover_color=BORDER,
            text_color=TEXT_DIM,
            font=ModernFont(family="Segoe UI", size=10),
            command=self._clear_completed_jobs,
        )
        self._clear_done_btn.pack(side="right", padx=(4, 0))
        Tooltip(self._clear_done_btn, "Remove finished and failed jobs from the queue.")

        self._log = ModernTextbox(
            log_card,
            fg_color=BG_PANEL,
            text_color=TEXT_MAIN,
            font=ModernFont(family="Consolas", size=11),
            state="disabled",
            wrap="word",
            height=60,
        )
        self._log.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 6))

        self._queue_container = ModernFrame(log_card, fg_color="transparent")
        self._queue_container.grid_columnconfigure(0, weight=1)
        self._queue_container.grid_rowconfigure(0, weight=1)

        self._queue_scroll = ModernScrollableFrame(
            self._queue_container,
            fg_color=BG_PANEL,
            scrollbar_button_color=BG_CARD,
            scrollbar_button_hover_color=BORDER,
        )
        self._queue_scroll.grid(row=0, column=0, sticky="nsew")
        self._queue_scroll.grid_columnconfigure(0, weight=1)
        self._queue_scroll.inner.grid_columnconfigure(0, weight=1)
        # Content parent for queue cards (outer is the scroll viewport).
        self._queue_inner = self._queue_scroll.inner

    # Widget helpers

    def _section(self, parent, row, label):
        ModernLabel(
            parent,
            text=label,
            font=ModernFont(family="Segoe UI", size=12, weight="bold"),
            text_color=TEXT_LABEL,
            anchor="w",
        ).grid(row=row, column=0, sticky="w", padx=20, pady=(14, 4))

    def _slider(
        self,
        parent,
        row,
        label,
        var,
        from_,
        to,
        is_int,
        tip="",
        curve=1.0,
        step=None,
    ):
        s = _SliderRow(
            self, parent, row, label, var, from_, to, is_int, tip=tip, curve=curve, step=step
        )
        self._slider_registry.append(s)

    def _refresh_slider_labels(self):
        for row in self._slider_registry:
            row.refresh()

    # Mode & Preset change handlers

    def _on_mode_change(self, mode):
        prev_mode = self._previous_mode
        video_modes = ("Dream Zoom", "Dream Video")

        # Decimate iterations when switching from Single Image to a Video mode
        if prev_mode == "Single Image" and mode in video_modes:
            cur_iters = self._iterations.get()
            self._saved_single_image_iters = cur_iters
            new_iters = max(1, int(round(cur_iters / 10.0)))
            self._iterations.set(new_iters)
            self._refresh_slider_labels()
            self._log_msg(
                f"Mode switched to {mode}: Model iterations adjusted to {new_iters} "
                f"(1/10th of single-image iterations to compensate for video accumulation)\n",
                color=ACCENT,
            )
        # Restore iterations when switching from Video mode back to Single Image
        elif prev_mode in video_modes and mode == "Single Image":
            saved_iters = self._saved_single_image_iters
            if saved_iters is None or saved_iters < 1:
                saved_iters = max(1, self._iterations.get() * 10)
            self._iterations.set(saved_iters)
            self._refresh_slider_labels()
            self._log_msg(
                f"Mode switched to Single Image: Restored model iterations to {saved_iters}\n",
                color=ACCENT,
            )

        self._previous_mode = mode
        self._mode_var.set(mode)
        self._reset_run_button_if_resuming()
        self._zoom_frame.grid_remove()
        self._video_frame.grid_remove()
        if mode == "Dream Zoom":
            self._zoom_frame.grid(row=16, column=0, sticky="ew", padx=16, pady=(0, 8))
            self._timelapse_cb.grid_remove()
            self._save_frames_cb.grid_remove()
            # Timelapse/frames only apply to single-image runs; drop stale
            # values so they can't leak into Zoom/Video jobs (their checkboxes
            # are hidden here, so a leftover True would be invisible state).
            self._timelapse.set(False)
            self._save_frames.set(False)
            if not self._image_path.get():
                self._drop_zone.configure(text="Click to browse\nor drop initial image here")
        elif mode == "Dream Video":
            self._video_frame.grid(row=16, column=0, sticky="ew", padx=16, pady=(0, 8))
            self._timelapse_cb.grid_remove()
            self._save_frames_cb.grid_remove()
            self._timelapse.set(False)
            self._save_frames.set(False)
            if not self._image_path.get():
                self._drop_zone.configure(text="Click to browse\nor drop a video file here")
        else:
            self._timelapse_cb.grid()
            self._save_frames_cb.grid()
            if not self._image_path.get():
                self._drop_zone.configure(text="Click to browse\nor drop an image here")
        self._update_video_quality_visibility(mode)
        # Entering Zoom mode with an input but no output yet: show the
        # input with the focus reticle so the user can click to aim right
        # away. If an output is already shown, leave it; the Aim button
        # brings the input back for re-aiming.
        if mode == "Dream Zoom":
            self._zoom_focus_label_update()
            idle = self._proc is None or self._proc.poll() is not None
            if (
                idle
                and self._input_pil is not None
                and not self._output_path
                and self._dream_pil is None
            ):
                self._render_zoom_focus_preview()
        else:
            self._showing_zoom_focus = False

    # Zoom focus (click-to-aim)

    def _zoom_focus_label_update(self):
        if self._zoom_focus_label is None:
            return
        try:
            fx = clamp01(self._zoom_cx.get())
            fy = clamp01(self._zoom_cy.get())
        except (AttributeError, tk.TclError, TypeError, ValueError):
            fx, fy = 0.5, 0.5
        if abs(fx - 0.5) < 0.005 and abs(fy - 0.5) < 0.005:
            text = "Focus: Center (0.50, 0.50) - click preview image to aim"
        else:
            text = f"Focus: ({fx:.2f}, {fy:.2f}) - click preview image to move"
        try:
            self._zoom_focus_label.configure(text=text)
        except (AttributeError, tk.TclError):
            pass

    def _set_zoom_focus(self, fx, fy):
        fx, fy = clamp01(fx), clamp01(fy)
        try:
            self._zoom_cx.set(fx)
            self._zoom_cy.set(fy)
        except (AttributeError, tk.TclError):
            return False
        self._zoom_focus_label_update()
        idle = self._proc is None or self._proc.poll() is not None
        if (
            self._mode_var.get() == "Dream Zoom"
            and idle
            and self._input_pil is not None
            and (self._showing_zoom_focus or self._dream_pil is None)
        ):
            self._render_zoom_focus_preview()
        return True

    def _reset_zoom_focus(self):
        self._set_zoom_focus(0.5, 0.5)

    def _show_zoom_focus_for_aiming(self):
        if self._input_pil is None:
            self._log_msg("Load an input image first, then click it to aim the zoom.\n")
            return
        self._render_zoom_focus_preview()

    def _render_zoom_focus_preview(self):
        """Show the input image with the focus reticle in the main preview."""
        if self._input_pil is None:
            return
        try:
            fx = clamp01(self._zoom_cx.get())
            fy = clamp01(self._zoom_cy.get())
        except (AttributeError, tk.TclError, TypeError, ValueError):
            fx, fy = 0.5, 0.5
        marked = draw_zoom_focus_marker(self._input_pil, fx, fy)
        self._showing_zoom_focus = True
        self._show_output_preview(marked)
        try:
            self._preview_title.configure(text="Zoom Focus: click image to aim")
        except (AttributeError, tk.TclError):
            pass
        self._zoom_focus_label_update()

    def _on_video_quality_seg_change(self, value: str):
        if value in VIDEO_QUALITY_INFO:
            self._video_crf.set(VIDEO_QUALITY_INFO[value]["crf"])
            self._video_quality_var.set(value)
            if self._video_quality_info_lbl is not None:
                self._video_quality_info_lbl.configure(text=VIDEO_QUALITY_INFO[value]["desc"])

    def _get_current_crf(self):
        try:
            return int(self._video_crf.get())
        except (ValueError, tk.TclError):
            return 17

    def _on_video_crf_changed(self, *args):
        try:
            crf = self._video_crf.get()
        except (ValueError, tk.TclError):
            return

        name = _CRF_TO_QUALITY.get(crf)
        if name is not None:
            desc = VIDEO_QUALITY_INFO[name]["desc"]
        else:
            name = "Custom"
            if crf < 17:
                qual = "Near-Lossless / Ultra"
            elif crf <= 28:
                qual = "High / Balanced"
            else:
                qual = "Compressed / Small"
            desc = f"Custom: {qual} (CRF {crf}, 0=lossless, 51=max compression)"

        if self._video_quality_seg is not None:
            self._video_quality_seg.set(name)
        self._video_quality_var.set(name)
        if self._video_quality_info_lbl is not None:
            self._video_quality_info_lbl.configure(text=desc)

    # Separated video export: generation dreams a frame sequence, and these
    # methods encode it to .mp4 at the chosen quality without re-dreaming.

    def _has_ready_video_frames(self):
        d = self._video_frames_dir
        try:
            return bool(d and os.path.isdir(d) and count_saved_video_frames(d) > 0)
        except OSError:
            return False

    def _refresh_video_save_ui(self):
        btns = [
            b
            for b in (self._save_video_btn, getattr(self, "_sidebar_save_video_btn", None))
            if b is not None
        ]
        lbl = self._video_frames_label
        if not btns and lbl is None:
            return
        if self._video_save_in_progress:
            for btn in btns:
                btn.configure(state="disabled", text="Encoding...")
            target = self._video_export_target
            target_name = Path(target).name if target else "video.mp4"
            if lbl is not None:
                try:
                    lbl.configure(
                        text=f"Encoding video -> {target_name}...",
                        text_color=WARNING,
                    )
                except (AttributeError, tk.TclError):
                    pass
            return
        if self._has_ready_video_frames():
            for btn in btns:
                btn.configure(state="normal", text="Save Video")
            try:
                count = int(self._video_frames_count) or count_saved_video_frames(
                    self._video_frames_dir
                )
            except (TypeError, ValueError, OSError):
                count = 0
            try:
                fps = float(self._video_frames_fps or 30.0)
            except (TypeError, ValueError):
                fps = 30.0
            jid = self._video_frames_job_id
            job_txt = f" (Job #{jid})" if jid is not None else ""
            text = f"{count} frames ready{job_txt} @ {fps:g} fps: pick a quality and Save Video."
            last = self._video_last_save
            if last:
                last_crf = self._video_last_crf
                suffix = f" (CRF {last_crf})" if last_crf is not None else ""
                text += f"\nLast saved: {Path(last).name}{suffix}."
            if lbl is not None:
                try:
                    lbl.configure(text=text, text_color=TEXT_LABEL)
                except (AttributeError, tk.TclError):
                    pass
        else:
            for btn in btns:
                btn.configure(state="disabled", text="Save Video")
            if lbl is not None:
                try:
                    lbl.configure(
                        text="No video frames yet. Run a Zoom/Video generation first.",
                        text_color=TEXT_DIM,
                    )
                except (AttributeError, tk.TclError):
                    pass

    def _default_video_save_name(self, out_dir=None):
        crf = self._get_current_crf()

        job = self._video_frames_job
        jid = self._video_frames_job_id
        if job is None and jid is not None:
            for j in self._job_queue:
                if getattr(j, "id", None) == jid:
                    job = j
                    break

        frames_dir = self._video_frames_dir
        meta = read_video_frames_meta(frames_dir) if frames_dir else {}

        source = getattr(job, "image_path", None) or self._video_frames_source or meta.get("source")
        custom_name = getattr(job, "custom_name", None) or meta.get("custom_name")
        size = getattr(job, "size", None) if job is not None else meta.get("size")
        width = meta.get("width")
        height = meta.get("height")
        model = getattr(job, "model_key", None) or meta.get("model")
        layers = getattr(job, "layer_spec", None) or meta.get("layers")
        iterations = getattr(job, "iterations", None) if job is not None else meta.get("iterations")
        target_class = getattr(job, "target_class", None) or meta.get("target_class")
        class_name = meta.get("class_name")
        guide = getattr(job, "guide_path", None) or meta.get("guide")
        mode = getattr(job, "mode", None) or self._video_frames_mode or meta.get("mode") or "zoom"

        candidate = build_systematic_video_name(
            source=source,
            custom_name=custom_name,
            size=size,
            width=width,
            height=height,
            model=model,
            layers=layers,
            iterations=iterations,
            target_class=target_class,
            class_name=class_name,
            guide=guide,
            mode=mode,
            crf=crf,
        )

        target_dir = out_dir if out_dir is not None else resolve_out_dir(self._out_dir.get())
        return resolve_name_conflict(target_dir, candidate)

    def _build_encode_cmd(self, frames_dir, out_path, fps, audio_from=None):
        cmd = [
            venv_python(),
            "-m",
            "app.deepdream",
            "--encode-frames",
            frames_dir,
            "--out",
            out_path,
            "--fps",
            f"{float(fps):.2f}",
        ]
        vq_lower = (self._video_quality_var.get() or "").lower()
        if vq_lower in ("small", "balanced", "lossless"):
            cmd += ["--video-quality", vq_lower]
        crf = self._get_current_crf()
        cmd += ["--crf", str(crf)]
        if audio_from and os.path.isfile(audio_from):
            cmd += ["--encode-audio-from", audio_from]
        return cmd

    def _on_save_video_click(self):
        if self._video_save_in_progress or not self._has_ready_video_frames():
            return
        frames_dir = self._video_frames_dir
        count = self._video_frames_count or count_saved_video_frames(frames_dir)
        fps = float(self._video_frames_fps or 30.0)
        crf = self._get_current_crf()
        mode = self._video_frames_mode or ""
        audio_from = None
        if "video" in mode.lower():
            src = self._video_frames_source
            if src and os.path.isfile(src):
                audio_from = src
        out_dir = resolve_out_dir(self._out_dir.get())
        try:
            os.makedirs(out_dir, exist_ok=True)
        except OSError:
            pass
        default_name = self._default_video_save_name(out_dir=out_dir)
        path = filedialog.asksaveasfilename(
            title="Save dreamed video",
            initialdir=out_dir,
            initialfile=default_name,
            defaultextension=".mp4",
            filetypes=[("MP4 video", "*.mp4"), ("All files", "*.*")],
        )
        if not path:
            return
        if not path.lower().endswith(".mp4"):
            path += ".mp4"
        cmd = self._build_encode_cmd(frames_dir, path, fps, audio_from)
        qname = (self._video_quality_var.get() or "Custom").strip() or "Custom"
        self._log_msg(
            f"\n[Video Export] {count} frames @ {fps:g} fps ({qname}, CRF {crf}) -> {path}\n"
            f"$ {' '.join(cmd)}\n\n",
            color=TEXT_DIM,
        )
        self._video_save_in_progress = True
        self._video_export_target = path
        self._tracker = ProgressTracker()
        self._progress.set(0)
        self._status_label.configure(text="Encoding video...", text_color=WARNING)
        self._stop_video_playback()
        try:
            self._preview_title.configure(text=f"Encoding Video -> {Path(path).name}...")
        except tk.TclError:
            pass
        self._output_path = None
        if not self._queue_running and self._proc is None:
            self._stop_btn.configure(state="normal", text_color=TEXT_MAIN)
        self._refresh_video_save_ui()
        job_id = self._video_frames_job_id
        try:
            proc = self._start_encode_process(cmd)
        except OSError as exc:
            self._finish_video_encode(
                -1, f"Error: could not start video export: {exc}\n", path, job_id
            )
            return
        # Started on the Tk thread so _stop() can never miss the process.
        self._encode_proc = proc
        self._ensure_poll()
        threading.Thread(
            target=self._run_video_encode,
            daemon=True,
            args=(cmd, proc, path, job_id, count, fps, crf),
        ).start()

    def _handle_encode_line(self, line, out_path):
        if self._is_closing:
            return
        if parse_event(line) is None:
            self._log_msg(line)
        frac, status = self._tracker.update(line)
        if frac is not None:
            self._progress.set(frac)
        if status:
            self._status_label.configure(text=status, text_color=WARNING)
            lbl = self._video_frames_label
            if lbl is not None and self._video_save_in_progress:
                try:
                    dest_name = Path(out_path).name
                    lbl.configure(
                        text=f"{status} -> {dest_name}",
                        text_color=WARNING,
                    )
                except tk.TclError:
                    pass
        if line.strip().startswith("Done in"):
            self._status_label.configure(text="Video saved ✓", text_color=SUCCESS)
            self._progress.set(1.0)

    def _start_encode_process(self, cmd):
        """Start the encode subprocess. Called on the Tk thread."""
        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="backslashreplace",
            cwd=str(ROOT),
            creationflags=(
                getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
            ),
            start_new_session=(sys.platform != "win32"),
        )

    def _run_video_encode(self, cmd, proc, out_path, job_id, count, fps, crf):
        """Worker thread: stream an already-started encode subprocess into the queue.

        Touches no Tk state and no Tcl calls. The process is started on the Tk
        thread by _on_save_video_click so _stop() can always see it; log lines
        and the final result are queued for _poll_log to apply on the Tk thread.
        """
        rc = -1
        err_msg = ""
        try:
            if proc.stdout:
                for line in iter(proc.stdout.readline, ""):
                    if not line:
                        break
                    self._encode_queue.put(("line", line, out_path))
            proc.wait()
            rc = proc.returncode
        except OSError as exc:
            err_msg = f"Error: could not start video export: {exc}\n"
        except Exception as exc:
            err_msg = f"Error during video export: {exc}\n"
        finally:
            self._encode_queue.put(("done", rc, err_msg, out_path, job_id))

    def _finish_video_encode(self, rc, log_text, out_path, job_id):
        self._video_save_in_progress = False
        self._video_export_target = None
        # Cleared on the Tk thread so _stop() never races the worker's assignment.
        self._encode_proc = None
        if not self._queue_running and self._proc is None:
            self._stop_btn.configure(state="disabled", text_color=TEXT_DIM)
        if self._is_closing:
            return
        if log_text:
            self._log_msg(log_text if log_text.endswith("\n") else log_text + "\n")
        ok = rc == 0 and os.path.isfile(out_path) and os.path.getsize(out_path) > 0
        if ok:
            try:
                crf = int(self._video_crf.get())
            except (ValueError, tk.TclError):
                crf = None
            self._output_path = out_path
            self._video_last_save = out_path
            self._video_last_crf = crf
            self._progress.set(1.0)
            self._status_label.configure(text="Video saved ✓", text_color=SUCCESS)
            for j in self._job_queue:
                if getattr(j, "id", None) == job_id:
                    j.output_path = out_path
                    break
            self._is_video_output = True
            self._timelapse_mp4 = None
            try:
                self._view_image_btn.grid_remove()
                self._view_timelapse_btn.grid_remove()
            except tk.TclError:
                pass
            ok_preview = self._setup_mp4_preview(
                out_path, title=f"Video Saved: {Path(out_path).name}"
            )
            if not ok_preview:
                try:
                    self._preview_title.configure(text=f"Video Saved: {Path(out_path).name}")
                except tk.TclError:
                    pass
            self._open_dir_btn.grid()
            self._log_msg(
                f"\nSaved: {out_path}\n"
                "Previewing in-window; scrub or press Play. "
                "You can Save Video again at another quality without re-dreaming.\n",
                color=SUCCESS,
            )
        else:
            self._output_path = None
            self._progress.set(0)
            self._status_label.configure(text="Export failed", text_color=ERROR_COL)
            self._log_msg("\nVideo export failed. See the log above.\n", color=ERROR_COL)
            try:
                messagebox.showerror(
                    "Video Export Failed",
                    "Could not export the video.\nSee the console log for details.",
                )
            except tk.TclError:
                pass
        self._refresh_video_save_ui()

    def _on_timelapse_toggle(self):
        self._update_video_quality_visibility()

    def _update_video_quality_visibility(self, mode=None):
        if self._video_quality_frame is None:
            return
        if mode is None:
            mode = self._mode_var.get()
        if (
            mode in ("Dream Zoom", "Dream Video")
            or self._timelapse.get()
            or self._has_ready_video_frames()
        ):
            self._video_quality_frame.grid(row=17, column=0, sticky="ew", padx=16, pady=(0, 8))
        else:
            self._video_quality_frame.grid_remove()

    def _on_preset_change(self, name):
        p = PRESETS.get(name)
        if not p:
            return
        if "model" in p:
            for lbl, k in MODEL_BY_LABEL.items():
                if k == p["model"]:
                    if self._model_var.get() != lbl:
                        self._model_var.set(lbl)
                        self._on_model_change(lbl, from_preset=True)
                    break
        self._preset_var.set(name)
        if "layer" in p:
            model_key = p.get("model", DEFAULT_MODEL_KEY)
            layer_key = p["layer"]
            for lk, ldisp in MODEL_META[model_key]["layers"]:
                if lk == layer_key:
                    self._layer_var.set(ldisp)
                    break
        if p.get("custom_layers"):
            self._layer_mode_var.set("Multi-Layer")
            self._custom_layers_var.set(p["custom_layers"])
        else:
            self._layer_mode_var.set("Single Layer")
            self._custom_layers_var.set("")
        self._update_layer_ui()
        if "iterations" in p:
            base_iters = p["iterations"]
            self._saved_single_image_iters = base_iters
            if self._mode_var.get() in ("Dream Zoom", "Dream Video"):
                self._iterations.set(max(1, int(round(base_iters / 10.0))))
            else:
                self._iterations.set(base_iters)
        if "octaves" in p:
            self._octaves.set(p["octaves"])
        if "octave_scale" in p:
            self._octave_scale.set(p["octave_scale"])
        if "lr" in p:
            self._lr.set(p["lr"])
        if "jitter" in p:
            self._jitter.set(p["jitter"])
        if "size" in p:
            self._size.set(p["size"])
        self._refresh_slider_labels()
        if "desc" in p:
            self._log_msg(f"Preset [{name}]: {p['desc']}\n", color=ACCENT)

    def _on_guidance_change(self, mode):
        self._guide_frame.grid_remove()
        self._class_frame.grid_remove()
        if mode == "Guide Image":
            self._guide_frame.grid(row=9, column=0, sticky="ew", padx=16, pady=(0, 8))
        elif mode == "Target Class":
            self._class_frame.grid(row=9, column=0, sticky="ew", padx=16, pady=(0, 8))

    # Target Class management (searchable & mouse-selectable)

    def _on_target_class_var_changed(self, *args):
        if self._class_selected_label is None:
            return
        val = self._target_class_var.get().strip()
        if val:
            res = lookup_category(val)
            if res:
                idx, name, label = res
                self._class_selected_label.configure(
                    text=f"Target: {label}",
                    text_color=SUCCESS,
                )
                if self._class_dropdown is not None:
                    try:
                        cur_vals = self._class_dropdown.cget("values") or []
                        if label not in cur_vals:
                            cur_vals = [label] + [v for v in cur_vals if v != label]
                            self._class_dropdown.configure(values=cur_vals)
                        self._class_dropdown.set(label)
                    except (tk.TclError, AttributeError):
                        pass
            else:
                # An unresolvable value must not read as a valid target: the
                # CLI would reject it and the job would fail after starting.
                self._class_selected_label.configure(
                    text=f"Target: {val} (unrecognized \u2014 pick from the list)",
                    text_color=ERROR_COL,
                )
        else:
            self._class_selected_label.configure(
                text="No target selected (search or click below)",
                text_color=TEXT_LABEL,
            )

    def _on_class_search_changed(self, *args):
        self._update_class_list_display()

    def _update_class_list_display(self):
        if self._class_dropdown is None or self._class_listbox is None:
            return
        query = self._class_search_var.get().strip()
        filter_name = self._class_filter_name

        cat_filter = (
            None if (filter_name in ("Popular", "★ Popular", "All") and query) else filter_name
        )
        matches = search_categories(query, category_filter=cat_filter)
        if not matches and cat_filter is not None:
            matches = search_categories(query, category_filter=None)

        if matches:
            vals = [m[2] for m in matches]
        else:
            vals = ["No matching categories"]

        self._class_dropdown.configure(values=vals)
        if matches:
            self._class_dropdown.set(vals[0])
        else:
            self._class_dropdown.set("No matching categories")

        self._class_listbox.delete(0, tk.END)
        for v in vals:
            self._class_listbox.insert(tk.END, f"  {v}")

    def _on_class_category_filter_change(self, val):
        self._class_filter_name = val
        if self._class_search_var.get():
            self._class_search_var.set("")
        else:
            self._update_class_list_display()

    def _on_class_dropdown_select(self, choice):
        if not choice or choice == "No matching categories":
            return
        self._target_class_var.set(choice)
        if self._class_listbox is not None:
            self._class_listbox.selection_clear(0, tk.END)
            for i in range(self._class_listbox.size()):
                item_text = self._class_listbox.get(i).strip()
                if item_text == choice.strip():
                    self._class_listbox.selection_set(i)
                    self._class_listbox.see(i)
                    break

    def _on_class_listbox_select(self, event=None):
        if self._class_listbox is None:
            return
        sel = self._class_listbox.curselection()
        if not sel:
            return
        chosen = self._class_listbox.get(sel[0]).strip()
        if not chosen or chosen == "No matching categories":
            return
        self._target_class_var.set(chosen)
        if self._class_dropdown is not None:
            self._class_dropdown.set(chosen)

    def _class_list_row_pixels(self):
        """Approximate pixel height of one listbox row for drag-resizing."""
        try:
            import tkinter.font as _tkfont

            f = _tkfont.Font(font=self._class_listbox.cget("font"))
            return max(12, int(f.metrics("linespace")) + 2)
        except (tk.TclError, RuntimeError, TypeError):
            return 16

    def _on_class_list_resize_start(self, event=None):
        try:
            h = int(self._class_listbox.cget("height"))
        except (ValueError, tk.TclError):
            h = self._class_listbox_height
        y = getattr(event, "y_root", 0) or 0
        self._class_list_resize_state = (y, h)
        return "break"

    def _on_class_list_resize_drag(self, event=None):
        state = self._class_list_resize_state
        if state is None or event is None:
            return "break"
        start_y, start_h = state
        dy = getattr(event, "y_root", start_y) - start_y
        row_px = max(1, self._class_list_row_pixels())
        min_h = int(self._class_listbox_min_height)
        max_h = int(self._class_listbox_max_height)
        new_h = min(max_h, max(min_h, start_h + int(round(dy / float(row_px)))))
        try:
            if int(self._class_listbox.cget("height")) != new_h:
                self._class_listbox.configure(height=new_h)
                self._class_listbox_height = new_h
        except (ValueError, tk.TclError):
            pass
        return "break"

    def _on_class_list_resize_end(self, event=None):
        self._class_list_resize_state = None
        return "break"

    def _clear_target_class(self):
        self._target_class_var.set("")
        if self._class_listbox is not None:
            self._class_listbox.selection_clear(0, tk.END)
        if self._class_dropdown is not None:
            self._class_dropdown.set("")

    def _clear_class_search(self):
        self._class_search_var.set("")

    def _on_class_search_enter(self, event=None):
        if self._class_listbox is None or self._class_listbox.size() == 0:
            return
        first = self._class_listbox.get(0).strip()
        if first and first != "No matching categories":
            self._target_class_var.set(first)
            if self._class_dropdown is not None:
                self._class_dropdown.set(first)

    # Multi-layer management (integrated in-window)

    def _on_single_layer_select(self, val):
        self._custom_layers_var.set("")
        self._layer_mode_var.set("Single Layer")
        self._update_layer_ui()

    def _on_layer_mode_change(self, mode=None):
        if mode is not None:
            self._layer_mode_var.set(mode)
        else:
            mode = self._layer_mode_var.get()
        if mode == "Multi-Layer":
            if not self._custom_layers_var.get().strip():
                # Seed multi-layer with currently selected single layer
                model_key = MODEL_BY_LABEL.get(self._model_var.get(), DEFAULT_MODEL_KEY)
                layer_value_map = _layer_value_map(model_key)
                cur_key = layer_value_map.get(
                    self._layer_var.get(), self._layer_var.get().split()[0]
                )
                self._custom_layers_var.set(cur_key)
        else:
            self._custom_layers_var.set("")
        self._update_layer_ui()

    def _on_custom_layers_var_changed(self, *args):
        if self._layer_chips_container is not None:
            self._refresh_layer_chips()

    def _toggle_layer(self, key):
        if self._proc and self._proc.poll() is None:
            return
        raw = self._custom_layers_var.get().strip()
        layers = [layer_name.strip() for layer_name in raw.split(",") if layer_name.strip()]
        if key in layers:
            layers.remove(key)
        else:
            layers.append(key)
        self._custom_layers_var.set(",".join(layers))
        self._refresh_layer_chips()

    def _select_recommended_layers(self):
        if self._proc and self._proc.poll() is None:
            return
        model_key = MODEL_BY_LABEL.get(self._model_var.get(), DEFAULT_MODEL_KEY)
        rec = RECOMMENDED_MULTI_LAYERS.get(model_key, ["inception4c", "inception5b"])
        self._custom_layers_var.set(",".join(rec))
        self._refresh_layer_chips()

    def _select_all_layers(self):
        if self._proc and self._proc.poll() is None:
            return
        model_key = MODEL_BY_LABEL.get(self._model_var.get(), DEFAULT_MODEL_KEY)
        meta = MODEL_META.get(model_key, MODEL_META[DEFAULT_MODEL_KEY])
        all_keys = [k for k, _ in meta["layers"]]
        self._custom_layers_var.set(",".join(all_keys))
        self._refresh_layer_chips()

    def _remove_layer_chip(self, key):
        if self._proc and self._proc.poll() is None:
            return
        raw = self._custom_layers_var.get().strip()
        layers = [layer_name.strip() for layer_name in raw.split(",") if layer_name.strip()]
        if key in layers:
            layers.remove(key)
        self._custom_layers_var.set(",".join(layers))
        self._refresh_layer_chips()

    def _clear_multi_layers(self):
        if self._proc and self._proc.poll() is None:
            return
        self._custom_layers_var.set("")
        self._refresh_layer_chips()

    def _update_layer_ui(self):
        mode = self._layer_mode_var.get()
        if self._layer_mode_seg is not None:
            self._layer_mode_seg.set(mode)
        if mode == "Multi-Layer":
            if self._single_layer_frame is not None:
                self._single_layer_frame.grid_remove()
            if self._multi_layer_frame is not None:
                self._multi_layer_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 8))
            self._refresh_layer_chips()
        else:
            if self._multi_layer_frame is not None:
                self._multi_layer_frame.grid_remove()
            if self._single_layer_frame is not None:
                self._single_layer_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 8))

    def _refresh_layer_chips(self):
        if self._layer_chips_container is None:
            return
        for child in self._layer_chips_container.winfo_children():
            child.destroy()

        model_key = MODEL_BY_LABEL.get(self._model_var.get(), DEFAULT_MODEL_KEY)
        meta = MODEL_META.get(model_key, MODEL_META[DEFAULT_MODEL_KEY])
        layer_items = meta["layers"]
        layer_desc_map = dict(layer_items)
        tiers = LAYER_TIERS.get(model_key, [])

        raw = self._custom_layers_var.get().strip()
        active_layers = [layer_name.strip() for layer_name in raw.split(",") if layer_name.strip()]

        if self._layer_badge_count is not None:
            self._layer_badge_count.configure(text=f"Active ({len(active_layers)})")

        self._layer_chips_container.grid_columnconfigure(0, weight=1)
        self._layer_chips_container.grid_columnconfigure(1, weight=1)

        row_idx = 0
        used_keys = set()

        for tier_title, _tier_sub, tier_keys in tiers:
            matching = [k for k in tier_keys if k in layer_desc_map]
            if not matching:
                continue
            row_idx = self._add_layer_tier_label(tier_title, row_idx)
            used_keys.update(matching)
            row_idx = self._add_layer_chips(matching, active_layers, layer_desc_map, row_idx)

        # Any leftover keys not in tiers
        leftovers = [k for k, _ in layer_items if k not in used_keys]
        if leftovers:
            row_idx = self._add_layer_tier_label("Other Layers", row_idx)
            self._add_layer_chips(leftovers, active_layers, layer_desc_map, row_idx)

    def _add_layer_tier_label(self, title, row_idx):
        """Render a tier heading in the chips grid; returns the next free row."""
        lbl = ModernLabel(
            self._layer_chips_container,
            text=title,
            font=ModernFont(family="Segoe UI", size=9, weight="bold"),
            text_color=TEXT_LABEL,
            anchor="w",
        )
        lbl.grid(row=row_idx, column=0, columnspan=2, sticky="w", padx=6, pady=(6, 2))
        return row_idx + 1

    def _add_layer_chips(self, keys, active_layers, layer_desc_map, row_idx):
        """Render one group of layer chips in a two-column grid; returns the next row."""
        for j, key in enumerate(keys):
            is_active = key in active_layers
            raw_desc = layer_desc_map[key]
            clean_desc = raw_desc.split(" - ", 1)[1].strip() if " - " in raw_desc else raw_desc

            r = row_idx + (j // 2)
            c = j % 2

            btn = ModernButton(
                self._layer_chips_container,
                text=f"✓ {key}" if is_active else f"+ {key}",
                height=26,
                fg_color=ACCENT if is_active else BG_CARD,
                hover_color=ACCENT_HOVER if is_active else BORDER,
                text_color=BG_DEEP if is_active else TEXT_DIM,
                border_width=0 if is_active else 1,
                border_color=BORDER,
                font=ModernFont(
                    family="Segoe UI",
                    size=10,
                    weight="bold" if is_active else "normal",
                ),
                command=(
                    (lambda k=key: self._remove_layer_chip(k))
                    if is_active
                    else (lambda k=key: self._toggle_layer(k))
                ),
            )
            if len(keys) % 2 == 1 and j == len(keys) - 1 and c == 0:
                btn.grid(row=r, column=0, columnspan=2, padx=3, pady=2, sticky="ew")
            else:
                btn.grid(row=r, column=c, padx=3, pady=2, sticky="ew")

            Tooltip(
                btn,
                f"{key}: {clean_desc}\n"
                f"Status: {'Active (click to deactivate)' if is_active else 'Inactive (click to activate)'}",
            )
        return row_idx + (len(keys) + 1) // 2

    # Image & Media browsing / preview

    def _on_drop(self, event):
        """Handle files dropped from Explorer onto the drop zone."""
        paths = parse_dropped_paths(getattr(event, "data", event))
        if not paths:
            self._drop_zone.configure(fg_color=BG_PANEL)
            return

        valid_paths = [p for p in paths if os.path.isfile(p)]
        media_paths = [p for p in valid_paths if Path(p).suffix.lower() in MEDIA_EXTENSIONS]
        skipped = len(valid_paths) - len(media_paths)
        if skipped:
            self._log_msg(
                f"\n[Dropped {skipped} unsupported file(s) ignored]\n",
                color=WARNING,
            )
        if not media_paths:
            self._drop_zone.configure(fg_color=BG_PANEL)
            return

        if len(media_paths) == 1:
            self._load_image_preview(media_paths[0])
        else:
            # Multi-file drop: load the first into preview, and queue all of them
            self._load_image_preview(media_paths[0])

            # Validate guidance settings once before queuing to avoid
            # showing a blocking modal dialog for every single file.
            guidance_mode = self._guidance_mode_var.get()
            if guidance_mode == "Guide Image":
                guide_path = self._guide_path.get().strip().strip("\"'")
                if not guide_path or not os.path.isfile(guide_path):
                    self._drop_zone.configure(fg_color=BG_PANEL)
                    messagebox.showwarning(
                        "Missing guide",
                        "Please select a valid guide image before batch queuing.",
                    )
                    return
            elif guidance_mode == "Target Class":
                target_class = self._target_class_var.get().strip()
                if not target_class:
                    self._drop_zone.configure(fg_color=BG_PANEL)
                    messagebox.showwarning(
                        "Missing Target Class",
                        "Please select or search for an ImageNet target category before batch queuing.",
                    )
                    return

            # The filename stem is one global setting, so validate it once
            # here too: otherwise the per-job check in _create_job_from_inputs
            # would pop one modal dialog for every dropped file.
            custom_name = self._custom_name.get().strip() or None
            if custom_name is not None:
                name_error = filename_stem_error(
                    custom_name, what="Filename stem", max_len=MAX_NAME_LEN
                )
                if name_error:
                    self._drop_zone.configure(fg_color=BG_PANEL)
                    messagebox.showwarning("Invalid Filename", name_error.removeprefix("Error: "))
                    return

            queued_count = 0
            for p in media_paths:
                job = self._create_job_from_inputs(image_path=p)
                if job:
                    self._job_queue.append(job)
                    queued_count += 1
            if queued_count > 0:
                self._log_msg(
                    f"\n[Batch Queue] Added {queued_count} jobs from dropped files.\n",
                    color=ACCENT,
                )
                self._update_queue_badge()
                self._refresh_queue_ui()
                if not self._queue_running:
                    self._start_queue()
                badge_text = self._get_queue_badge_text()
                self._bottom_tab_seg.set(badge_text)
                self._on_bottom_tab_change(badge_text)

        self._drop_zone.configure(fg_color=BG_PANEL)

    def _browse_image(self):
        image_patterns = " ".join(_dialog_pattern_for_ext(ext) for ext in IMAGE_EXTENSIONS)
        video_patterns = " ".join(_dialog_pattern_for_ext(ext) for ext in VIDEO_EXTENSIONS)
        if self._mode_var.get() == "Dream Video":
            filetypes = [
                ("Video files", video_patterns),
                ("Images", image_patterns),
                ("All files", "*.*"),
            ]
        else:
            filetypes = [
                ("Images", image_patterns),
                ("Video files", video_patterns),
                ("All files", "*.*"),
            ]
        path = filedialog.askopenfilename(
            title="Select input media",
            filetypes=filetypes,
        )
        if path:
            self._load_image_preview(path)

    def _load_image_preview(self, path):
        self._reset_run_button_if_resuming()
        path = str(path).strip().strip("\"'")
        ext = Path(path).suffix.lower()
        if ext not in MEDIA_EXTENSIONS:
            messagebox.showwarning(
                "Unsupported file",
                f"Please select an image or video file.\n\nUnsupported: {path}",
            )
            return
        if ext in VIDEO_EXTENSIONS:
            try:
                first_frame = iio.imread(path, index=0)
                pil = Image.fromarray(first_frame).convert("RGB")
            except (OSError, ValueError):
                try:
                    reader = imageio.get_reader(path)
                    try:
                        pil = Image.fromarray(reader.get_data(0)).convert("RGB")
                    finally:
                        reader.close()
                except (OSError, ValueError) as exc:
                    messagebox.showerror("Cannot open video", str(exc))
                    return
        else:
            try:
                with Image.open(path) as _raw:
                    pil = ImageOps.exif_transpose(_raw).convert("RGB")
            except (OSError, ValueError) as exc:
                messagebox.showerror("Cannot open image", str(exc))
                return

        self._image_path.set(path)
        self._input_pil = pil.copy()
        self._input_pil_path = os.path.normcase(os.path.abspath(path))
        self._input_pil_mtime = self._file_mtime(path)
        self._dream_pil = None
        self._aligned_input_pil = None
        self._compare_mode = "Dream"
        self._preview_pil = None
        self._output_path = None
        self._timelapse_mp4 = None
        self._viewer_mode = "image"
        self._teardown_video()
        self._reset_viewer()
        self._open_dir_btn.grid_remove()
        try:
            self._view_image_btn.grid_remove()
            self._view_timelapse_btn.grid_remove()
        except tk.TclError:
            pass
        self._compare_frame.grid_remove()

        thumb = pil.copy()
        thumb.thumbnail((280, 110))
        preview_thumb = ModernImage(thumb, size=(thumb.width, thumb.height))
        # width=0/height=0 auto-sizes to the thumbnail: a fixed text-unit height
        # is read as *pixels* once an image is present, crushing the preview to a
        # thin strip.
        self._drop_zone.configure(image=preview_thumb, text="", width=0, height=0)
        self._drop_zone.image = preview_thumb
        if self._mode_var.get() == "Dream Zoom":
            # Large preview doubles as the zoom-aiming surface: show the
            # input with the focus reticle so a click lands precisely.
            self._showing_zoom_focus = False
            try:
                self._render_zoom_focus_preview()
            except (tk.TclError, OSError, ValueError):
                self._preview_label.configure(
                    image="", text="Input loaded \u2014 click preview to aim zoom"
                )
                self._preview_title.configure(text="Zoom Focus: click image to aim")
        else:
            self._showing_zoom_focus = False
            self._hide_zoom_bar()
            self._preview_label.configure(image="", text="Input loaded \u2014 press \u25b6 Dream")
            self._preview_title.configure(text="Output Preview")
        self._log_msg(f"Loaded: {path}\n")

    def _browse_guide_image(self):
        path = filedialog.askopenfilename(
            title="Select guide image",
            filetypes=[
                (
                    "Images",
                    " ".join(_dialog_pattern_for_ext(ext) for ext in IMAGE_EXTENSIONS),
                ),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._load_guide_preview(path)

    def _load_guide_preview(self, path):
        path = str(path).strip().strip("\"'")
        try:
            with Image.open(path) as _raw:
                pil = ImageOps.exif_transpose(_raw).convert("RGB")
        except (OSError, ValueError) as exc:
            messagebox.showerror("Cannot open guide image", str(exc))
            return
        self._guide_path.set(path)
        thumb = pil.copy()
        thumb.thumbnail((260, 60))
        preview_thumb = ModernImage(thumb, size=(thumb.width, thumb.height))
        # Same auto-size fix as the input preview: a leftover text-unit
        # height would collapse the guide thumbnail to a thin strip.
        self._guide_drop_zone.configure(image=preview_thumb, text="", width=0, height=0)
        self._guide_drop_zone.image = preview_thumb
        self._log_msg(f"Loaded guide: {path}\n")

    def _browse_outdir(self):
        d = filedialog.askdirectory(title="Select output folder")
        if d:
            self._out_dir.set(d)

    # Integrated preview: zoom/pan images + in-window video. Images render at
    # full source resolution (NEAREST past 100%, LANCZOS at/below); videos play
    # in place with a timeline scrubber, no external viewer or re-encode.

    def _get_view_size(self):
        # Prefer the label's real mapped size: it is the exact space the
        # toolbar rows leave for the image, so nothing is wasted or clipped.
        # Before the first layout pass (winfo reports 1) fall back to a
        # card-based estimate; a label <Configure> re-renders once real.
        try:
            lw, lh = self._preview_label.winfo_width(), self._preview_label.winfo_height()
        except (tk.TclError, AttributeError):
            lw, lh = 0, 0
        if lw > 2 and lh > 2:
            return max(int(lw), 100), max(int(lh), 100)
        try:
            card_w = self._preview_card.winfo_width()
            card_h = self._preview_card.winfo_height()
        except (tk.TclError, AttributeError):
            card_w, card_h = 700, 500
        w = max(int(card_w) - 8, 100)
        # Reserve space for the header and action row, then account for toolbars
        # that are visible in the current preview mode.
        extra = 70
        try:
            if self._zoom_bar is not None and self._zoom_bar.grid_info():
                extra += 30
            if self._video_bar is not None and self._video_bar.grid_info():
                extra += 38
        except (tk.TclError, AttributeError):
            extra = 150
        h = max(int(card_h) - extra, 100)
        return w, h

    def _get_current_preview_pil(self):
        """Return the PIL image corresponding to the current comparison mode."""
        # While aiming the Dream Zoom focus, the viewer must show the
        # input-with-reticle stored in _preview_pil, even if a still dream
        # result (_dream_pil) is present from an earlier run.
        if self._showing_zoom_focus and self._preview_pil is not None:
            return self._preview_pil
        if self._dream_pil is None:
            return self._preview_pil
        if self._compare_mode == "Original":
            return self._aligned_input_pil or self._input_pil
        elif self._compare_mode == "Slide Reveal":
            if self._aligned_input_pil is not None:
                return create_split_image(
                    self._aligned_input_pil, self._dream_pil, self._reveal_split
                )
            return self._dream_pil
        else:  # "Dream"
            return self._dream_pil

    def _get_viewer_source(self):
        """Source PIL for the viewer, honoring video/timelapse modes."""
        mode = self._viewer_mode
        if mode in ("video", "timelapse"):
            # Cache-only lookup: a miss must not decode on the Tk thread. The
            # requested frame is already queued by _show_video_frame; leaving
            # the previous frame on screen beats falling back to an unrelated
            # still (the input photo or an older dream).
            return self._load_video_frame(self._video_index)
        return self._get_current_preview_pil()

    def _ensure_viewer_center(self, src):
        if src is None:
            return 0.5, 0.5
        sw, sh = src.size
        cx = self._viewer_cx
        cy = self._viewer_cy
        if cx is None or cy is None:
            return sw / 2.0, sh / 2.0
        try:
            return max(0.0, min(sw, float(cx))), max(0.0, min(sh, float(cy)))
        except (TypeError, ValueError):
            return sw / 2.0, sh / 2.0

    def _reset_viewer(self):
        self._viewer_fit = True
        self._viewer_zoom = 1.0
        self._viewer_cx = None
        self._viewer_cy = None
        self._viewer_drag = None
        self._update_zoom_label()

    def _effective_zoom(self, src, view_w, view_h):
        if self._viewer_fit:
            try:
                sw, sh = src.size
                return fit_zoom_for(sw, sh, view_w, view_h)
            except (AttributeError, ValueError, TypeError):
                return 1.0
        return clamp_zoom(self._viewer_zoom)

    def _render_viewer(self):
        src = self._get_viewer_source()
        if src is None:
            return
        # Slide Reveal composites at dream resolution; zoom still applies to
        # the composite so before/after stays pixel-aligned.
        view_w, view_h = self._get_view_size()
        zoom = self._effective_zoom(src, view_w, view_h)
        cx, cy = self._ensure_viewer_center(src)
        # Keep the stored center in sync (e.g. after a new image resets it).
        self._viewer_cx, self._viewer_cy = cx, cy
        try:
            viewport = render_zoomed_preview(src, view_w, view_h, zoom, cx, cy)
            preview_img = ModernImage(viewport, size=(view_w, view_h))
            self._preview_label.configure(image=preview_img, text="")
            self._preview_label.image = preview_img
        except (tk.TclError, OSError, AttributeError):
            return
        # Remember the size this frame used so <Configure> can ignore the
        # re-entrant events that configuring the image itself emits.
        self._last_render_view_size = (view_w, view_h)
        self._update_zoom_label(zoom)

    def _on_viewer_configure(self, event=None):
        """Re-render when the label's real size changes (toolbar show/hide)."""
        try:
            w = int(getattr(event, "width", 0) or 0)
            h = int(getattr(event, "height", 0) or 0)
        except (TypeError, ValueError):
            return
        if w <= 2 or h <= 2:
            return
        if self._last_render_view_size == (w, h):
            return
        try:
            self._render_viewer()
        except (tk.TclError, AttributeError):
            pass

    def _update_zoom_label(self, zoom=None):
        lbl = self._zoom_label
        if lbl is None:
            return
        try:
            if self._viewer_fit:
                lbl.configure(text="Fit")
                return
            z = clamp_zoom(zoom if zoom is not None else self._viewer_zoom)
            lbl.configure(text=f"{z * 100:.0f}%")
        except (tk.TclError, AttributeError):
            pass

    def _show_zoom_bar(self):
        try:
            self._zoom_bar.grid(row=1, column=0, sticky="ew", padx=10, pady=(2, 0))
        except (tk.TclError, AttributeError):
            pass

    def _hide_zoom_bar(self):
        try:
            self._zoom_bar.grid_remove()
        except (tk.TclError, AttributeError):
            pass

    def _on_preview_resize(self, event):
        # Resize re-renders the current source (image or video frame) so the
        # viewport always fills the preview area at full quality.
        try:
            if self._video_kind is not None and getattr(self, "_viewer_mode", "image") in (
                "video",
                "timelapse",
            ):
                self._render_viewer()
            elif self._dream_pil is not None:
                self._update_preview()
            elif self._preview_pil is not None:
                self._render_viewer()
        except (tk.TclError, AttributeError):
            pass

    def _update_preview(self):
        if self._viewer_mode in ("video", "timelapse"):
            self._render_viewer()
            return
        pil_img = self._get_current_preview_pil()
        if pil_img is not None:
            # Keep _preview_pil in sync for callers that read it directly
            # (live preview, resize handler, tests).
            if self._dream_pil is None:
                self._preview_pil = pil_img
            self._render_viewer()

    def _show_output_preview(self, pil_img, reset_view=False):
        if pil_img is None:
            return
        # Live/dream/input images share one viewer: store the source then
        # render through the zoom/pan viewport at full quality.
        if self._viewer_mode in ("video", "timelapse"):
            # A new still arriving while a video is shown (e.g. live preview
            # of the next queued job) drops back to image mode.
            self._stop_video_playback()
            self._viewer_mode = "image"
            self._hide_video_bar()
        self._preview_pil = pil_img
        if reset_view:
            self._reset_viewer()
        else:
            # First image ever (or after teardown): center the viewport.
            if self._viewer_cx is None:
                sw, sh = pil_img.size
                self._viewer_cx, self._viewer_cy = sw / 2.0, sh / 2.0
        self._show_zoom_bar()
        self._render_viewer()

    # Zoom / pan event handlers

    def _viewer_label_size(self):
        try:
            w = self._preview_label.winfo_width()
            h = self._preview_label.winfo_height()
            if w > 2 and h > 2:
                return w, h
        except tk.TclError:
            pass
        return self._get_view_size()

    def _view_to_source(self, click_x, click_y):
        """Map a click in label pixels to normalized (fx, fy) in source 0..1."""
        src = self._get_viewer_source()
        if src is None:
            return None
        try:
            sw, sh = src.size
            lw, lh = self._viewer_label_size()
            vw, vh = self._get_view_size()
            # The rendered viewport is exactly (vw, vh) centered in the label.
            off_x = (lw - vw) // 2
            off_y = (lh - vh) // 2
            vx, vy = float(click_x) - off_x, float(click_y) - off_y
            if not (0 <= vx < vw and 0 <= vy < vh):
                return None
            view_w, view_h = vw, vh
            zoom = self._effective_zoom(src, view_w, view_h)
            cx, cy = self._ensure_viewer_center(src)
            view = compute_zoom_view(sw, sh, view_w, view_h, zoom, cx, cy)
            left, top, right, bottom = view["crop"]
            dw, dh = view["size"]
            ox, oy = view["offset"]
            # Click outside the pasted image (letterbox) maps to nothing.
            if not (ox <= vx < ox + dw and oy <= vy < oy + dh):
                return None
            sx = left + (vx - ox) * (right - left) / max(1, dw)
            sy = top + (vy - oy) * (bottom - top) / max(1, dh)
            return (clamp01(sx / sw), clamp01(sy / sh))
        except (TypeError, ValueError, ZeroDivisionError):
            return None

    def _zoom_at(self, factor, cursor_x=None, cursor_y=None):
        src = self._get_viewer_source()
        if src is None:
            return
        sw, sh = src.size
        vw, vh = self._get_view_size()
        old_zoom = self._effective_zoom(src, vw, vh)
        new_zoom = clamp_zoom(old_zoom * factor)
        if abs(new_zoom - old_zoom) < 1e-6:
            return
        # Zoom toward the cursor: keep the source point under the cursor fixed.
        cx, cy = self._ensure_viewer_center(src)
        if cursor_x is not None and cursor_y is not None:
            try:
                lw, lh = self._viewer_label_size()
                off_x = (lw - vw) // 2
                off_y = (lh - vh) // 2
                vx, vy = float(cursor_x) - off_x, float(cursor_y) - off_y
                if 0 <= vx < vw and 0 <= vy < vh:
                    # Source point currently under the cursor.
                    view = compute_zoom_view(sw, sh, vw, vh, old_zoom, cx, cy)
                    left, top, right, bottom = view["crop"]
                    dw, dh = view["size"]
                    ox, oy = view["offset"]
                    if ox <= vx < ox + dw and oy <= vy < oy + dh:
                        sx = left + (vx - ox) * (right - left) / max(1, dw)
                        sy = top + (vy - oy) * (bottom - top) / max(1, dh)
                        # New center so (sx, sy) stays under the cursor.
                        frac_x = (vx - ox) / max(1, dw) - 0.5
                        frac_y = (vy - oy) / max(1, dh) - 0.5
                        vis_w, vis_h = vw / new_zoom, vh / new_zoom
                        ncx = sx - frac_x * min(sw, vis_w)
                        ncy = sy - frac_y * min(sh, vis_h)
                        self._viewer_fit = False
                        self._viewer_zoom = new_zoom
                        self._viewer_cx = max(0.0, min(sw, ncx))
                        self._viewer_cy = max(0.0, min(sh, ncy))
                        self._render_viewer()
                        return
            except (TypeError, ValueError, ZeroDivisionError):
                pass
        self._viewer_fit = False
        self._viewer_zoom = new_zoom
        self._viewer_cx, self._viewer_cy = cx, cy
        self._render_viewer()

    def _on_zoom_in(self):
        self._show_zoom_bar()
        self._zoom_at(1.25)

    def _on_zoom_out(self):
        self._show_zoom_bar()
        self._zoom_at(1.0 / 1.25)

    def _on_zoom_fit(self):
        self._viewer_fit = True
        src = self._get_viewer_source()
        if src is not None:
            sw, sh = src.size
            self._viewer_cx, self._viewer_cy = sw / 2.0, sh / 2.0
        self._render_viewer()

    def _on_zoom_100(self):
        src = self._get_viewer_source()
        self._viewer_fit = False
        self._viewer_zoom = 1.0
        if src is not None:
            sw, sh = src.size
            self._viewer_cx, self._viewer_cy = sw / 2.0, sh / 2.0
        self._show_zoom_bar()
        self._render_viewer()

    def _on_viewer_wheel(self, event=None):
        if event is None:
            return None
        raw = getattr(event, "delta", 0)
        if raw:
            zoomin = raw > 0
        else:
            zoomin = getattr(event, "num", 5) == 4
        self._zoom_at(
            1.15 if zoomin else 1.0 / 1.15, getattr(event, "x", None), getattr(event, "y", None)
        )
        return "break"

    def _split_ratio_to_label_x(self, split_ratio=None):
        if split_ratio is None:
            split_ratio = self._reveal_split
        src = self._get_viewer_source()
        if src is None:
            return None
        try:
            sw, sh = src.size
            lw, lh = self._viewer_label_size()
            vw, vh = self._get_view_size()
            off_x = (lw - vw) // 2
            zoom = self._effective_zoom(src, vw, vh)
            cx, cy = self._ensure_viewer_center(src)
            view = compute_zoom_view(sw, sh, vw, vh, zoom, cx, cy)
            left, top, right, bottom = view["crop"]
            dw, dh = view["size"]
            ox, oy = view["offset"]
            crop_w = max(1, right - left)
            sx = float(split_ratio) * sw
            if sx < left or sx > right:
                return None
            vx = ox + (sx - left) * dw / crop_w
            return off_x + vx
        except (TypeError, ValueError, ZeroDivisionError, AttributeError):
            return None

    def _view_x_to_split_ratio(self, click_x):
        src = self._get_viewer_source()
        if src is None:
            return None
        try:
            sw, sh = src.size
            lw, lh = self._viewer_label_size()
            vw, vh = self._get_view_size()
            off_x = (lw - vw) // 2
            vx = float(click_x) - off_x
            zoom = self._effective_zoom(src, vw, vh)
            cx, cy = self._ensure_viewer_center(src)
            view = compute_zoom_view(sw, sh, vw, vh, zoom, cx, cy)
            left, top, right, bottom = view["crop"]
            dw, dh = view["size"]
            ox, oy = view["offset"]
            if dw <= 0 or sw <= 0:
                return 0.5
            clamped_vx = max(ox, min(ox + dw, vx))
            sx = left + (clamped_vx - ox) * (right - left) / max(1, dw)
            return clamp01(sx / sw)
        except (TypeError, ValueError, ZeroDivisionError, AttributeError):
            return 0.5

    def _is_near_reveal_line(self, x, tolerance=18):
        if (
            self._compare_mode != "Slide Reveal"
            or self._aligned_input_pil is None
            or self._dream_pil is None
        ):
            return False
        line_x = self._split_ratio_to_label_x()
        if line_x is None:
            return False
        return abs(x - line_x) <= tolerance

    def _on_viewer_motion(self, event=None):
        if event is None:
            return
        if getattr(self, "_reveal_dragging", False):
            try:
                self._preview_label.configure(cursor="sb_h_double_arrow")
            except tk.TclError:
                pass
            return
        if self._is_near_reveal_line(event.x):
            try:
                self._preview_label.configure(cursor="sb_h_double_arrow")
            except tk.TclError:
                pass
        else:
            try:
                self._preview_label.configure(cursor="")
            except tk.TclError:
                pass

    def _on_viewer_leave(self, event=None):
        if not getattr(self, "_reveal_dragging", False):
            try:
                self._preview_label.configure(cursor="")
            except tk.TclError:
                pass

    def _update_reveal_from_x(self, x):
        ratio = self._view_x_to_split_ratio(x)
        if ratio is None:
            return
        self._reveal_split = ratio
        pct = int(round(ratio * 100))
        if self._compare_mode == "Slide Reveal":
            try:
                self._preview_title.configure(text=f"Slide to Reveal ({pct}%)")
            except tk.TclError:
                pass
        self._render_viewer()

    def _on_viewer_press(self, event=None):
        if (
            event is not None
            and self._compare_mode == "Slide Reveal"
            and self._aligned_input_pil is not None
        ):
            line_x = self._split_ratio_to_label_x()
            if (line_x is not None and abs(event.x - line_x) <= 24) or self._viewer_fit:
                self._reveal_dragging = True
                self._update_reveal_from_x(event.x)
                return
        cx, cy = self._ensure_viewer_center(self._get_viewer_source() or self._preview_pil)
        self._viewer_drag = (event.x, event.y, cx, cy, False) if event else None

    def _on_viewer_drag(self, event=None):
        if getattr(self, "_reveal_dragging", False):
            if event is not None:
                self._update_reveal_from_x(event.x)
            return
        drag = self._viewer_drag
        src = self._get_viewer_source()
        if drag is None or event is None or src is None:
            return
        sx, sy, cx0, cy0, moved = drag
        dx, dy = float(event.x) - sx, float(event.y) - sy
        # Only treat as a pan once past the click threshold; a plain click
        # (or the 1px jitter Tk emits mid-click) must not leave "Fit" mode.
        moved = moved or (abs(dx) + abs(dy) > 3)
        self._viewer_drag = (sx, sy, cx0, cy0, moved)
        if not moved:
            return
        if self._viewer_fit:
            # Start panning from fit: switch to manual zoom at the fit scale.
            vw, vh = self._get_view_size()
            sw, sh = src.size
            self._viewer_fit = False
            self._viewer_zoom = fit_zoom_for(sw, sh, vw, vh)
        z = clamp_zoom(self._viewer_zoom)
        sw, sh = src.size
        vw, vh = self._get_view_size()
        vis_w, vis_h = vw / z, vh / z
        if vis_w < sw or vis_h < sh:
            self._viewer_cx = max(0.0, min(sw, cx0 - dx / z))
            self._viewer_cy = max(0.0, min(sh, cy0 - dy / z))
            self._render_viewer()

    def _on_viewer_release(self, event=None):
        if getattr(self, "_reveal_dragging", False):
            self._reveal_dragging = False
            self._viewer_drag = None
            if event is not None and not self._is_near_reveal_line(event.x):
                try:
                    self._preview_label.configure(cursor="")
                except tk.TclError:
                    pass
            return
        drag = self._viewer_drag
        self._viewer_drag = None
        moved = bool(drag and len(drag) > 4 and drag[4])
        if moved:
            return
        self._handle_viewer_click(event)

    def _handle_viewer_click(self, event=None):
        zoom_mode = self._mode_var.get() == "Dream Zoom"
        if zoom_mode and self._showing_zoom_focus:
            focus = self._view_to_source(event.x, event.y) if event is not None else None
            if focus is not None:
                self._set_zoom_focus(*focus)
                return True
        return False

    # In-window video preview. Two sources: a dreamed PNG sequence awaiting
    # Save Video, or a saved .mp4 / timelapse decoded via imageio. Both use the
    # same zoom/pan viewport as stills.

    def _hide_video_bar(self):
        self._stop_video_playback()
        try:
            self._video_bar.grid_remove()
        except (tk.TclError, AttributeError):
            pass

    def _show_video_bar(self):
        try:
            self._video_bar.grid(row=4, column=0, sticky="ew", padx=10, pady=(2, 2))
        except (tk.TclError, AttributeError):
            pass

    def _loader_generation(self):
        loader = getattr(self, "_video_loader", None)
        return loader.generation() if loader is not None else None

    def _on_video_probe_result(self, path, total, fps, token, generation):
        """Tk-thread handler for an async probe."""
        if self._is_closing:
            return
        if self._video_kind != "mp4" or self._video_mp4 != path:
            return
        if generation != self._loader_generation():
            return
        if token is not None and token != self._video_probe_token:
            return
        if total <= 0:
            # Undecodable header: keep the frame the loader already fetched as a
            # still, so a codec hiccup never blanks the preview.
            frame = self._video_cache.get(("mp4", path, 0))
            if frame is None:
                return
            self._teardown_video()
            self._viewer_mode = "image"
            self._show_output_preview(frame, reset_view=True)
            title = getattr(self, "_video_pending_title", None)
            if title:
                try:
                    self._preview_title.configure(text=title)
                except tk.TclError:
                    pass
            return
        self._video_total = int(total)
        self._video_fps = float(fps) if fps and float(fps) > 0 else 30.0
        self._video_pending_title = None
        self._show_video_frame(self._video_index)

    def _on_video_frame_result(self, kind, source, index, frame, token, generation):
        """Tk-thread handler for a decoded frame."""
        if self._is_closing or frame is None:
            return
        if generation != self._loader_generation():
            return
        if kind != self._video_kind:
            return
        if kind == "mp4" and source != self._video_mp4:
            return
        if kind == "frames":
            paths = self._video_paths or []
            if not (0 <= int(index) < len(paths)) or paths[int(index)] != source:
                return
        self._video_cache.put((kind, source, int(index)), frame)
        # Only display if the user is still on this frame; otherwise the decode
        # is just a cache warm-up for a scrub that has already moved on.
        if int(index) == int(self._video_requested_index):
            self._render_viewer()

    def _request_video_frame(self, index):
        """Ask for a frame and render it if already cached."""
        kind = self._video_kind
        if kind == "frames":
            paths = self._video_paths or []
            if not paths:
                return
            i = max(0, min(len(paths) - 1, int(index)))
            source = paths[i]
        elif kind == "mp4":
            mp4 = self._video_mp4
            if not mp4 or not os.path.isfile(mp4):
                return
            i = max(0, int(index))
            source = mp4
        else:
            return
        self._video_requested_index = i
        key = (kind, source, i)
        cached = self._video_cache.get(key)
        if cached is not None:
            self._render_viewer()
            return
        loader = getattr(self, "_video_loader", None)
        if loader is not None:
            loader.request_frame(kind, source, i)
            self._ensure_video_poll()

    def _ensure_video_poll(self):
        if self._video_poll_after_id is not None or self._is_closing:
            return
        try:
            self._video_poll_after_id = self.after(16, self._poll_video_loader)
        except tk.TclError:
            self._video_poll_after_id = None

    def _poll_video_loader(self):
        self._video_poll_after_id = None
        if self._is_closing:
            return
        loader = getattr(self, "_video_loader", None)
        if loader is None:
            return
        pending = loader.poll()
        if pending:
            self._video_poll_after_id = self.after(16, self._poll_video_loader)

    def _load_video_frame(self, index):
        """PIL frame for the active video source at *index* (or None).

        Cache-only: decoding happens on the loader thread so the Tk thread never
        blocks. Use _request_video_frame to ask for a frame that is not cached.
        """
        kind = self._video_kind
        if kind == "frames":
            paths = self._video_paths or []
            if not paths:
                return None
            i = max(0, min(len(paths) - 1, int(index)))
            return self._video_cache.get((kind, paths[i], i))
        if kind == "mp4":
            mp4 = self._video_mp4
            if not mp4:
                return None
            return self._video_cache.get((kind, mp4, max(0, int(index))))
        return None

    def _setup_frames_preview(self, frames_dir, fps=30.0, title=None):
        paths = list_saved_video_frames(frames_dir)
        if not paths:
            return False
        self._stop_video_playback()
        self._reset_video_loader()
        self._video_kind = "frames"
        self._video_paths = paths
        self._video_mp4 = None
        self._video_total = len(paths)
        try:
            self._video_fps = float(fps) if float(fps) > 0 else 30.0
        except (TypeError, ValueError):
            self._video_fps = 30.0
        self._video_index = len(paths) - 1
        self._video_cache = FrameCache()
        self._viewer_mode = "video"
        self._reset_viewer()
        self._show_zoom_bar()
        self._show_video_bar()
        self._show_video_frame(self._video_index)
        if title:
            try:
                self._preview_title.configure(text=title)
            except tk.TclError:
                pass
        return True

    def _setup_mp4_preview(self, mp4_path, title=None):
        """Show an .mp4 in the viewer. Probing and decoding happen off the Tk thread."""
        if not mp4_path or not os.path.isfile(mp4_path):
            return False
        self._stop_video_playback()
        self._reset_video_loader()
        self._video_kind = "mp4"
        self._video_mp4 = mp4_path
        self._video_paths = []
        self._video_total = 0
        self._video_fps = 30.0
        self._video_index = 0
        self._video_cache = FrameCache()
        self._viewer_mode = "video"
        self._reset_viewer()
        self._show_zoom_bar()
        self._show_video_bar()
        # Ask for frame 0 and the frame count now; the timeline becomes usable
        # once the probe lands, and the frame renders as soon as it decodes.
        self._video_requested_index = 0
        self._video_pending_title = title
        loader = getattr(self, "_video_loader", None)
        if loader is not None:
            self._video_probe_token += 1
            loader.request_frame("mp4", mp4_path, 0, self._video_probe_token)
            loader.request_probe(mp4_path, self._video_probe_token)
            self._ensure_video_poll()
        if title:
            try:
                self._preview_title.configure(text=title)
            except tk.TclError:
                pass
        return True

    def _reset_video_loader(self):
        """Point the loader at a new source: drop queued decodes and stale results."""
        loader = getattr(self, "_video_loader", None)
        if loader is not None:
            loader.invalidate()

    def _teardown_video(self):
        self._stop_video_playback()
        self._reset_video_loader()
        self._video_kind = None
        self._video_paths = []
        self._video_mp4 = None
        self._video_total = 0
        self._video_index = 0
        self._video_cache = FrameCache()
        if self._viewer_mode == "video":
            self._viewer_mode = "image"
        if self._video_bar is not None:
            try:
                self._video_bar.grid_remove()
            except tk.TclError:
                pass

    def _show_video_frame(self, index):
        total = int(self._video_total or 0)
        if total <= 0:
            return
        i = max(0, min(total - 1, int(index)))
        self._video_index = i
        self._request_video_frame(i)
        try:
            fps = float(self._video_fps or 30.0)
        except (TypeError, ValueError):
            fps = 30.0
        if self._video_pos_label is not None:
            try:
                self._video_pos_label.configure(text=format_frame_label(i, total, fps))
            except tk.TclError:
                pass
        self._video_slider_sync = True
        try:
            if self._video_slider is not None:
                self._video_slider.set(timeline_index_to_t(i, total))
        except tk.TclError:
            pass
        finally:
            self._video_slider_sync = False
        if self._video_play_btn is not None:
            try:
                self._video_play_btn.configure(state="normal" if total > 1 else "disabled")
            except tk.TclError:
                pass

    def _on_video_play_pause(self):
        if self._video_playing:
            self._stop_video_playback()
        else:
            self._start_video_playback()

    def _start_video_playback(self):
        total = int(self._video_total or 0)
        if total <= 1:
            return
        # Restart from the start when pressing play at the end.
        if int(self._video_index) >= total - 1:
            self._video_index = 0
        self._video_playing = True
        if self._video_play_btn is not None:
            try:
                self._video_play_btn.configure(text="⏸ Pause")
            except tk.TclError:
                pass
        self._schedule_video_tick()

    def _stop_video_playback(self):
        self._video_playing = False
        aid = self._video_after_id
        if aid is not None:
            try:
                self.after_cancel(aid)
            except tk.TclError:
                pass
            self._video_after_id = None
        if self._video_play_btn is not None:
            try:
                self._video_play_btn.configure(text="▶ Play")
            except tk.TclError:
                pass

    def _schedule_video_tick(self):
        try:
            fps = float(self._video_fps or 30.0)
        except (TypeError, ValueError):
            fps = 30.0
        delay = int(round(1000.0 / fps)) if fps > 0 else 33
        delay = max(16, min(2000, delay))
        _pending = self._video_after_id
        if _pending is not None:
            try:
                self.after_cancel(_pending)
            except tk.TclError:
                pass
        try:
            self._video_after_id = self.after(delay, self._video_tick)
        except tk.TclError:
            pass

    def _video_tick(self):
        self._video_after_id = None
        if not self._video_playing:
            return
        if self._is_closing:
            self._video_playing = False
            return
        total = int(self._video_total or 0)
        if total <= 0:
            self._stop_video_playback()
            return
        nxt = int(self._video_index) + 1
        if nxt >= total:
            # Loop the preview so short clips can be judged without replay.
            nxt = 0
        self._show_video_frame(nxt)
        if self._video_playing:
            self._schedule_video_tick()

    def _on_video_slider_change(self, value):
        if self._video_slider_sync:
            return
        total = int(self._video_total or 0)
        if total <= 0:
            return
        try:
            t = float(value)
        except (TypeError, ValueError):
            return
        self._show_video_frame(timeline_t_to_index(t, total))

    def _on_view_image_toggle(self):
        # Back to the still image when both a still and a timelapse exist.
        self._viewer_mode = "image"
        self._stop_video_playback()
        self._hide_video_bar()
        try:
            self._view_image_btn.grid_remove()
            if self._timelapse_mp4:
                self._view_timelapse_btn.grid()
        except tk.TclError:
            pass
        self._reset_viewer()
        self._update_preview()
        # Restore compare controls now that a still is shown again (they are
        # hidden while the timelapse video plays).
        try:
            if self._aligned_input_pil is not None:
                self._compare_frame.grid()
            self._preview_title.configure(text="Dream Result")
        except tk.TclError:
            pass

    def _on_view_timelapse_toggle(self):
        mp4 = self._timelapse_mp4
        if not mp4 or not os.path.isfile(mp4):
            return
        self._stop_video_playback()
        self._reset_video_loader()
        self._video_kind = "mp4"
        self._video_mp4 = mp4
        self._video_paths = []
        self._video_total = 0
        self._video_fps = 30.0
        self._video_index = 0
        self._video_cache = FrameCache()
        self._video_requested_index = 0
        self._video_pending_title = None
        self._viewer_mode = "timelapse"
        self._reset_viewer()
        self._show_zoom_bar()
        self._show_video_bar()
        self._video_probe_token += 1
        loader = getattr(self, "_video_loader", None)
        if loader is None:
            return
        loader.request_frame("mp4", mp4, 0, self._video_probe_token)
        loader.request_probe(mp4, self._video_probe_token)
        self._ensure_video_poll()
        # Compare controls only apply to stills; hide them so they can't be
        # clicked into a no-op while the timelapse plays.
        try:
            self._compare_frame.grid_remove()
            self._view_timelapse_btn.grid_remove()
            self._view_image_btn.grid()
            self._preview_title.configure(text="Timelapse Preview")
        except tk.TclError:
            pass

    def _set_compare_mode(self, mode):
        self._compare_mode = mode
        if self._compare_seg is not None:
            self._compare_seg.set(mode)

        if mode == "Slide Reveal":
            pct = int(round(self._reveal_split * 100))
            self._preview_title.configure(text=f"Slide to Reveal ({pct}%)")
        else:
            if mode == "Original":
                self._preview_title.configure(text="Original Image")
            else:
                self._preview_title.configure(text="Dream Result")

        self._update_preview()

    def _on_compare_mode_change(self, value):
        self._set_compare_mode(value)

    def _toggle_compare(self, event=None):
        if self._dream_pil is None or self._aligned_input_pil is None:
            return
        if self._compare_mode == "Dream":
            self._set_compare_mode("Original")
        else:
            self._set_compare_mode("Dream")

    def _on_space_pressed(self, event=None):
        w = getattr(event, "widget", None)
        if w is not None:
            try:
                w_class = w.winfo_class()
                if w_class in ("Entry", "Text", "TEntry", "Combobox", "TCombobox", "Listbox"):
                    return
            except tk.TclError:
                pass
        if self._dream_pil is not None and self._aligned_input_pil is not None:
            self._toggle_compare()

    def _on_arrow_nudge(self, delta, event=None):
        w = getattr(event, "widget", None)
        if w is not None:
            try:
                w_class = w.winfo_class()
                if w_class in ("Entry", "Text", "TEntry", "Combobox", "TCombobox", "Listbox"):
                    return
            except tk.TclError:
                pass
        if self._active_hover_nudge is not None:
            self._active_hover_nudge(delta)
            return "break"

    def _on_window_leave(self, event=None):
        try:
            x = self.winfo_pointerx() - self.winfo_rootx()
            y = self.winfo_pointery() - self.winfo_rooty()
            if not (0 <= x < self.winfo_width() and 0 <= y < self.winfo_height()):
                self._active_hover_nudge = None
        except tk.TclError:
            self._active_hover_nudge = None

    # Log helpers

    def _log_msg(self, text, color=None):
        if self._is_closing:
            return
        try:
            self._log.configure(state="normal")
            if color:
                tag = f"col_{color.replace('#', '')}"
                self._log.tag_config(tag, foreground=color)
                self._log.insert("end", text, tag)
            else:
                self._log.insert("end", text)
            line_count = int(self._log.index("end-1c").split(".")[0])
            if line_count > MAX_LOG_LINES:
                self._log.delete("1.0", f"{line_count - MAX_LOG_LINES}.0")
            self._log.see("end")
            self._log.configure(state="disabled")
        except tk.TclError:
            pass

    def _clear_log(self):
        if self._is_closing:
            return
        try:
            self._log.configure(state="normal")
            self._log.delete("1.0", "end")
            self._log.configure(state="disabled")
        except tk.TclError:
            pass

    # Run / Stop / Queue

    def _create_job_from_inputs(self, image_path=None):
        path = image_path if image_path is not None else self._image_path.get().strip()
        path = path.strip().strip("\"'")
        if not path:
            messagebox.showwarning("No media", "Please select an input image or video first.")
            return None
        if not os.path.isfile(path):
            messagebox.showerror("File not found", f"Cannot find:\n{path}")
            return None
        if Path(path).suffix.lower() not in MEDIA_EXTENSIONS:
            messagebox.showwarning(
                "Unsupported file",
                f"Please select an image or video file.\n\nUnsupported: {path}",
            )
            return None

        guidance_mode = self._guidance_mode_var.get()
        guide_path = None
        target_class = None
        if guidance_mode == "Guide Image":
            guide_path = self._guide_path.get().strip().strip("\"'")
            if not guide_path or not os.path.isfile(guide_path):
                messagebox.showwarning("Missing guide", "Please select a valid guide image.")
                return None
        elif guidance_mode == "Target Class":
            target_class = self._target_class_var.get().strip()
            if not target_class:
                messagebox.showwarning(
                    "Missing Target Class",
                    "Please select or search for an ImageNet target category.",
                )
                return None
            if lookup_category(target_class) is None:
                messagebox.showwarning(
                    "Unknown Target Class",
                    f"'{target_class}' is not a recognized ImageNet category.\n\n"
                    "Pick one from the search box, list, or dropdown.",
                )
                return None

        out_dir = resolve_out_dir(self._out_dir.get())
        try:
            os.makedirs(out_dir, exist_ok=True)
        except OSError as exc:
            messagebox.showerror(
                "Invalid Output Directory",
                f"Cannot create output directory:\n{out_dir}\n\n{exc}",
            )
            return None

        model_key = MODEL_BY_LABEL.get(self._model_var.get(), DEFAULT_MODEL_KEY)
        custom_layers = self._custom_layers_var.get().strip()
        if custom_layers:
            layer_spec = custom_layers
        else:
            if guidance_mode == "Target Class":
                layer_spec = ""
            else:
                layer_value_map = _layer_value_map(model_key)
                layer_spec = layer_value_map.get(self._layer_var.get(), self._layer_var.get())

        ui_mode = self._mode_var.get()
        mode = ui_mode
        # Automatically match mode if input file format requires it
        is_video_input = Path(path).suffix.lower() in VIDEO_EXTENSIONS
        if is_video_input and mode in ("Single Image", "Dream Zoom"):
            mode = "Dream Video"
        elif not is_video_input and mode == "Dream Video":
            mode = "Single Image"

        iterations = self._iterations.get()
        if mode in ("Dream Zoom", "Dream Video") and ui_mode == "Single Image":
            iterations = max(1, int(round(iterations / 10.0)))
        elif mode == "Single Image" and ui_mode in ("Dream Zoom", "Dream Video"):
            saved_iters = self._saved_single_image_iters
            iterations = (
                saved_iters
                if saved_iters is not None and saved_iters >= 1
                else max(1, iterations * 10)
            )

        self._job_counter += 1
        job_id = self._job_counter

        custom_name = self._custom_name.get().strip() or None
        # Each GUI job is its own single-image subprocess, so the CLI's
        # multi-image disambiguation never applies: two queued jobs with the
        # same base name would overwrite each other. Make it unique with the id.
        base_name = custom_name or Path(path).stem
        if any(
            (j.custom_name or Path(j.image_path).stem).lower() == base_name.lower()
            for j in self._job_queue
        ):
            custom_name = f"{base_name}_{job_id}"

        # Reject a name the CLI would refuse mid-run (invalid Windows chars,
        # reserved device name, trailing dot/space, or over the length cap) so
        # the user is warned before a job is queued instead of watching it fail.
        if custom_name is not None:
            name_error = filename_stem_error(
                custom_name, what="Filename stem", max_len=MAX_NAME_LEN
            )
            if name_error:
                messagebox.showwarning("Invalid Filename", name_error.removeprefix("Error: "))
                return None

        # Per-job frames dir (pid included so two GUI instances sharing one
        # out_dir stay isolated). Single Image uses it for frames/timelapse;
        # Zoom/Video always use it for the separated Save Video step.
        frames_dir = None
        if (
            mode == "Single Image" and (self._save_frames.get() or self._timelapse.get())
        ) or mode in ("Dream Zoom", "Dream Video"):
            frames_dir = str(Path(out_dir) / f"frames_job{job_id}_p{os.getpid()}")

        job = GenerationJob(
            id=job_id,
            image_path=path,
            model_key=model_key,
            layer_spec=layer_spec,
            iterations=iterations,
            octaves=self._octaves.get(),
            octave_scale=self._octave_scale.get(),
            lr=self._lr.get(),
            size=self._size.get(),
            jitter=self._jitter.get(),
            out_dir=out_dir,
            mode=cast("Literal['Single Image', 'Dream Zoom', 'Dream Video']", mode),
            timelapse=self._timelapse.get(),
            zoom_frames=self._zoom_frames.get(),
            zoom_scale=self._zoom_scale.get(),
            zoom_rot=self._zoom_rot.get(),
            zoom_fps=self._zoom_fps.get(),
            zoom_cx=float(self._zoom_cx.get()),
            zoom_cy=float(self._zoom_cy.get()),
            video_blend=self._video_blend.get(),
            video_quality=cast(
                "Literal['Small', 'Balanced', 'Lossless', 'Custom']", self._video_quality_var.get()
            ),
            video_crf=self._video_crf.get(),
            guidance_mode=cast("Literal['Unguided', 'Guide Image', 'Target Class']", guidance_mode),
            guide_path=guide_path,
            target_class=target_class,
            custom_name=custom_name,
            save_frames=self._save_frames.get(),
            no_cuda=self._no_cuda.get(),
            status="Pending",
            frames_dir=frames_dir,
            seed=getattr(self, "_seed", None),
        )
        return job

    def _add_to_queue(self, job, auto_start=True):
        self._job_queue.append(job)
        self._update_queue_badge()
        self._refresh_queue_ui()
        if auto_start and not self._queue_running:
            self._start_queue()

    def _on_add_to_queue_click(self):
        self._reset_run_button_if_resuming()
        job = self._create_job_from_inputs()
        if not job:
            return
        is_running = self._proc is not None and self._proc.poll() is None
        self._add_to_queue(job, auto_start=True)
        if is_running:
            self._log_msg(
                f"\n[Queue] Added Job #{job.id} ({Path(job.image_path).name}) to queue (will execute in order).\n",
                color=ACCENT,
            )
        else:
            self._log_msg(
                f"\n[Queue] Added Job #{job.id} ({Path(job.image_path).name}) to queue.\n",
                color=ACCENT,
            )
        badge_text = self._get_queue_badge_text()
        self._bottom_tab_seg.set(badge_text)
        self._on_bottom_tab_change(badge_text)

    def _on_main_resume_click(self):
        if (
            self._resumable_job is not None
            and getattr(self._resumable_job, "status", None) == "Stopped"
        ):
            self._resume_job(self._resumable_job)

    def _update_resume_button_visibility(self):
        btn = getattr(self, "_main_resume_btn", None)
        if btn is None:
            return
        job = self._resumable_job
        if job is not None and getattr(job, "status", None) == "Stopped":
            frames_dir = getattr(job, "frames_dir", None)
            cnt = (
                count_saved_video_frames(frames_dir)
                if (frames_dir and os.path.isdir(frames_dir))
                else 0
            )
            if cnt > 0:
                meta = read_video_frames_meta(frames_dir) if frames_dir else {}
                tot = meta.get("total_frames") or getattr(job, "zoom_frames", None)
                tot_txt = f"/{tot}" if tot else ""
                btn.grid()
                btn.configure(state="normal", text="\u25b6 Resume")
                Tooltip(
                    btn,
                    f"Resume generation from frame {cnt + 1}{tot_txt}.\n"
                    "Or click '▶ Dream' to start fresh from frame 1.",
                )
                try:
                    Tooltip(
                        self._run_btn,
                        "Start fresh from frame 1 (or click '▶ Resume' to continue where you left off).",
                    )
                except (tk.TclError, AttributeError):
                    pass
                return
        btn.grid_remove()
        try:
            Tooltip(self._run_btn, "Start dreaming immediately with current settings.")
        except (tk.TclError, AttributeError):
            pass

    def _reset_run_button_if_resuming(self):
        if self._resumable_job is not None:
            self._resumable_job = None
            self._update_resume_button_visibility()

    def _resume_job(self, job):
        if not job:
            return
        job.resume = True
        job.status = "Pending"
        self._resumable_job = None
        self._update_resume_button_visibility()
        if job not in self._job_queue:
            self._job_queue.append(job)
        self._update_queue_badge()
        self._refresh_queue_ui()
        if not self._queue_running:
            self._start_queue()

    def _run(self):
        if self._proc is not None and self._proc.poll() is None:
            # A generation is in flight (running *or* paused after a Stop):
            # never let Dream hijack it. Snapshot the current settings behind
            # it; auto_start ensures the queue runs once the active process
            # exits, instead of silently dropping the current input.
            job = self._create_job_from_inputs()
            if job:
                self._add_to_queue(job, auto_start=True)
                self._log_msg(
                    f"\n[Queue] Queued Job #{job.id} ({Path(job.image_path).name}) while active run completes.\n",
                    color=ACCENT,
                )
            return

        # If a paused/stopped run was pending, clicking "▶ Dream" always starts
        # fresh from frame 1 (merging "start fresh" with Dream as requested).
        self._resumable_job = None
        self._update_resume_button_visibility()

        pending_jobs = [j for j in self._job_queue if j.status == "Pending"]
        if pending_jobs and not self._queue_running:
            self._queue_running = True
            self._process_next_job()
            return

        job = self._create_job_from_inputs()
        if not job:
            return
        self._add_to_queue(job, auto_start=True)

    def _cancel_next_job_timer(self):
        """Cancel any pending next-job transition timer."""
        timer_id = self._next_job_after_id
        if timer_id is not None:
            try:
                self.after_cancel(timer_id)
            except (ValueError, tk.TclError):
                pass
            self._next_job_after_id = None

    def _schedule_next_job(self, delay_ms=100):
        """Schedule the next job processing run, canceling any prior timer."""
        self._cancel_next_job_timer()
        if not self._is_closing and self._queue_running:
            self._next_job_after_id = self.after(delay_ms, self._process_next_job)

    def _start_queue(self):
        self._cancel_next_job_timer()
        self._queue_running = True
        self._process_next_job()

    def _process_next_job(self):
        if self._is_closing or not self._queue_running:
            return
        if self._proc is not None:
            # Active process is running or its exit sentinel is still being processed
            self._schedule_next_job(100)
            return
        self._cancel_next_job_timer()

        pending = [j for j in self._job_queue if j.status == "Pending"]
        if not pending:
            self._queue_running = False
            self._current_job = None
            self._run_btn.configure(state="normal")
            self._stop_btn.configure(state="disabled", text_color=TEXT_DIM)
            self._update_queue_badge()
            self._refresh_queue_ui()
            return

        job = pending[0]
        self._current_job = job
        job.status = "Running"
        self._update_queue_badge()
        self._refresh_queue_ui()
        self._execute_job(job)

    @staticmethod
    def _file_mtime(path):
        """st_mtime_ns for *path*, or None if it can't be stat'ed."""
        try:
            return os.stat(path).st_mtime_ns
        except OSError:
            return None

    def _load_input_image_for_job(self, job):
        """Ensure self._input_pil matches the job's input image for output comparison."""
        path = getattr(job, "image_path", None) if job else None
        key = os.path.normcase(os.path.abspath(path)) if path else None
        # Keep the already-loaded copy for this exact input: re-decoding is
        # wasteful, and it preserves the before/after comparison if the input
        # is deleted mid-run. Only a changed file (new mtime) forces a reload.
        if self._input_pil is not None and key is not None and self._input_pil_path == key:
            mtime = self._file_mtime(path)
            if mtime is None or mtime == self._input_pil_mtime:
                return
        if not path or not os.path.isfile(path):
            self._input_pil = None
            self._input_pil_path = None
            self._input_pil_mtime = None
            return
        mtime = self._file_mtime(path)
        pil = None
        if Path(path).suffix.lower() in VIDEO_EXTENSIONS:
            try:
                first_frame = iio.imread(path, index=0)
                pil = Image.fromarray(first_frame).convert("RGB")
            except (OSError, ValueError):
                try:
                    reader = imageio.get_reader(path)
                    try:
                        pil = Image.fromarray(reader.get_data(0)).convert("RGB")
                    finally:
                        reader.close()
                except (OSError, ValueError):
                    pil = None
        else:
            try:
                with Image.open(path) as _raw:
                    pil = ImageOps.exif_transpose(_raw).convert("RGB")
            except (OSError, ValueError):
                pil = None
        self._input_pil = pil
        self._input_pil_path = key if pil is not None else None
        self._input_pil_mtime = mtime if pil is not None else None

    def _execute_job(self, job):
        self._out_dir.set(job.out_dir)
        # Clean up the previous job's preview temp before overwriting.
        if self._preview_tmp_path:
            for p in (self._preview_tmp_path, f"{self._preview_tmp_path}.tmp"):
                if os.path.isfile(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
        # Unique per job + process so queued jobs and concurrent GUI
        # instances can't clobber each other's live preview. The prefix
        # stays "_gui_preview" so _newest_matching() still excludes it.
        preview_tmp_path = str(Path(job.out_dir) / f"_gui_preview_p{os.getpid()}_j{job.id}.png")
        self._preview_tmp_path = preview_tmp_path

        # Backfill per-job frames dir for jobs created before it existed
        # (e.g. in tests) so they still isolate instead of sharing out/frames.
        if not getattr(job, "frames_dir", None) and (
            (job.mode == "Single Image" and (job.save_frames or job.timelapse))
            or job.mode in ("Dream Zoom", "Dream Video")
        ):
            job.frames_dir = str(Path(job.out_dir) / f"frames_job{job.id}_p{os.getpid()}")
        cmd = job.build_cmd(preview_tmp_path=preview_tmp_path)
        job.cmd = cmd

        idx = sum(
            1 for j in self._job_queue if j.status in ("Done", "Running", "Failed", "Stopped")
        )
        total = len(self._job_queue)
        self._log_msg(
            f"\n[Queue {idx}/{total}] Job #{job.id}: {Path(job.image_path).name}\n$ {' '.join(cmd)}\n\n",
            color=TEXT_DIM,
        )

        self._run_btn.configure(state="disabled")
        if self._main_resume_btn is not None:
            try:
                self._main_resume_btn.configure(state="disabled")
            except tk.TclError:
                pass
        self._stop_btn.configure(state="normal", text_color=TEXT_MAIN)
        self._progress.set(0)
        self._status_label.configure(text=f"Job #{job.id} starting\u2026", text_color=WARNING)
        self._open_dir_btn.grid_remove()
        try:
            self._view_image_btn.grid_remove()
            self._view_timelapse_btn.grid_remove()
        except tk.TclError:
            pass
        self._teardown_video()
        self._compare_frame.grid_remove()

        self._load_input_image_for_job(job)
        self._tracker = ProgressTracker()
        self._expected_output = None
        self._output_path = None
        self._aborted = False
        self._dream_pil = None
        self._aligned_input_pil = None
        self._compare_mode = "Dream"
        self._showing_zoom_focus = False
        self._timelapse_mp4 = None
        self._viewer_mode = "image"
        self._reset_viewer()
        self._is_video_output = job.mode in ("Dream Zoom", "Dream Video")

        # Track this run's frames dir so an aborted run cleans up its partial
        # frames. Each job has its own frames_dir, so clear_stale_frames() for
        # one job cannot delete another's frames.
        track_frames = job.mode in ("Dream Zoom", "Dream Video") or (
            job.mode == "Single Image" and (job.save_frames or job.timelapse)
        )
        if track_frames and getattr(job, "frames_dir", None):
            frames_dir = Path(job.frames_dir)
            self._frames_dir = str(frames_dir)
            self._frames_preexisting = frames_dir.is_dir()
        else:
            self._frames_dir = None
            self._frames_preexisting = False

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                encoding="utf-8",
                errors="backslashreplace",
                cwd=str(ROOT),
                creationflags=(
                    getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
                ),
                # Own process group on POSIX so terminate_process() can
                # signal the whole tree (deepdream + short-lived ffmpeg).
                start_new_session=(sys.platform != "win32"),
            )
        except OSError as exc:
            self._log_msg(
                f"\nFailed to start deepdream for Job #{job.id}: {exc}\n", color=ERROR_COL
            )
            job.status = "Failed"
            self._status_label.configure(text="Error", text_color=ERROR_COL)
            self._progress.set(0)
            self._proc = None
            self._update_queue_badge()
            self._refresh_queue_ui()
            if self._queue_running and any(j.status == "Pending" for j in self._job_queue):
                self._schedule_next_job(100)
            else:
                self._queue_running = False
                self._current_job = None
                self._run_btn.configure(state="normal")
                self._stop_btn.configure(state="disabled", text_color=TEXT_DIM)
            return

        self._proc = proc
        self._ensure_poll()
        threading.Thread(target=self._reader, daemon=True, args=(proc, job)).start()

    def _stop(self):
        self._queue_running = False
        self._cancel_next_job_timer()
        encode_proc = self._encode_proc
        if self._proc and self._proc.poll() is None:
            self._aborted = True
            job = self._current_job
            if job is not None and getattr(job, "id", None) is not None:
                self._aborted_jobs.add(job.id)
            terminate_process(self._proc)
            self._log_msg("\n[Queue stopped by user]\n", color=WARNING)
        elif encode_proc and encode_proc.poll() is None:
            terminate_process(encode_proc)
            self._log_msg("\n[Video export stopped by user]\n", color=WARNING)
        elif self._proc is not None:
            # Process already exited but its reader-thread sentinel is still
            # queued, so don't paint "Stopped" now. The pending
            # _on_process_finished() owns the final status.
            self._aborted = True
            job = self._current_job
            if job is not None and getattr(job, "id", None) is not None:
                self._aborted_jobs.add(job.id)
            self._log_msg("\n[Queue stopped by user]\n", color=WARNING)
        else:
            self._status_label.configure(text="Stopped", text_color=WARNING)
            self._stop_btn.configure(state="disabled", text_color=TEXT_DIM)
            self._run_btn.configure(state="normal")
            self._refresh_queue_ui()

    def _resume_queue(self):
        if not self._queue_running:
            self._queue_running = True
            self._process_next_job()
            self._refresh_queue_ui()

    def _reader(self, proc, job):
        """Background thread: read subprocess output into the log queue.
        Binds to its own proc and on exit pushes that process's id in the
        completion marker, so a newer run is never blamed (PROC_DONE_PREFIX)."""
        try:
            for line in proc.stdout:
                self._log_queue.put(line)
        finally:
            try:
                if proc.stdout:
                    proc.stdout.close()
            except OSError:
                pass
            try:
                proc.wait()
            finally:
                self._log_queue.put(f"{PROC_DONE_PREFIX}{id(proc)}:{getattr(job, 'id', 0)}")

    def _ensure_poll(self):
        """(Re)start the periodic drain if it is not already scheduled."""
        poll_after_id = self._poll_after_id
        if poll_after_id is not None:
            try:
                self.after_cancel(poll_after_id)
            except tk.TclError:
                pass
            self._poll_after_id = None
        self._poll_log()

    # Periodic log drain

    def _poll_log(self):
        if self._is_closing:
            return
        handled_finish = False
        # A burst of dreams can queue hundreds of lines between ticks. Cap the
        # work per tick so the UI keeps responding, and carry the newest preview
        # path instead of decoding every queued one.
        budget = MAX_LINES_PER_TICK
        latest_preview = None
        while budget > 0:
            try:
                line = self._log_queue.get_nowait()
            except queue.Empty:
                break
            budget -= 1

            if isinstance(line, str) and line.startswith(PROC_DONE_PREFIX):
                # "<prefix><proc_id>:<job_id>" marks a reader-thread exit.
                try:
                    proc_id_str, job_id_str = line[len(PROC_DONE_PREFIX) :].split(":", 1)
                    proc_id = int(proc_id_str)
                except (ValueError, IndexError):
                    proc_id = -1
                    job_id_str = "0"
                if self._proc is not None and id(self._proc) == proc_id:
                    self._on_process_finished()
                    handled_finish = True
                else:
                    # A stale exit from a subprocess that has since been replaced
                    # (Stop, then the next job started). Attribute it to its own
                    # job and move on; never disturb the currently running one.
                    try:
                        job_id = int(job_id_str)
                    except (ValueError, IndexError):
                        job_id = None
                    self._mark_stale_job_stopped(job_id)
                    curr_job_id = getattr(self._current_job, "id", None)
                    if curr_job_id is None or curr_job_id not in self._aborted_jobs:
                        self._aborted = False
                break

            # Structured events: never shown in the log, and they carry the
            # progress/output information the GUI needs.
            event = parse_event(line)
            if event is not None:
                if self._apply_event(event):
                    latest_preview = event.get("path")
                continue

            # Legacy "[PREVIEW] path" marker, for a CLI without events.
            if line.startswith("[PREVIEW]"):
                latest_preview = line[len("[PREVIEW]") :].strip()
                continue

            self._log_msg(line)
            frac, status = self._tracker.update(line)
            if status:
                self._status_label.configure(text=status, text_color=WARNING)
            if frac is not None:
                self._progress.set(frac)
            path = extract_output_path(line)
            if path:
                self._expected_output = path
            if line.strip().startswith("Done in"):
                self._status_label.configure(text="Done \u2713", text_color=SUCCESS)
                self._progress.set(1.0)

        if latest_preview:
            self._apply_preview_path(latest_preview)

        self._drain_encode_queue()

        if self._video_save_in_progress or not self._encode_queue.empty():
            # A video export can outlive the dream subprocess that started it,
            # and its events are drained only here, so keep polling even on the
            # tick that handles a process exit.
            self._poll_after_id = self.after(80, self._poll_log)
        elif not handled_finish and (self._proc is not None or not self._log_queue.empty()):
            self._poll_after_id = self.after(80, self._poll_log)
        else:
            self._poll_after_id = None

    def _apply_event(self, event):
        """Apply one structured CLI event on the Tk thread.

        Returns True for events that carry a preview path. Human log lines are
        still emitted alongside these, so the log stays complete.
        """
        kind = event.get("event")
        if kind == EV_PREVIEW:
            return True
        if kind == EV_DONE:
            path = event.get("path")
            if path:
                self._expected_output = path
            self._status_label.configure(text="Done ✓", text_color=SUCCESS)
            self._progress.set(1.0)
            return False
        frac, status = self._tracker.update_event(event)
        if status:
            self._status_label.configure(text=status, text_color=WARNING)
        if frac is not None:
            self._progress.set(frac)
        return False

    def _apply_preview_path(self, prev_path):
        """Show the newest live intermediate preview, if it is still readable."""
        if not prev_path or not os.path.isfile(prev_path):
            return
        try:
            with Image.open(prev_path) as img:
                copy_img = img.convert("RGB")
            self._show_output_preview(copy_img)
            self._showing_zoom_focus = False
            self._preview_title.configure(text="Dreaming… (Live Intermediate)")
        except (OSError, ValueError):
            pass

    def _drain_encode_queue(self) -> bool:
        """Apply queued video-encode events on the Tk thread."""
        finished = False
        while True:
            try:
                event = self._encode_queue.get_nowait()
            except queue.Empty:
                break
            if not isinstance(event, tuple) or not event:
                continue
            kind = event[0]
            if kind == "line" and len(event) >= 3:
                self._handle_encode_line(event[1], event[2])
            elif kind == "done" and len(event) >= 5:
                finished = True
                self._finish_video_encode(event[1], event[2], event[3], event[4])
        return finished

    def _mark_stale_job_stopped(self, job_id):
        """Mark a job whose subprocess was replaced as Stopped (idempotent)."""
        if job_id is None:
            return
        self._aborted_jobs.discard(job_id)
        for j in self._job_queue:
            if j.id == job_id and j.status == "Running":
                j.status = "Stopped"
                break
        self._update_queue_badge()
        self._refresh_queue_ui()

    def _recover_partial_video_frames(self, job):
        """Load a stopped-but-resumable video when frames were already saved.
        Returns True when saved frames were found and shown."""
        if not (
            job
            and job.mode in ("Dream Zoom", "Dream Video")
            and self._frames_dir
            and os.path.isdir(self._frames_dir)
        ):
            return False
        cnt = count_saved_video_frames(self._frames_dir)
        if cnt <= 0:
            return False
        # Status must read "Stopped" before the viewer loads: it decides the
        # "Paused" title and marks the job resumable.
        job.status = "Stopped"
        meta = read_video_frames_meta(self._frames_dir)
        tot = meta.get("total_frames") or getattr(job, "zoom_frames", None)
        tot_txt = f"/{tot}" if tot else ""
        stop_status = f"Stopped ({cnt}{tot_txt} frames ready)"
        self._status_label.configure(text=stop_status, text_color=WARNING)
        self._progress.set(cnt / float(tot) if tot else 0.5)
        self._load_video_frames_ready(job, self._frames_dir, status_text=stop_status)
        return True

    def _on_process_finished(self):
        """Handle the subprocess-exit sentinel."""
        rc = self._proc.returncode if self._proc else -1
        job = self._current_job
        is_aborted = self._aborted or (job is not None and job.id in self._aborted_jobs)
        if is_aborted and rc == 0:
            # Stop raced a run that had already completed successfully: the
            # output on disk is whole, so honor the completed run instead of
            # marking it Stopped and hiding its output.
            is_aborted = False
        if is_aborted:
            if job:
                job.status = "Stopped"
            if not self._recover_partial_video_frames(job):
                self._status_label.configure(text="Stopped", text_color=WARNING)
                self._progress.set(0)
                self._cleanup_partial_frames()
        elif rc == 0:
            if job:
                job.status = "Done"
                job.resume = False
            if self._resumable_job is job:
                self._resumable_job = None
                self._update_resume_button_visibility()
            self._status_label.configure(text="Done \u2713", text_color=SUCCESS)
            self._progress.set(1.0)
            self._load_output_image(job=job)
        else:
            if job:
                job.status = "Failed"
            if not self._recover_partial_video_frames(job):
                self._status_label.configure(text="Error", text_color=ERROR_COL)
                self._progress.set(0)
                self._cleanup_partial_frames()

        self._cleanup_preview_temps()

        if job is not None:
            # Kill-style exits (Stop / close) skip deepdream.py's own temp
            # cleanup, so sweep them here now that the child is reaped.
            self._cleanup_job_temps(job.out_dir, getattr(self._proc, "pid", None))
            self._aborted_jobs.discard(job.id)
        self._aborted = bool(self._aborted_jobs)
        self._proc = None
        self._update_queue_badge()
        self._refresh_queue_ui()

        if self._queue_running and any(j.status == "Pending" for j in self._job_queue):
            self._schedule_next_job(100)
            return

        self._queue_running = False
        self._current_job = None
        self._run_btn.configure(state="normal")
        self._stop_btn.configure(state="disabled", text_color=TEXT_DIM)
        self._drop_zone.configure(cursor="hand2")
        pending_count = sum(1 for j in self._job_queue if j.status == "Pending")
        total_processed = sum(
            1 for j in self._job_queue if j.status in ("Done", "Failed", "Stopped")
        )
        if pending_count > 0 and total_processed > 0:
            done_count = sum(1 for j in self._job_queue if j.status == "Done")
            fail_count = sum(1 for j in self._job_queue if j.status in ("Failed", "Stopped"))
            self._log_msg(
                f"\n[Queue Paused: {done_count} completed, {fail_count} failed/stopped, {pending_count} pending]\n",
                color=WARNING,
            )
        elif pending_count == 0 and total_processed > 1:
            done_count = sum(1 for j in self._job_queue if j.status == "Done")
            fail_count = sum(1 for j in self._job_queue if j.status in ("Failed", "Stopped"))
            if fail_count > 0:
                self._log_msg(
                    f"\n[Queue Finished: {done_count} completed, {fail_count} failed/stopped]\n",
                    color=WARNING if done_count > 0 else ERROR_COL,
                )
            else:
                self._log_msg(f"\n[Queue Finished: {done_count} jobs completed]\n", color=SUCCESS)

    def _cleanup_partial_frames(self):
        """Remove a frames/ dir this run created if the run ended prematurely."""
        if not self._frames_dir:
            return
        if self._frames_preexisting:
            self._log_msg("Partial frames left in existing folder.\n", color=WARNING)
            return
        if (
            self._frames_dir == self._video_frames_dir
            and count_saved_video_frames(self._frames_dir) > 0
        ):
            return
        if os.path.isdir(self._frames_dir):
            shutil.rmtree(self._frames_dir, ignore_errors=True)
            self._log_msg("Removed partial frames.\n", color=WARNING)

    def _cleanup_preview_temps(self):
        if not self._preview_tmp_path:
            return
        for path in (self._preview_tmp_path, f"{self._preview_tmp_path}.tmp"):
            if os.path.isfile(path):
                try:
                    os.remove(path)
                except OSError:
                    pass

    def _cleanup_job_temps(self, out_dir, pid):
        """Remove pid-qualified temp files an interrupted child may have left.
        deepdream.py cleans these on normal failures, but a force-killed child
        (Stop / window close) runs no cleanup: ``*.p<pid>.tmp`` and temp mp4s."""
        if not out_dir or not pid:
            return
        suffixes = (f".p{pid}.tmp", f".p{pid}.temp.mp4", f".p{pid}.remux.temp.mp4")
        try:
            entries = os.listdir(out_dir)
        except OSError:
            return
        for name in entries:
            if name.lower().endswith(suffixes):
                try:
                    os.remove(os.path.join(out_dir, name))
                except OSError:
                    pass

    def _load_output_image(self, job=None):
        """Preview the output media this run produced."""
        if job is None:
            job = self._current_job
        if job is not None:
            self._load_input_image_for_job(job)
        path = self._expected_output
        if path and os.path.isdir(path) and count_saved_video_frames(path) > 0:
            # Frames-only generation (separated save): a dreamed frame
            # sequence awaiting export via Save Video, not an .mp4 yet.
            # Checked before the isfile fallback below, which would otherwise
            # discard directory paths.
            self._load_video_frames_ready(job, path)
            return
        if not path or not os.path.isfile(path):
            path = (
                self._newest_output_media()
                if self._is_video_output
                else (self._newest_output_png() or self._newest_output_media())
            )
        if not path:
            return
        self._output_path = path
        if job:
            job.output_path = path

        is_video = path.lower().endswith(VIDEO_EXTENSIONS)
        if is_video:
            self._is_video_output = True
            # Drop any still-comparison state from a previously viewed image
            # result: a later window resize (or Spacebar) would otherwise
            # replace the video preview with that stale image.
            self._dream_pil = None
            self._aligned_input_pil = None
            self._compare_mode = "Dream"
            self._timelapse_mp4 = None
            title = f"Dream Video Result (Job #{job.id})" if job else "Dream Video Result"
            ok = self._setup_mp4_preview(path, title=title)
            if not ok:
                try:
                    self._preview_title.configure(text=title)
                except tk.TclError:
                    pass
            self._showing_zoom_focus = False
            try:
                self._view_image_btn.grid_remove()
                self._view_timelapse_btn.grid_remove()
            except tk.TclError:
                pass
            self._compare_frame.grid_remove()
        else:
            self._is_video_output = False
            try:
                with Image.open(path) as _raw:
                    pil = _raw.convert("RGB")
            except (OSError, ValueError):
                return
            self._dream_pil = pil.copy()
            dw, dh = self._dream_pil.size
            if self._input_pil is not None:
                if self._input_pil.size != (dw, dh):
                    self._aligned_input_pil = self._input_pil.resize(
                        (dw, dh), Image.Resampling.LANCZOS
                    )
                else:
                    self._aligned_input_pil = self._input_pil.copy()
            else:
                self._aligned_input_pil = None

            self._compare_mode = "Dream"
            if self._compare_seg is not None:
                self._compare_seg.set("Dream")
            self._teardown_video()
            self._viewer_mode = "image"
            self._timelapse_mp4 = None
            self._reset_viewer()
            self._show_output_preview(self._dream_pil)
            self._showing_zoom_focus = False
            title = f"Dream Result (Job #{job.id})" if job else "Dream Result"
            try:
                self._preview_title.configure(text=title)
            except tk.TclError:
                pass
            if self._aligned_input_pil is not None:
                self._compare_frame.grid()
            else:
                self._compare_frame.grid_remove()

            timelapse_path = None
            wants_timelapse = job.timelapse if job else self._timelapse.get()
            if wants_timelapse:
                candidate = str(Path(path).with_name(f"{Path(path).stem}_timelapse.mp4"))
                if os.path.isfile(candidate):
                    timelapse_path = candidate
                    if job:
                        job.timelapse_path = candidate
            self._timelapse_mp4 = timelapse_path
            try:
                if timelapse_path:
                    self._view_timelapse_btn.grid()
                    self._view_image_btn.grid_remove()
                else:
                    self._view_timelapse_btn.grid_remove()
                    self._view_image_btn.grid_remove()
            except tk.TclError:
                pass

        self._open_dir_btn.grid()
        self._log_msg(f"\nSaved \u2192 {path}\n", color=SUCCESS)

    def _load_video_frames_ready(self, job, frames_dir, status_text=None):
        """Show a freshly dreamed frame sequence awaiting Save Video export."""
        count = count_saved_video_frames(frames_dir)
        if count <= 0:
            return
        meta = read_video_frames_meta(frames_dir)
        job_id = getattr(job, "id", None) if job is not None else None
        mode = getattr(job, "mode", None) if job is not None else None
        if not mode:
            mode = "Dream Zoom" if (meta.get("mode") == "zoom") else "Dream Video"
        try:
            fps = float(meta.get("fps") or 0)
        except (TypeError, ValueError):
            fps = 0.0
        if not fps or fps <= 0:
            try:
                fps = float(getattr(job, "zoom_fps", 0) or 0)
            except (TypeError, ValueError):
                fps = 0.0
        if not fps or fps <= 0:
            fps = 30.0
        source = meta.get("source")
        if not source and mode == "Dream Video" and job is not None:
            cand = getattr(job, "image_path", None)
            if cand and os.path.isfile(cand):
                source = cand
        self._video_frames_dir = frames_dir
        self._video_frames_count = count
        self._video_frames_fps = fps
        self._video_frames_mode = mode
        self._video_frames_source = source
        self._video_frames_job_id = job_id
        self._video_frames_job = job
        if job is not None and meta.get("seed") is not None and getattr(job, "seed", None) is None:
            job.seed = meta.get("seed")
        # No .mp4 yet: preview the dreamed PNGs directly as a scrubbable
        # timeline (no throwaway encode). Saving via Save Video still writes
        # the final file at the chosen quality.
        self._output_path = None
        self._dream_pil = None
        self._aligned_input_pil = None
        self._compare_mode = "Dream"
        self._is_video_output = True
        self._timelapse_mp4 = None

        total_frames = meta.get("total_frames") or getattr(job, "zoom_frames", None)
        tot_txt = f"/{total_frames}" if total_frames else ""
        is_stopped = job is not None and getattr(job, "status", None) == "Stopped"

        if is_stopped:
            title = f"Frames Ready ({count}{tot_txt} frames, Paused)"
        else:
            title = f"Frames Ready (Job #{job_id})" if job_id is not None else "Frames Ready"

        ok = self._setup_frames_preview(frames_dir, fps=fps, title=title)
        if not ok:
            try:
                files = list_saved_video_frames(frames_dir)
                if files:
                    with Image.open(files[-1]) as _raw:
                        self._viewer_mode = "image"
                        self._show_output_preview(_raw.convert("RGB"), reset_view=True)
                self._showing_zoom_focus = False
                self._preview_title.configure(text=title)
            except (OSError, ValueError, tk.TclError):
                pass
        else:
            self._showing_zoom_focus = False
            try:
                self._preview_title.configure(text=f"{title}: scrub or press Play")
            except tk.TclError:
                pass
        try:
            self._view_image_btn.grid_remove()
            self._view_timelapse_btn.grid_remove()
        except tk.TclError:
            pass
        self._compare_frame.grid_remove()
        self._open_dir_btn.grid()
        try:
            if status_text:
                self._status_label.configure(
                    text=status_text,
                    text_color=WARNING if is_stopped else SUCCESS,
                )
            elif is_stopped:
                self._status_label.configure(
                    text=f"Stopped ({count}{tot_txt} frames ready)",
                    text_color=WARNING,
                )
            else:
                self._status_label.configure(text="Frames ready \u2713", text_color=SUCCESS)
        except tk.TclError:
            pass

        if is_stopped:
            self._resumable_job = job
            self._update_resume_button_visibility()
        elif self._resumable_job is job:
            self._resumable_job = None
            self._update_resume_button_visibility()

        self._update_video_quality_visibility()
        self._refresh_video_save_ui()
        if is_stopped:
            self._log_msg(
                f"\nPaused at {count}{tot_txt} frames \u2192 {frames_dir}\n"
                f"• Press 'Save Video' to export an .mp4 of the {count} frames generated so far.\n"
                f"• Press '▶ Resume' to pick up at frame {count + 1} and generate the rest.\n",
                color=WARNING,
            )
        else:
            self._log_msg(
                f"\nSaved {count} frames \u2192 {frames_dir}\n"
                "Choose a quality in Video Export and press Save Video to write the .mp4 "
                "(repeatable at other qualities without re-dreaming).\n",
                color=SUCCESS,
            )

    # Queue management and rendering

    def _get_queue_badge_text(self):
        pending = sum(1 for j in self._job_queue if j.status == "Pending")
        running = sum(1 for j in self._job_queue if j.status == "Running")
        active = pending + running
        if active > 0:
            return f"Queue ({active})"
        elif self._job_queue:
            return f"Queue ({len(self._job_queue)})"
        return "Queue (0)"

    def _update_queue_badge(self):
        badge_text = self._get_queue_badge_text()
        cur = self._bottom_tab_seg.get()
        new_values = ["Console Output", badge_text]
        self._bottom_tab_seg.configure(values=new_values)
        if cur.startswith("Queue"):
            self._bottom_tab_seg.set(badge_text)
        else:
            self._bottom_tab_seg.set("Console Output")

    def _on_bottom_tab_change(self, tab):
        if tab.startswith("Queue"):
            self._log.grid_remove()
            self._clear_log_btn.grid_remove()
            self._queue_action_frame.grid(row=0, column=1, sticky="e")
            self._queue_container.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 6))
            self._refresh_queue_ui()
        else:
            self._queue_container.grid_remove()
            self._queue_action_frame.grid_remove()
            self._clear_log_btn.grid(row=0, column=1, sticky="e")
            self._log.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 6))

    def _refresh_queue_ui(self):
        if self._queue_scroll is None:
            return
        queue_parent = self._queue_inner
        if queue_parent is None:
            queue_parent = getattr(self._queue_scroll, "inner", self._queue_scroll)
        if queue_parent is None:
            return

        for child in queue_parent.winfo_children():
            child.destroy()

        if not self._queue_running and any(j.status == "Pending" for j in self._job_queue):
            self._resume_btn.pack(side="right", padx=(4, 0))
        else:
            self._resume_btn.pack_forget()

        if not self._job_queue:
            empty_lbl = ModernLabel(
                queue_parent,
                text="Queue is empty.\n\nUse '+ Add to Queue' or drop multiple images to batch dream.",
                font=ModernFont(family="Segoe UI", size=12),
                text_color=TEXT_DIM,
                justify="center",
            )
            empty_lbl.pack(expand=True, pady=36)
            return

        for job in self._job_queue:
            card = ModernFrame(
                queue_parent,
                fg_color=BG_CARD,
                border_width=1,
                border_color=BORDER,
            )
            card.pack(fill="x", padx=4, pady=3)
            card.grid_columnconfigure(1, weight=1)

            id_lbl = ModernLabel(
                card,
                text=f"#{job.id}",
                font=ModernFont(family="Segoe UI", size=11, weight="bold"),
                text_color=TEXT_MAIN,
                width=32,
            )
            id_lbl.grid(row=0, column=0, rowspan=2, padx=(8, 4), pady=4)

            fname = Path(job.image_path).name
            title_lbl = ModernLabel(
                card,
                text=fname,
                font=ModernFont(family="Segoe UI", size=11, weight="bold"),
                text_color=TEXT_MAIN,
                anchor="w",
            )
            title_lbl.grid(row=0, column=1, sticky="w", padx=(2, 4), pady=(4, 0))
            Tooltip(title_lbl, job.image_path)

            details = (
                f"{job.model_key.upper()} • {job.layer_spec or 'default'} • {job.iterations} iters"
            )
            if job.guidance_mode == "Target Class" and job.target_class:
                details += f" • Class: {job.target_class}"
            elif job.guidance_mode == "Guide Image" and job.guide_path:
                details += f" • Guide: {Path(job.guide_path).name}"
            if job.mode != "Single Image":
                details += f" • {job.mode}"
            if job.mode != "Single Image" or job.timelapse:
                details += f" • {job.video_quality or 'Balanced'} (CRF {job.video_crf})"
            fdir = getattr(job, "frames_dir", None)
            cnt = count_saved_video_frames(fdir) if (fdir and os.path.isdir(fdir)) else 0

            if (
                job.mode in ("Dream Zoom", "Dream Video")
                and job.status == "Done"
                and not getattr(job, "output_path", None)
            ):
                details += " • frames ready: Save Video"
            elif job.mode in ("Dream Zoom", "Dream Video") and job.status == "Stopped" and cnt > 0:
                details += " • paused: can Save Video or Resume"

            sub_lbl = ModernLabel(
                card,
                text=details,
                font=ModernFont(family="Segoe UI", size=10),
                text_color=TEXT_DIM,
                anchor="w",
            )
            sub_lbl.grid(row=1, column=1, sticky="w", padx=(2, 4), pady=(0, 4))

            status_color = {
                "Pending": TEXT_DIM,
                "Running": WARNING,
                "Done": SUCCESS,
                "Stopped": WARNING,
                "Failed": ERROR_COL,
            }.get(job.status, TEXT_DIM)

            has_stopped_frames = False
            if job.status == "Stopped" and cnt > 0:
                has_stopped_frames = True
                meta = read_video_frames_meta(fdir) if fdir else {}
                tot = meta.get("total_frames") or getattr(job, "zoom_frames", None)
                tot_txt = f"/{tot}" if tot else ""
                status_text = f"⏹ Stopped ({cnt}{tot_txt})"
            else:
                status_text = {
                    "Pending": "⏳ Pending",
                    "Running": "▶ Running",
                    "Done": "✓ Done",
                    "Stopped": "⏹ Stopped",
                    "Failed": "✕ Failed",
                }.get(job.status, job.status)

            status_lbl = ModernLabel(
                card,
                text=status_text,
                font=ModernFont(family="Segoe UI", size=10, weight="bold"),
                text_color=status_color,
                anchor="e",
            )
            status_lbl.grid(row=0, column=2, rowspan=2, padx=(4, 6), pady=4)

            if job.status == "Pending":
                del_btn = ModernButton(
                    card,
                    text="✕",
                    width=24,
                    height=24,
                    fg_color=BG_PANEL,
                    hover_color=ERROR_COL,
                    text_color=TEXT_MAIN,
                    font=ModernFont(family="Segoe UI", size=10, weight="bold"),
                    command=lambda j=job: self._remove_job(j),
                )
                del_btn.grid(row=0, column=3, rowspan=2, padx=(2, 8), pady=4)
                Tooltip(del_btn, "Remove from queue")
            elif job.status == "Done" and (job.output_path or cnt > 0):
                view_btn = ModernButton(
                    card,
                    text="View",
                    width=44,
                    height=24,
                    fg_color=BG_PANEL,
                    hover_color=BORDER,
                    text_color=TEXT_MAIN,
                    font=ModernFont(family="Segoe UI", size=10, weight="bold"),
                    command=lambda j=job: self._view_job_output(j),
                )
                view_btn.grid(row=0, column=3, rowspan=2, padx=(2, 8), pady=4)
                Tooltip(view_btn, "Preview dream result")
            elif has_stopped_frames:
                resume_btn = ModernButton(
                    card,
                    text="Resume",
                    width=54,
                    height=24,
                    fg_color=BG_PANEL,
                    hover_color=ACCENT,
                    text_color=TEXT_MAIN,
                    font=ModernFont(family="Segoe UI", size=10, weight="bold"),
                    command=lambda j=job: self._resume_job(j),
                )
                resume_btn.grid(row=0, column=3, rowspan=2, padx=(2, 8), pady=4)
                Tooltip(resume_btn, f"Resume generation from frame {cnt + 1}{tot_txt}")
            elif job.status in ("Stopped", "Failed"):
                retry_btn = ModernButton(
                    card,
                    text="Retry",
                    width=44,
                    height=24,
                    fg_color=BG_PANEL,
                    hover_color=BORDER,
                    text_color=TEXT_MAIN,
                    font=ModernFont(family="Segoe UI", size=10),
                    command=lambda j=job: self._retry_job(j),
                )
                retry_btn.grid(row=0, column=3, rowspan=2, padx=(2, 8), pady=4)
                Tooltip(retry_btn, "Re-queue this job")

            if job.status in ("Done", "Stopped"):
                self._bind_job_output(card, job)
                self._bind_job_output(title_lbl, job)
                self._bind_job_output(sub_lbl, job)
                self._bind_job_output(status_lbl, job)

    def _bind_job_output(self, widget, job):
        def show_output(_event: tk.Event) -> None:
            self._view_job_output(job)

        widget.bind("<Button-1>", show_output)

    def _remove_job(self, job):
        if job in self._job_queue and job.status == "Pending":
            self._job_queue.remove(job)
            self._update_queue_badge()
            self._refresh_queue_ui()

    def _clear_pending_jobs(self):
        self._job_queue = [j for j in self._job_queue if j.status != "Pending"]
        self._update_queue_badge()
        self._refresh_queue_ui()

    def _clear_completed_jobs(self):
        self._job_queue = [j for j in self._job_queue if j.status in ("Pending", "Running")]
        self._update_queue_badge()
        self._refresh_queue_ui()

    def _retry_job(self, job):
        job.status = "Pending"
        self._update_queue_badge()
        self._refresh_queue_ui()
        if not self._queue_running:
            self._start_queue()

    def _view_job_output(self, job):
        if job.output_path and os.path.isfile(job.output_path):
            self._expected_output = job.output_path
            self._output_path = job.output_path
            self._is_video_output = job.mode in ("Dream Zoom", "Dream Video")
            self._load_output_image(job=job)
            return
        # Frames-only video job not yet exported: review its frames and offer Save Video.
        frames_dir = getattr(job, "frames_dir", None)
        if frames_dir and os.path.isdir(frames_dir) and count_saved_video_frames(frames_dir) > 0:
            self._expected_output = frames_dir
            self._is_video_output = job.mode in ("Dream Zoom", "Dream Video")
            self._load_output_image(job=job)
            return
        messagebox.showinfo("No Output", f"No output file available for Job #{job.id}.")
        return

    def _newest_output_png(self):
        """Fallback: the most recently modified PNG in the output folder."""
        return newest_png(resolve_out_dir(self._out_dir.get()))

    def _newest_output_media(self):
        """Fallback: the most recently modified media in the output folder."""
        return newest_media(resolve_out_dir(self._out_dir.get()))

    def _open_file_cross_platform(self, path):
        """Open a file-manager path with the platform's default application."""
        if not path:
            return
        try:
            if sys.platform == "win32":
                os.startfile(path)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except (OSError, ValueError):
            pass

    def _reveal_target(self):
        """File to highlight, or folder to open when no single file applies."""
        mode = self._viewer_mode
        if mode == "timelapse":
            mp4 = self._timelapse_mp4
            if mp4 and os.path.isfile(mp4):
                return os.path.abspath(mp4)
        elif mode == "video":
            if self._video_kind == "mp4":
                mp4 = self._video_mp4
                if mp4 and os.path.isfile(mp4):
                    return os.path.abspath(mp4)
            elif self._video_kind == "frames":
                d = self._video_frames_dir
                if d and os.path.isdir(d):
                    return os.path.abspath(d)

        if self._output_path and os.path.isfile(self._output_path):
            return os.path.abspath(self._output_path)

        d = self._video_frames_dir
        if d and os.path.isdir(d) and self._has_ready_video_frames():
            return os.path.abspath(d)

        try:
            out_d = resolve_out_dir(self._out_dir.get())
            return os.path.abspath(out_d) if out_d else None
        except (AttributeError, tk.TclError):
            return None

    def _show_in_folder(self):
        """Reveal the output in the OS file manager, highlighted when possible.
        Windows ``explorer /select,"<file>"`` and macOS ``open -R`` select the
        file; Linux has no such API, so its folder opens instead."""
        target = self._reveal_target()
        if not target:
            try:
                target = resolve_out_dir(self._out_dir.get())
            except (AttributeError, tk.TclError):
                pass
        if not target:
            return
        target = os.path.abspath(target)
        try:
            if os.path.isfile(target):
                cmd = build_reveal_command(target)
                if cmd is not None:
                    subprocess.Popen(cmd)
                    return
                # No select API (Linux): open the containing folder.
                self._open_file_cross_platform(os.path.dirname(target) or target)
                return
            if os.path.isdir(target):
                self._open_file_cross_platform(target)
                return
            # If target does not exist on disk as file or dir, open containing folder or out_dir
            parent = os.path.dirname(target)
            if parent and os.path.isdir(parent):
                self._open_file_cross_platform(parent)
                return
            out_d = resolve_out_dir(self._out_dir.get()) if hasattr(self, "_out_dir") else None
            if out_d:
                os.makedirs(out_d, exist_ok=True)
                self._open_file_cross_platform(out_d)
        except (OSError, ValueError):
            pass


# Entry point


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    initial_image = argv[0] if argv else None
    try:
        app = DeepDreamGUI(initial_image=initial_image)
        app.mainloop()
    except BaseException as exc:
        if isinstance(exc, SystemExit) and exc.code in (0, None):
            return
        err = traceback.format_exc()
        try:
            messagebox.showerror(
                "DeepDream Studio - Error",
                f"An unexpected error occurred:\n\n{err}",
            )
        except tk.TclError:
            pass
        raise


if __name__ == "__main__":
    main()
