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
from .gui_classifier import TargetClassPanelMixin

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
    _dialog_pattern_for_ext,
    build_reveal_command,
    clamp01,
    clamp_zoom,
    count_saved_video_frames,
    create_split_image,
    draw_zoom_focus_marker,
    extract_output_path,
    fit_zoom_for,
    list_saved_video_frames,
    newest_media,
    newest_png,
    parse_dropped_paths,
    render_zoomed_preview,
    resolve_out_dir,
    terminate_process,
)
from .gui_job import (
    GenerationJob,
    ProgressTracker,
)
from .gui_layers import LayerPanelMixin
from .gui_playback import VideoPlaybackMixin
from .gui_process import read_generation_output, start_cli_process
from .gui_video import VideoExportMixin
from .gui_viewer import ViewerGeometry
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
    ModernSegmentedButton,
    ModernTextbox,
    Tooltip,
    _apply_modern_style,
    _SliderRow,
)
from .imagenet_categories import lookup_category
from .model_data import (
    DEFAULT_MODEL_KEY,
    DEFAULT_MODEL_LABEL,
    MAX_NAME_LEN,
    MODEL_BY_LABEL,
    MODEL_KEYS,
    MODEL_META,
    VIDEO_EXTENSIONS,
    filename_stem_error,
    layer_display_list,
    layer_value_map,
    read_video_frames_meta,
)
from .protocol import EV_DONE, EV_PID, EV_PREVIEW, parse_event

__all__ = ["DeepDreamGUI", "main"]

# Ordered model display labels for the model dropdown.
MODEL_LABELS = [MODEL_META[k]["label"] for k in MODEL_KEYS]


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


# Main window


class DeepDreamGUI(
    TargetClassPanelMixin, LayerPanelMixin, VideoExportMixin, VideoPlaybackMixin, tk.Tk
):
    def __init__(self, initial_image=None):
        super().__init__()

        # UI widget handles populated during _build_ui
        self._zoom_focus_label: Any = None
        self._status_label: Any = None
        self._preset_menu: Any = None
        self._main_resume_btn: Any = None
        self._zoom_bar: Any = None
        self._zoom_label: Any = None
        self._compare_seg: Any = None
        self._queue_scroll: Any = None
        self._queue_inner: Any = None

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
        self._init_target_class_state()

        # Multi-layer
        self._init_layer_state()

        # Video & animation options
        self._timelapse = tk.BooleanVar(value=False)
        self._zoom_frames = tk.IntVar(value=60)
        self._zoom_scale = tk.DoubleVar(value=1.03)
        self._zoom_rot = tk.DoubleVar(value=0.0)
        self._zoom_fps = tk.IntVar(value=30)
        self._zoom_cx = tk.DoubleVar(value=0.5)
        self._zoom_cy = tk.DoubleVar(value=0.5)
        self._video_blend = tk.DoubleVar(value=0.5)
        self._init_video_export_state()

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
        self._init_video_playback_state()
        self._timelapse_mp4: str | None = None
        self._viewer_mode = "image"  # "image" | "video" | "timelapse"

        self._is_closing = False
        self._reported_callback_errors: set = set()
        self._proc = None
        # Temp-file pid that each generation child announced, keyed by process.
        self._proc_pids: dict[int, int | None] = {}
        self._proc_reader_threads: dict[int, threading.Thread] = {}
        self._poll_after_id = None
        self._next_job_after_id = None
        self._log_queue: queue.Queue[str] = queue.Queue()
        self._output_path = None  # last successfully saved image
        self._preview_pil = None  # PIL image for the input preview
        self._tracker = ProgressTracker()
        self._expected_output = None  # path from the "Done in ... ->" log line
        self._aborted = False  # True when the user pressed Stop
        self._aborted_jobs: set[int] = set()
        self._frames_dir = None  # frames/ dir for the current run
        self._frames_preexisting = False

        # Generation Queue state
        self._job_queue = []
        self._current_job = None
        self._job_counter = 0
        self._queue_running = False
        self._resumable_job = None

        # Additional UI and viewer state
        self._last_render_view_size = None
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
            pid_ref = self._proc_pids.get(id(proc_ref))
            reader_ref = self._proc_reader_threads.get(id(proc_ref))
            encode_reader_ref = self._encode_reader_thread
            # The encoder reports its own pid; its process handle can be a venv launcher.
            export_target = self._video_export_target
            export_dir = os.path.dirname(export_target) if export_target else None

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
                if reader_ref is not None:
                    reader_ref.join()
                if encode_reader_ref is not None:
                    encode_reader_ref.join()
                cleanup_pid = self._proc_pids.get(id(proc_ref), pid_ref)
                encode_pid = self._video_export_pid
                # Sweep only after the child is reaped, otherwise it can
                # recreate a partial frame/temp after the cleanup runs.
                # _cleanup_partial_frames() (and its _log_msg calls)
                # short-circuits on _is_closing, so this is safe after destroy().
                self._cleanup_partial_frames()
                if job_ref is not None:
                    self._cleanup_job_temps(job_ref.out_dir, cleanup_pid)
                vfd = self._video_frames_dir
                if encode_pid is not None:
                    if vfd:
                        self._cleanup_job_temps(os.path.dirname(vfd), encode_pid)
                    self._cleanup_job_temps(export_dir, encode_pid)

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

        self._build_target_class_panel(sb)

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

        self._build_video_export_controls(sb)

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
        new_layers = layer_display_list(model_key)
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
            return self._viewer_geometry(src).normalized_point(click_x, click_y)
        except (TypeError, ValueError, ZeroDivisionError):
            return None

    def _viewer_geometry(self, src):
        viewport = self._get_view_size()
        return ViewerGeometry(
            source_size=src.size,
            viewport_size=viewport,
            label_size=self._viewer_label_size(),
            zoom=self._effective_zoom(src, *viewport),
            center=self._ensure_viewer_center(src),
        )

    def _zoom_at(self, factor, cursor_x=None, cursor_y=None):
        src = self._get_viewer_source()
        if src is None:
            return
        geometry = self._viewer_geometry(src)
        old_zoom = geometry.zoom
        cursor = (cursor_x, cursor_y) if cursor_x is not None and cursor_y is not None else None
        try:
            new_zoom, cx, cy = geometry.zoom_target(factor, cursor)
        except (TypeError, ValueError, ZeroDivisionError):
            new_zoom, cx, cy = geometry.zoom_target(factor)
        if abs(new_zoom - old_zoom) < 1e-6:
            return
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
            return self._viewer_geometry(src).split_label_x(split_ratio)
        except (TypeError, ValueError, ZeroDivisionError, AttributeError):
            return None

    def _view_x_to_split_ratio(self, click_x):
        src = self._get_viewer_source()
        if src is None:
            return None
        try:
            return self._viewer_geometry(src).split_ratio(click_x)
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
                label_to_key = layer_value_map(model_key)
                layer_spec = label_to_key.get(self._layer_var.get(), self._layer_var.get())

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
            proc = start_cli_process(cmd, ROOT)
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
        # Sweep under the handle's pid until the child announces its own.
        handle_pid = getattr(proc, "pid", None)
        self._proc_pids[id(proc)] = handle_pid if isinstance(handle_pid, int) else None
        self._ensure_poll()
        reader = threading.Thread(target=self._reader, daemon=True, args=(proc, job))
        self._proc_reader_threads[id(proc)] = reader
        reader.start()

    def _stop(self):
        self._queue_running = False
        self._cancel_next_job_timer()
        encode_proc = self._encode_proc
        stopped_child = False
        if self._proc and self._proc.poll() is None:
            self._aborted = True
            job = self._current_job
            if job is not None and getattr(job, "id", None) is not None:
                self._aborted_jobs.add(job.id)
            terminate_process(self._proc)
            self._log_msg("\n[Queue stopped by user]\n", color=WARNING)
            stopped_child = True
        if encode_proc and encode_proc.poll() is None:
            self._video_export_cancelled = True
            terminate_process(encode_proc)
            self._log_msg("\n[Video export stopped by user]\n", color=WARNING)
            stopped_child = True
        if stopped_child:
            return
        if self._proc is not None:
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

        def record_pid(line):
            event = parse_event(line)
            if event is not None and event.get("event") == EV_PID:
                pid = event.get("pid")
                if isinstance(pid, int):
                    self._proc_pids[id(proc)] = pid

        read_generation_output(proc, getattr(job, "id", 0), self._log_queue, on_line=record_pid)

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
                    # The stale child is reaped, so sweep the temp files it left.
                    stale_pid = self._proc_pids.pop(proc_id, None)
                    self._proc_reader_threads.pop(proc_id, None)
                    for j in self._job_queue:
                        if j.id == job_id:
                            self._cleanup_job_temps(j.out_dir, stale_pid)
                            break
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
        if kind == EV_PID:
            # Names this child's temp files. Its process handle can be a venv
            # launcher, so the sweep must use the pid the child reports.
            pid = event.get("pid")
            if isinstance(pid, int) and self._proc is not None:
                self._proc_pids[id(self._proc)] = pid
            return False
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
        proc_id = id(self._proc)
        finished_pid = self._proc_pids.pop(proc_id, None)
        self._proc_reader_threads.pop(proc_id, None)
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
            self._cleanup_job_temps(job.out_dir, finished_pid)
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
