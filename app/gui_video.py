"""Video export: quality controls, frame-sequence encoding, and encode drain.

Builds and drives the sidebar "Video Export" card and the encode subprocess
lifecycle for saving dreamed frame sequences to .mp4. Mixed into DeepDreamGUI;
the owning window drives state setup via _init_video_export_state and builds
the panel via _build_video_export_controls.
"""

import os
import queue
import threading
import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Any

from .gui_export import VideoExportRequest, suggest_video_name
from .gui_helpers import ROOT, count_saved_video_frames, resolve_out_dir
from .gui_job import _CRF_TO_QUALITY, VIDEO_QUALITY_INFO, ProgressTracker
from .gui_process import read_encode_output, start_cli_process
from .gui_widgets import (
    ACCENT,
    BG_CARD,
    BORDER,
    ERROR_COL,
    SUCCESS,
    TEXT_DIM,
    TEXT_LABEL,
    TEXT_MAIN,
    WARNING,
    ModernButton,
    ModernFont,
    ModernFrame,
    ModernLabel,
    ModernSegmentedButton,
    Tooltip,
)
from .protocol import EV_ERROR, EV_PID, parse_event


class VideoExportMixin:
    """Quality preset + Save Video UI and its encode-subprocess lifecycle."""

    # Methods owned by DeepDreamGUI.
    _log_msg: Callable[..., Any]
    _setup_mp4_preview: Callable[..., Any]
    _ensure_poll: Callable[..., Any]
    _stop_video_playback: Callable[..., Any]
    _cleanup_job_temps: Callable[..., Any]
    _slider: Callable[..., Any]

    # Shared window state.
    _mode_var: tk.StringVar
    _out_dir: tk.StringVar
    _timelapse: tk.BooleanVar
    _video_quality_var: tk.StringVar
    _video_crf: tk.IntVar
    _tracker: ProgressTracker
    _encode_queue: queue.Queue
    _encode_proc: Any
    _job_queue: Any
    _proc: Any
    _queue_running: bool
    _is_closing: bool
    _is_video_output: bool
    _progress: Any
    _status_label: Any
    _stop_btn: Any
    _preview_title: Any
    _open_dir_btn: Any
    _view_image_btn: Any
    _view_timelapse_btn: Any
    _output_path: Any
    _timelapse_mp4: Any

    # Widget handles populated during build.
    _video_quality_seg: Any
    _video_quality_info_lbl: Any
    _video_quality_frame: Any
    _save_video_btn: Any
    _sidebar_save_video_btn: Any
    _video_frames_label: Any

    # Separated video-save state.
    _video_frames_dir: str | None
    _video_frames_count: int
    _video_frames_fps: float
    _video_frames_mode: str | None
    _video_frames_source: str | None
    _video_frames_job_id: int | None
    _video_frames_job: Any
    _video_save_in_progress: bool
    _video_export_target: str | None
    _video_export_request: VideoExportRequest | None
    _video_export_cancelled: bool
    _video_export_failed: bool
    _video_export_pid: int | None
    _video_export_prev_title: str | None
    _encode_reader_thread: threading.Thread | None
    _video_last_save: str | None
    _video_last_crf: int | None

    def _init_video_export_state(self) -> None:
        """Create the export controls' variables and frame-sequence state."""
        self._video_quality_seg = None
        self._video_quality_info_lbl = None
        self._video_quality_frame = None
        self._save_video_btn = None
        self._sidebar_save_video_btn = None
        self._video_frames_label = None
        self._video_quality_var = tk.StringVar(value="Balanced")
        self._video_crf = tk.IntVar(value=17)
        self._encode_proc = None
        # Encode-subprocess events (log lines and completion) as tuples, drained
        # on the Tk thread by _poll_log so workers never touch Tk.
        self._encode_queue: queue.Queue[tuple] = queue.Queue()
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
        self._video_export_request = None
        # Set by _stop() when the user terminates a running encode; consumed by
        # _finish_video_encode so cancellation is not reported as a failure.
        self._video_export_cancelled = False
        # Reported by the encode child: whether it announced a failure, and its
        # own pid (the process handle can be a venv launcher).
        self._video_export_failed = False
        self._video_export_pid = None
        self._encode_reader_thread = None
        # Preview header replaced by "Encoding Video -> ..."; put back when the
        # export does not produce a preview.
        self._video_export_prev_title = None
        self._video_last_save = None
        self._video_last_crf = None

    def _build_video_export_controls(self, sb):
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

        return suggest_video_name(
            job=job,
            frames_dir=self._video_frames_dir,
            source=self._video_frames_source,
            mode=self._video_frames_mode,
            crf=crf,
            out_dir=out_dir if out_dir is not None else resolve_out_dir(self._out_dir.get()),
        )

    def _build_export_request(
        self, frames_dir, out_path, fps, audio_from=None
    ) -> VideoExportRequest:
        return VideoExportRequest(
            frames_dir=frames_dir,
            out_path=out_path,
            fps=float(fps),
            quality=self._video_quality_var.get() or "",
            crf=self._get_current_crf(),
            audio_from=audio_from,
        )

    def _build_encode_cmd(self, frames_dir, out_path, fps, audio_from=None):
        return self._build_export_request(frames_dir, out_path, fps, audio_from).build_cmd()

    def _on_save_video_click(self):
        if self._video_save_in_progress or not self._has_ready_video_frames():
            return
        frames_dir = self._video_frames_dir
        count = self._video_frames_count or count_saved_video_frames(frames_dir)
        fps = float(self._video_frames_fps or 30.0)
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
        request = self._build_export_request(frames_dir, path, fps, audio_from)
        cmd = request.build_cmd()
        crf = request.crf
        qname = request.quality.strip() or "Custom"
        self._log_msg(
            f"\n[Video Export] {count} frames @ {fps:g} fps ({qname}, CRF {crf}) -> {path}\n"
            f"$ {' '.join(cmd)}\n\n",
            color=TEXT_DIM,
        )
        self._video_save_in_progress = True
        self._video_export_target = path
        self._video_export_request = request
        self._tracker = ProgressTracker()
        self._progress.set(0)
        self._status_label.configure(text="Encoding video...", text_color=WARNING)
        self._stop_video_playback()
        self._video_export_failed = False
        self._video_export_pid = None
        self._video_export_prev_title = None
        try:
            self._video_export_prev_title = self._preview_title.cget("text")
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
        # Sweep under the handle's pid until the encoder announces its own.
        handle_pid = getattr(proc, "pid", None)
        self._video_export_pid = handle_pid if isinstance(handle_pid, int) else None
        self._ensure_poll()
        reader = threading.Thread(
            target=self._run_video_encode,
            daemon=True,
            args=(cmd, proc, path, job_id, count, fps, crf),
        )
        self._encode_reader_thread = reader
        reader.start()

    def _handle_encode_line(self, line, out_path):
        event = parse_event(line)
        if event is not None:
            self._record_encode_event(event)
        if self._is_closing:
            return
        if event is None:
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

    def _record_encode_event(self, event):
        """Keep what the encode child announced about itself until the export ends."""
        kind = event.get("event")
        if kind == EV_PID:
            pid = event.get("pid")
            self._video_export_pid = pid if isinstance(pid, int) else None
        elif kind == EV_ERROR:
            self._video_export_failed = True

    def _start_encode_process(self, cmd):
        """Start the encode subprocess. Called on the Tk thread."""
        return start_cli_process(cmd, ROOT)

    def _run_video_encode(self, cmd, proc, out_path, job_id, count, fps, crf):
        """Worker thread: stream an already-started encode subprocess into the queue.

        Touches no Tk state and no Tcl calls. The process is started on the Tk
        thread by _on_save_video_click so _stop() can always see it; log lines
        and the final result are queued for _poll_log to apply on the Tk thread.
        """

        def record_event(line):
            event = parse_event(line)
            if event is not None:
                self._record_encode_event(event)

        read_encode_output(proc, out_path, job_id, self._encode_queue, on_line=record_event)

    def _finish_video_encode(self, rc, log_text, out_path, job_id):
        request = self._video_export_request
        self._video_export_request = None
        cancelled = self._video_export_cancelled
        self._video_export_cancelled = False
        failed = self._video_export_failed
        self._video_export_failed = False
        encode_pid = self._video_export_pid
        self._video_export_pid = None
        prev_title = self._video_export_prev_title
        self._video_export_prev_title = None
        self._video_save_in_progress = False
        self._video_export_target = None
        # Cleared on the Tk thread so _stop() never races the worker's assignment.
        self._encode_proc = None
        self._encode_reader_thread = None
        if not self._queue_running and self._proc is None:
            self._stop_btn.configure(state="disabled", text_color=TEXT_DIM)
        if self._is_closing:
            return
        if log_text:
            self._log_msg(log_text if log_text.endswith("\n") else log_text + "\n")
        ok = rc == 0 and os.path.isfile(out_path) and os.path.getsize(out_path) > 0
        if ok:
            self._output_path = out_path
            self._video_last_save = out_path
            self._video_last_crf = request.crf if request is not None else None
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
            # A killed encoder runs no cleanup, and a failed one may have been
            # killed mid-write: remove the temp files it left in the export folder.
            self._cleanup_job_temps(os.path.dirname(out_path), encode_pid)
            if prev_title is not None:
                # The export produced no preview, so restore the header it replaced.
                try:
                    self._preview_title.configure(text=prev_title)
                except tk.TclError:
                    pass
            if cancelled and not failed:
                # _stop() terminated a healthy encode and already logged it. A
                # failure the encoder announced before Stop is still a failure.
                self._status_label.configure(text="Stopped", text_color=WARNING)
            else:
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
