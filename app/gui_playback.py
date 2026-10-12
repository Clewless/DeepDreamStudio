"""In-window video playback: timeline bar, scrubbing, and async frame loading.

Builds and drives the preview's video bar and the frame loader that decodes
dreamed PNG sequences or saved .mp4 files off the Tk thread. Mixed into
DeepDreamGUI; the owning window drives state setup via
_init_video_playback_state and builds the bar via _build_main_video_bar.
"""

import os
import tkinter as tk
from collections.abc import Callable
from typing import Any

from .gui_helpers import (
    FrameCache,
    VideoFrameLoader,
    format_frame_label,
    timeline_index_to_t,
    timeline_t_to_index,
)
from .gui_widgets import (
    ACCENT,
    BG_PANEL,
    BORDER,
    TEXT_DIM,
    TEXT_MAIN,
    ModernButton,
    ModernFont,
    ModernFrame,
    ModernLabel,
    ModernSlider,
    Tooltip,
)


class VideoPlaybackMixin:
    """Timeline scrubber, play/pause, and async decode for in-window video."""

    # Methods owned by DeepDreamGUI.
    _render_viewer: Callable[..., Any]
    _show_output_preview: Callable[..., Any]
    _on_view_image_toggle: Callable[..., Any]
    _on_view_timelapse_toggle: Callable[..., Any]
    _show_in_folder: Callable[..., Any]
    # Tk scheduling methods; typed Any so the MRO merge with tk.Misc stays valid.
    after: Any
    after_cancel: Any

    # Shared window state.
    _is_closing: bool
    _preview_title: Any
    _viewer_mode: str

    # Widget handles populated during build.
    _video_bar: Any
    _video_play_btn: Any
    _video_slider: Any
    _video_pos_label: Any
    _view_image_btn: Any
    _view_timelapse_btn: Any
    _open_dir_btn: Any

    # Playback state.
    _video_kind: str | None
    _video_paths: list[str]
    _video_mp4: str | None
    _video_index: int
    _video_total: int
    _video_fps: float
    _video_playing: bool
    _video_after_id: Any
    _video_slider_sync: bool
    _video_cache: FrameCache
    _video_loader: VideoFrameLoader | None
    _video_poll_after_id: Any
    _video_requested_index: int
    _video_probe_token: int
    _video_pending_title: str | None

    def _init_video_playback_state(self) -> None:
        """Create the playback handles, state, and decode worker."""
        self._video_bar = None
        self._video_play_btn = None
        self._video_slider = None
        self._video_pos_label = None
        self._view_image_btn = None
        self._view_timelapse_btn = None
        self._open_dir_btn = None
        self._video_kind = None  # None | "frames" | "mp4"
        self._video_paths = []
        self._video_mp4 = None
        self._video_index = 0
        self._video_total = 0
        self._video_fps = 30.0
        self._video_playing = False
        self._video_after_id = None
        self._video_slider_sync = False
        self._video_cache = FrameCache()
        # Decodes and frame counts run on a worker thread so a long mp4 cannot
        # freeze the window; results are applied by _poll_video_loader.
        self._video_loader = None
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
