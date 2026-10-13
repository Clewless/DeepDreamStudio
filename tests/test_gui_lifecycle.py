"""Headless lifecycle regressions using deterministic schedulers and process doubles."""

from collections.abc import Callable
from typing import cast
from unittest import mock

import pytest
from PIL import Image

from app import gui
from app.gui_helpers import FrameCache
from app.gui_playback import VideoPlaybackMixin


@pytest.fixture
def playback() -> VideoPlaybackMixin:
    player = VideoPlaybackMixin()
    player._is_closing = False
    player._viewer_mode = "video"
    player._video_kind = "frames"
    player._video_paths = ["a.png", "b.png", "c.png"]
    player._video_mp4 = None
    player._video_index = 2
    player._video_total = 3
    player._video_fps = 30.0
    player._video_playing = False
    player._video_after_id = None
    player._video_poll_after_id = None
    player._video_play_btn = mock.Mock()
    player._video_bar = mock.Mock()
    player._video_cache = FrameCache()
    player._video_loader = None
    player._video_requested_index = 2
    player._video_probe_token = 11
    player._render_viewer = mock.Mock()
    player.after = mock.Mock(return_value="next-tick")
    player.after_cancel = mock.Mock()
    return player


def test_playback_restarts_at_end_and_stop_prevents_late_tick(playback: VideoPlaybackMixin) -> None:
    with mock.patch.object(playback, "_show_video_frame") as show:
        playback._start_video_playback()
        assert playback._video_index == 0
        assert playback._video_playing
        playback.after.assert_called_once_with(33, playback._video_tick)
        playback._stop_video_playback()
        playback.after_cancel.assert_called_once_with("next-tick")
        assert playback._video_after_id is None
        playback.after.reset_mock()
        playback._video_tick()  # A callback already delivered before cancellation.
        show.assert_not_called()
        playback.after.assert_not_called()


def test_playback_tick_loops_and_replaces_existing_timer(playback: VideoPlaybackMixin) -> None:
    playback._video_playing = True
    playback._video_after_id = "old-tick"
    playback._schedule_video_tick()
    playback.after_cancel.assert_called_once_with("old-tick")
    playback.after.reset_mock()
    with mock.patch.object(playback, "_show_video_frame") as show:
        playback._video_tick()
        show.assert_called_once_with(0)
    playback.after.assert_called_once_with(33, playback._video_tick)


def test_closing_window_does_not_reschedule_playback(playback: VideoPlaybackMixin) -> None:
    playback._is_closing = True
    playback._video_playing = True
    with mock.patch.object(playback, "_show_video_frame") as show:
        playback._video_tick()
    show.assert_not_called()
    playback.after.assert_not_called()


@pytest.mark.parametrize("generation,source", [(6, "c.png"), (7, "old.png")])
def test_stale_decode_cannot_change_cache_or_view(
    playback: VideoPlaybackMixin, generation: int, source: str
) -> None:
    frame = Image.new("RGB", (4, 4))
    with mock.patch.object(playback, "_loader_generation", return_value=7):
        playback._on_video_frame_result("frames", source, 2, frame, None, generation)
    assert playback._video_cache.get(("frames", source, 2)) is None
    cast(mock.Mock, playback._render_viewer).assert_not_called()


def test_previous_scrub_frame_warms_cache_without_repainting(playback: VideoPlaybackMixin) -> None:
    frame = Image.new("RGB", (4, 4))
    with mock.patch.object(playback, "_loader_generation", return_value=7):
        playback._on_video_frame_result("frames", "a.png", 0, frame, None, 7)
        assert playback._video_cache.get(("frames", "a.png", 0)) is frame
        cast(mock.Mock, playback._render_viewer).assert_not_called()
        playback._on_video_frame_result("frames", "c.png", 2, frame, None, 7)
    cast(mock.Mock, playback._render_viewer).assert_called_once_with()


@pytest.mark.parametrize("generation,token", [(6, 11), (7, 10)])
def test_stale_video_probe_does_not_replace_timeline(
    playback: VideoPlaybackMixin, generation: int, token: int
) -> None:
    playback._video_kind = "mp4"
    playback._video_mp4 = "clip.mp4"
    with (
        mock.patch.object(playback, "_loader_generation", return_value=7),
        mock.patch.object(playback, "_show_video_frame") as show,
    ):
        playback._on_video_probe_result("clip.mp4", 100, 60.0, token, generation)
    assert playback._video_total == 3
    assert playback._video_fps == 30.0
    show.assert_not_called()


def test_stop_terminates_generation_and_export_without_starting_next_job() -> None:
    window = mock.Mock(spec=gui.DeepDreamGUI)
    generation = mock.Mock(pid=101)
    generation.poll.return_value = None
    export = mock.Mock(pid=102)
    export.poll.return_value = None
    window._proc = generation
    window._encode_proc = export
    window._current_job = mock.Mock(id=9)
    window._aborted_jobs = set()
    window._queue_running = True
    with mock.patch.object(gui, "terminate_process") as terminate:
        gui.DeepDreamGUI._stop(window)
    assert terminate.call_args_list == [mock.call(generation), mock.call(export)]
    assert not window._queue_running
    assert window._aborted
    assert window._aborted_jobs == {9}
    assert window._video_export_cancelled
    window._cancel_next_job_timer.assert_called_once_with()
    window._process_next_job.assert_not_called()


def test_close_cancels_timers_and_reaps_both_children_before_cleanup() -> None:
    window = mock.Mock(spec=gui.DeepDreamGUI)
    order: list[str] = []
    generation = mock.Mock(pid=101)
    export = mock.Mock(pid=102)
    generation.poll.return_value = export.poll.return_value = None
    generation.wait.side_effect = lambda **kw: order.append("generation reaped")
    export.wait.side_effect = lambda **kw: order.append("export reaped")
    window._proc = generation
    window._encode_proc = export
    window._current_job = mock.Mock(id=9, out_dir="output")
    window._aborted_jobs = set()
    window._poll_after_id = "log-poll"
    window._video_poll_after_id = "video-poll"
    window._video_frames_dir = "output/frames"
    window._slider_registry = []
    window._video_loader = mock.Mock()
    window._video_export_pid = export.pid
    window._video_export_target = "export/clip.mp4"
    window._proc_pids = {id(generation): generation.pid}
    reader = mock.Mock()

    def finish_reader(**kw: object) -> None:
        order.append("reader joined")
        window._proc_pids[id(generation)] = 303

    reader.join.side_effect = finish_reader
    window._proc_reader_threads = {id(generation): reader}
    encode_reader = mock.Mock()

    def finish_encode_reader(**kw: object) -> None:
        order.append("encode reader joined")
        window._video_export_pid = 606

    encode_reader.join.side_effect = finish_encode_reader
    window._encode_reader_thread = encode_reader
    window._cleanup_partial_frames.side_effect = lambda: order.append("frames cleaned")

    def immediate_thread(*, target: Callable[[], None], daemon: bool) -> mock.Mock:
        thread = mock.Mock()
        thread.start.side_effect = target
        return thread

    with (
        mock.patch.object(gui, "terminate_process") as terminate,
        mock.patch.object(gui.threading, "Thread", side_effect=immediate_thread),
    ):
        gui.DeepDreamGUI._on_close(window)
    assert order == [
        "generation reaped",
        "export reaped",
        "reader joined",
        "encode reader joined",
        "frames cleaned",
    ]
    assert terminate.call_args_list == [mock.call(generation), mock.call(export)]
    assert window.after_cancel.call_args_list == [mock.call("log-poll"), mock.call("video-poll")]
    window._stop_video_playback.assert_called_once_with()
    window._video_loader.close.assert_called_once_with()
    window._cancel_next_job_timer.assert_called_once_with()
    window.destroy.assert_called_once_with()
    assert window._is_closing
    assert window._poll_after_id is None
    assert window._video_poll_after_id is None
    window._cleanup_job_temps.assert_any_call("output", 606)
    window._cleanup_job_temps.assert_any_call("export", 606)
    window._cleanup_job_temps.assert_any_call("output", 303)
    reader.join.assert_called_once_with()
    encode_reader.join.assert_called_once_with()
