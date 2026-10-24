"""Behavioral checks for widget-free viewer and process components."""

import ctypes
import os
import queue
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from app import gui_helpers
from app.gui_helpers import PROC_DONE_PREFIX
from app.gui_process import read_encode_output, read_generation_output, start_cli_process
from app.gui_viewer import ViewerGeometry


def test_cursor_zoom_keeps_source_point_fixed_with_label_offset() -> None:
    geometry = ViewerGeometry((1000, 800), (400, 300), (440, 340), 1.0, (500, 400))
    cursor = (300.0, 190.0)
    before = geometry.source_point(*cursor)
    zoom, cx, cy = geometry.zoom_target(1.25, cursor)
    after = ViewerGeometry(
        geometry.source_size, geometry.viewport_size, geometry.label_size, zoom, (cx, cy)
    ).source_point(*cursor)
    assert after == pytest.approx(before, abs=1.0)


def test_letterbox_is_not_selectable_and_divider_mapping_roundtrips() -> None:
    geometry = ViewerGeometry((400, 200), (400, 400), (440, 440), 1.0, (200, 100))
    assert geometry.normalized_point(220, 30) is None
    assert geometry.normalized_point(220, 220) == pytest.approx((0.5, 0.5))
    for ratio in (0.1, 0.5, 0.9):
        x = geometry.split_label_x(ratio)
        assert x is not None
        assert geometry.split_ratio(x) == pytest.approx(ratio)


def test_zoom_from_letterboxed_fit_keeps_cursor_anchor_when_crop_allows_it() -> None:
    geometry = ViewerGeometry((400, 200), (400, 400), (440, 440), 1.0, (200, 100))
    cursor = (120.0, 170.0)
    before = geometry.source_point(*cursor)
    zoom, cx, cy = geometry.zoom_target(3.0, cursor)
    after = ViewerGeometry(
        geometry.source_size, geometry.viewport_size, geometry.label_size, zoom, (cx, cy)
    ).source_point(*cursor)
    assert after == pytest.approx(before, abs=1.0)


def test_zoom_out_to_letterbox_keeps_center() -> None:
    geometry = ViewerGeometry((400, 200), (400, 400), (440, 440), 3.0, (200, 100))
    cursor = (220.0, 30.0)
    assert geometry.source_point(*cursor) is not None
    zoom, cx, cy = geometry.zoom_target(1.0 / 3.0, cursor)
    assert zoom == pytest.approx(1.0)
    assert (cx, cy) == pytest.approx((200, 100))


def test_generation_stream_reports_nonzero_completion_and_closes_pipe(tmp_path: Path) -> None:
    proc = start_cli_process(
        [sys.executable, "-X", "utf8", "-c", "print('梦想'); raise SystemExit(7)"], tmp_path
    )
    messages: queue.Queue[str] = queue.Queue()
    read_generation_output(proc, 42, messages)
    assert messages.get_nowait() == "梦想\n"
    assert messages.get_nowait() == f"{PROC_DONE_PREFIX}{id(proc)}:42"
    assert proc.returncode == 7
    assert proc.stdout is not None and proc.stdout.closed


def test_generation_stream_reports_lines_to_callback_before_completion(tmp_path: Path) -> None:
    proc = start_cli_process(
        [
            sys.executable,
            "-c",
            'print(\'\\x00dd-event:{"event":"pid","pid":4242}\')',
        ],
        tmp_path,
    )
    messages: queue.Queue[str] = queue.Queue()
    lines: list[str] = []

    read_generation_output(proc, 42, messages, on_line=lines.append)

    assert lines == ['\x00dd-event:{"event":"pid","pid":4242}\n']
    assert messages.get_nowait() == lines[0]
    assert messages.get_nowait() == f"{PROC_DONE_PREFIX}{id(proc)}:42"


def test_export_stream_reports_failure_and_closes_pipe(tmp_path: Path) -> None:
    proc = start_cli_process(
        [sys.executable, "-c", "print('encoding'); raise SystemExit(3)"], tmp_path
    )
    messages: queue.Queue[tuple] = queue.Queue()
    lines: list[str] = []
    read_encode_output(proc, "output.mp4", 9, messages, on_line=lines.append)
    assert lines == ["encoding\n"]
    assert messages.get_nowait() == ("line", "encoding\n", "output.mp4")
    assert messages.get_nowait() == ("done", 3, "", "output.mp4", 9)
    assert proc.stdout is not None and proc.stdout.closed


def test_process_start_failure_is_not_hidden(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        start_cli_process([str(tmp_path / "missing-interpreter")], tmp_path)


def test_custom_crf_keeps_the_retained_preset_geometry_in_job_command() -> None:
    from app.gui_job import GenerationJob

    job = GenerationJob(
        id=1,
        image_path="input.jpg",
        model_key="googlenet",
        layer_spec="inception4c",
        iterations=2,
        octaves=0,
        octave_scale=1.4,
        lr=0.03,
        size=512,
        jitter=0,
        out_dir="out",
        mode="Dream Zoom",
        video_quality="Custom",
        video_crf=10,
        video_preset="Lossless",
    )
    cmd = job.build_cmd()
    assert cmd[cmd.index("--video-quality") + 1] == "lossless"
    assert cmd[cmd.index("--crf") + 1] == "10"


def test_launcher_outputs_stay_beside_source(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").touch()
    (tmp_path / "Launch.sh").touch()
    with mock.patch.object(gui_helpers, "ROOT", tmp_path):
        assert gui_helpers.resolve_out_dir("") == str(tmp_path / "out")
        assert gui_helpers.resolve_out_dir("artwork") == str(tmp_path / "artwork")


def test_windows_only_checkout_outputs_stay_beside_source(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").touch()
    (tmp_path / "Launch.bat").touch()
    with mock.patch.object(gui_helpers, "ROOT", tmp_path):
        assert gui_helpers.output_base_dir() == tmp_path


def test_installed_outputs_use_home_instead_of_site_packages(tmp_path: Path) -> None:
    package_root = tmp_path / "site-packages"
    package_root.mkdir()
    home = tmp_path / "home"
    with (
        mock.patch.object(gui_helpers, "ROOT", package_root),
        mock.patch.object(Path, "home", return_value=home),
    ):
        base = home / "DeepDream Studio"
        assert gui_helpers.resolve_out_dir("") == str(base / "out")
        assert gui_helpers.resolve_out_dir("artwork") == str(base / "artwork")
        explicit = tmp_path / "chosen"
        assert gui_helpers.resolve_out_dir(str(explicit)) == str(explicit)
        assert not list(package_root.iterdir())


def test_pid_is_running_probes_without_killing() -> None:
    assert gui_helpers.pid_is_running(os.getpid()) is True
    assert gui_helpers.pid_is_running(999999999) is False
    assert gui_helpers.pid_is_running(None) is False


@pytest.mark.skipif(sys.platform != "win32", reason="probes through kernel32")
@pytest.mark.parametrize(
    ("last_error", "running"),
    [
        (5, True),  # ERROR_ACCESS_DENIED: a live process owned by another account
        (87, False),  # ERROR_INVALID_PARAMETER: no process has this pid
    ],
)
def test_pid_is_running_on_windows_reads_why_open_failed(
    monkeypatch: pytest.MonkeyPatch, last_error: int, running: bool
) -> None:
    fake_kernel32 = SimpleNamespace(
        OpenProcess=lambda access, inherit, pid: None,
        GetExitCodeProcess=lambda handle, code: 0,
        CloseHandle=lambda handle: 1,
    )
    monkeypatch.setattr(ctypes, "WinDLL", lambda *args, **kwargs: fake_kernel32)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: last_error)
    assert gui_helpers.pid_is_running(1234) is running


def _backdate(*paths: Path) -> None:
    """Age files past the grace window, as if their writer had gone quiet."""
    stamp = time.time() - 3600
    for path in paths:
        os.utime(path, (stamp, stamp))


def test_startup_sweep_reclaims_only_temps_from_dead_processes(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    dead = 999999999
    live = os.getpid()

    orphan = out / f"clip.mp4.p{dead}.temp.mp4"
    orphan.write_bytes(b"partial")
    writing = out / f"clip.mp4.p{live}.temp.mp4"
    writing.write_bytes(b"in flight")
    preserved = out / f"shot.png.p{dead}.tmp"
    preserved.write_bytes(b"finished")
    (out / f"shot.png.p{dead}.tmp.complete").write_text("", encoding="utf-8")
    stale_marker = out / f"gone.png.p{dead}.tmp.complete"
    stale_marker.write_text("", encoding="utf-8")

    frames = out / f"frames_job1_p{dead}"
    frames.mkdir()
    (frames / ".frame_000001.tmp.png").write_bytes(b"partial")
    (frames / "meta.json.tmp").write_text("{}", encoding="utf-8")
    (frames / "frame_000001.png").write_bytes(b"finished")
    live_frames = out / f"frames_job2_p{live}"
    live_frames.mkdir()
    (live_frames / ".frame_000001.tmp.png").write_bytes(b"in flight")

    # Only temps that have stopped changing count as orphaned.
    _backdate(orphan, frames / ".frame_000001.tmp.png", frames / "meta.json.tmp")

    removed = gui_helpers.sweep_orphaned_temp_files(out)

    assert removed == 4
    assert not orphan.exists()
    assert writing.exists()
    assert preserved.exists()
    assert (out / f"shot.png.p{dead}.tmp.complete").exists()
    assert not stale_marker.exists()
    # Finished frames survive; only the atomic-save leftovers go.
    assert (frames / "frame_000001.png").exists()
    assert not (frames / ".frame_000001.tmp.png").exists()
    assert not (frames / "meta.json.tmp").exists()
    assert (live_frames / ".frame_000001.tmp.png").exists()


def test_sweep_spares_temps_that_are_still_being_written(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    dead = 999999999
    temp = out / f"clip.mp4.p{dead}.temp.mp4"
    temp.write_bytes(b"in flight")
    frames = out / f"frames_job1_p{dead}"
    frames.mkdir()
    frame_temp = frames / ".frame_000001.tmp.png"
    frame_temp.write_bytes(b"in flight")

    # The pid check cannot see a writer on another account or machine, so a
    # fresh write is what keeps its temps alive.
    assert gui_helpers.sweep_orphaned_temp_files(out) == 0
    assert temp.exists()
    assert frame_temp.exists()

    _backdate(temp, frame_temp)
    assert gui_helpers.sweep_orphaned_temp_files(out) == 2
    assert not temp.exists()
    assert not frame_temp.exists()
