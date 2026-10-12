"""Behavioral checks for widget-free viewer and process components."""

import queue
import sys
from pathlib import Path

import pytest

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


def test_export_stream_reports_failure_and_closes_pipe(tmp_path: Path) -> None:
    proc = start_cli_process(
        [sys.executable, "-c", "print('encoding'); raise SystemExit(3)"], tmp_path
    )
    messages: queue.Queue[tuple] = queue.Queue()
    read_encode_output(proc, "output.mp4", 9, messages)
    assert messages.get_nowait() == ("line", "encoding\n", "output.mp4")
    assert messages.get_nowait() == ("done", 3, "", "output.mp4", 9)
    assert proc.stdout is not None and proc.stdout.closed


def test_process_start_failure_is_not_hidden(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        start_cli_process([str(tmp_path / "missing-interpreter")], tmp_path)
