"""Sequence isolation and actual encoded video fidelity regressions."""

from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest
import torch
from PIL import Image

from app import deepdream

_ZOOM_KWARGS: dict[str, Any] = {
    "model": None,
    "model_key": "googlenet",
    "targets": ["inception4c"],
    "iterations_per_frame": 0,
    "lr": 0.03,
    "jitter": 0,
    "device": "cpu",
}


def _zoom_sequence(frames: Path, count: int, **extra: Any) -> None:
    deepdream.dream_zoom_animation(
        img=torch.zeros(1, 3, 32, 32),
        num_frames=count,
        zoom_scale=1.0,
        zoom_rot=0.0,
        out_path=None,
        frames_dir=str(frames),
        **{**_ZOOM_KWARGS, **extra},
    )


@pytest.mark.parametrize("mode", ["zoom", "video"])
def test_fresh_sequence_refuses_reused_folder_without_changing_outputs(
    tmp_path: Path, mode: str
) -> None:
    frames = tmp_path / "frames"
    source = tmp_path / "source.mp4"
    source.touch()
    common: dict[str, Any] = dict(_ZOOM_KWARGS)

    def generate(count: int) -> None:
        if mode == "zoom":
            _zoom_sequence(frames, count, **common)
        else:
            with (
                mock.patch.object(deepdream.iio, "immeta", return_value={"fps": 30}),
                mock.patch.object(
                    deepdream.iio,
                    "imiter",
                    # A fresh iterator per call: the preflight and the main loop
                    # each decode the input independently.
                    side_effect=lambda *a, **k: iter(
                        [np.zeros((32, 32, 3), dtype=np.uint8)] * count
                    ),
                ),
            ):
                deepdream.dream_video_file(
                    video_in_path=str(source),
                    video_out_path=None,
                    blend_alpha=0.5,
                    frames_dir=str(frames),
                    **common,
                )

    generate(3)
    original = {path.name: path.read_bytes() for path in frames.iterdir()}
    with pytest.raises(SystemExit, match="not empty"):
        generate(1)
    assert {path.name: path.read_bytes() for path in frames.iterdir()} == original


def test_fresh_sequence_refuses_foreign_files(tmp_path: Path) -> None:
    foreign = tmp_path / "important.txt"
    foreign.write_text("keep me", encoding="utf-8")
    with pytest.raises(SystemExit, match="not empty"):
        deepdream.prepare_video_frames_dir(str(tmp_path))
    assert foreign.read_text(encoding="utf-8") == "keep me"


def test_interrupted_atomic_saves_do_not_block_a_fresh_sequence(tmp_path: Path) -> None:
    """A kill during an atomic frame or metadata write must not need manual cleanup."""
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / ".frame_000007.tmp.png").write_bytes(b"partial")
    (frames / "meta.json.tmp").write_text("{", encoding="utf-8")

    deepdream.prepare_video_frames_dir(str(frames))

    assert list(frames.iterdir()) == []


def test_interrupted_atomic_saves_do_not_block_iteration_frames(tmp_path: Path) -> None:
    stale_temp = tmp_path / ".frame_000002.tmp.png"
    stale_temp.write_bytes(b"partial")
    iteration = tmp_path / "o1_i1.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(iteration)
    image_temp = tmp_path / "o1_i1.png.p4242.tmp"
    image_temp.write_bytes(b"partial")

    deepdream.clear_stale_frames(tmp_path, allow_delete=True)

    assert not stale_temp.exists()
    assert not iteration.exists()
    assert not image_temp.exists()


def test_killed_iteration_save_is_swept_without_clear(tmp_path: Path) -> None:
    """A kill mid-write leaves an unmarked temp that needs no --clear-frames."""
    (tmp_path / "o1_i1.png.p4242.tmp").write_bytes(b"partial")

    deepdream.clear_stale_frames(tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_completed_iteration_save_survives_rerun_until_recovered(tmp_path: Path) -> None:
    """A finished temp kept for recovery is never swept, not even by a clear.

    The locked-destination failure path preserves the completed output and
    tells the user where it is; a re-run (GUI retry passes --clear-frames)
    must leave it recoverable instead of deleting it first.
    """
    temp = tmp_path / "o1_i1.png.p4242.tmp"
    marker = tmp_path / "o1_i1.png.p4242.tmp.complete"
    temp.write_bytes(b"finished")
    marker.write_bytes(b"")

    deepdream.clear_stale_frames(tmp_path, allow_delete=True)

    assert temp.read_bytes() == b"finished"
    assert marker.exists()


def test_stale_completion_marker_without_its_temp_is_swept(tmp_path: Path) -> None:
    """A sidecar left by a kill after replacement is stale, not a recovery."""
    (tmp_path / "o1_i1.png.p4242.tmp.complete").write_bytes(b"")

    deepdream.clear_stale_frames(tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_video_clear_frames_waits_for_the_input_frame_to_decode(tmp_path: Path) -> None:
    """A broken input video must not wipe an existing sequence via --clear-frames."""
    frames = tmp_path / "frames"
    _zoom_sequence(frames, 2)
    original = {path.name: path.read_bytes() for path in frames.iterdir()}
    source = tmp_path / "source.mp4"
    source.touch()

    with (
        mock.patch.object(deepdream.iio, "imiter", side_effect=OSError("cannot decode")),
        pytest.raises(OSError, match="cannot decode"),
    ):
        deepdream.dream_video_file(
            video_in_path=str(source),
            video_out_path=None,
            blend_alpha=0.5,
            max_size=64,
            frames_dir=str(frames),
            clear_frames=True,
            **_ZOOM_KWARGS,
        )

    assert {path.name: path.read_bytes() for path in frames.iterdir()} == original


def test_video_clear_frames_waits_for_the_input_size_check(tmp_path: Path) -> None:
    """A too-small input frame must fail before --clear-frames removes the sequence."""
    frames = tmp_path / "frames"
    _zoom_sequence(frames, 1)
    original = {path.name: path.read_bytes() for path in frames.iterdir()}
    source = tmp_path / "source.mp4"
    source.touch()
    tiny_frame = np.zeros((8, 8, 3), dtype=np.uint8)

    with (
        mock.patch.object(deepdream.iio, "imiter", return_value=iter([tiny_frame])),
        pytest.raises(SystemExit, match="Video frame"),
    ):
        deepdream.dream_video_file(
            video_in_path=str(source),
            video_out_path=None,
            blend_alpha=0.5,
            max_size=64,
            frames_dir=str(frames),
            clear_frames=True,
            **_ZOOM_KWARGS,
        )

    assert {path.name: path.read_bytes() for path in frames.iterdir()} == original


def test_clear_frames_starts_a_new_sequence_in_the_same_folder(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    _zoom_sequence(frames, 3)

    _zoom_sequence(frames, 2, clear_frames=True)

    assert sorted(path.name for path in frames.iterdir()) == [
        "frame_000001.png",
        "frame_000002.png",
        "meta.json",
    ]
    # Stale metadata would make a later --resume compare against the old run.
    assert deepdream.read_video_frames_meta(frames)["total_frames"] == 2


def test_clear_frames_never_removes_files_it_did_not_write(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    _zoom_sequence(frames, 2)
    foreign = frames / "holiday.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(foreign)
    original = {path.name: path.read_bytes() for path in frames.iterdir()}

    with pytest.raises(SystemExit, match="not part"):
        deepdream.prepare_video_frames_dir(str(frames), clear=True)

    assert {path.name: path.read_bytes() for path in frames.iterdir()} == original


def test_clear_video_frames_keeps_recoverable_iteration_output(tmp_path: Path) -> None:
    """--clear-frames replaces the sequence but never an output kept for recovery."""
    frames = tmp_path / "frames"
    _zoom_sequence(frames, 2)
    temp = frames / "o1_i1.png.p4242.tmp"
    marker = frames / "o1_i1.png.p4242.tmp.complete"
    temp.write_bytes(b"finished")
    marker.write_bytes(b"")

    deepdream.prepare_video_frames_dir(str(frames), clear=True)

    assert temp.read_bytes() == b"finished"
    assert marker.exists()
    assert sorted(path.name for path in frames.iterdir()) == [
        "o1_i1.png.p4242.tmp",
        "o1_i1.png.p4242.tmp.complete",
    ]


@pytest.mark.parametrize("scale", [1, 2])
@pytest.mark.parametrize("timelapse", [False, True])
def test_video_encode_uses_yuv420_and_applies_scale(
    tmp_path: Path, scale: int, timelapse: bool
) -> None:
    # Smooth gradient (not noise): 4:2:0 + Lanczos keeps low-frequency color.
    yy, xx = np.mgrid[0:32, 0:32]
    rgb = np.stack([xx * 8 % 256, yy * 8 % 256, ((xx + yy) * 4) % 256], axis=-1).astype(np.uint8)
    frames = tmp_path / "frames"
    frames.mkdir()
    Image.fromarray(rgb).save(frames / ("o1_i1.png" if timelapse else "frame_000001.png"))
    output = tmp_path / "video.mp4"
    compile_frames = deepdream.compile_timelapse if timelapse else deepdream.compile_video_frames
    compile_frames(str(frames), str(output), fps=1, crf=12, scale=scale)

    meta = deepdream.iio.immeta(output)
    assert meta["pix_fmt"].startswith("yuv420p")
    assert meta["size"] == (32 * scale, 32 * scale)
    reader = deepdream.iio.imiter(output)
    try:
        decoded = next(reader)
    finally:
        close = getattr(reader, "close", None)
        if close is not None:
            close()
    assert decoded.shape[0] == 32 * scale and decoded.shape[1] == 32 * scale
    # Decode and compare in source space (downscale the encode when scaled).
    if scale > 1:
        decoded_pil = Image.fromarray(decoded).resize((32, 32), Image.Resampling.LANCZOS)
        decoded = np.asarray(decoded_pil)
    assert np.abs(rgb.astype(int) - decoded.astype(int)).mean() <= 6


@pytest.mark.parametrize(
    ("compiler", "frame_name"),
    [
        (deepdream.compile_video_frames, "frame_000001.png"),
        (deepdream.compile_timelapse, "o1_i1.png"),
    ],
)
def test_video_export_warns_before_overwriting(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], compiler: Any, frame_name: str
) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(frames / frame_name)
    output = tmp_path / "video.mp4"
    output.write_bytes(b"previous output")

    compiler(str(frames), str(output), fps=1)

    assert "already exists" in capsys.readouterr().err


def _video_resume_kwargs(source: Path, frames: Path, output: Path | None) -> dict[str, Any]:
    return {
        "model": None,
        "model_key": "googlenet",
        "video_in_path": str(source),
        "video_out_path": str(output) if output is not None else None,
        "targets": ["inception4c"],
        "iterations_per_frame": 0,
        "lr": 0.03,
        "jitter": 0,
        "blend_alpha": 0.5,
        "device": "cpu",
        "max_size": 64,
        "frames_dir": str(frames),
        "resume": True,
    }


def _write_frames(frames: Path, count: int) -> None:
    frames.mkdir()
    for index in range(1, count + 1):
        Image.fromarray(np.full((32, 32, 3), index * 40, dtype=np.uint8)).save(
            frames / f"frame_{index:06d}.png"
        )


def test_video_resume_refuses_saved_sequence_longer_than_source(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    _write_frames(frames, 4)
    source = tmp_path / "source.mp4"
    source.touch()
    output = tmp_path / "out.mp4"
    original_frames = {path.name: path.read_bytes() for path in frames.glob("frame_*.png")}

    with (
        mock.patch.object(deepdream.iio, "immeta", return_value={"fps": 30, "duration": 2 / 30}),
        mock.patch.object(
            deepdream.iio,
            "imiter",
            side_effect=lambda *a, **k: iter([np.zeros((32, 32, 3), dtype=np.uint8)] * 2),
        ),
        pytest.raises(SystemExit, match="saved sequence has 4 frames"),
    ):
        deepdream.dream_video_file(**_video_resume_kwargs(source, frames, output))

    assert not output.exists()
    assert not (frames / "meta.json").exists()
    assert {path.name: path.read_bytes() for path in frames.glob("frame_*.png")} == original_frames


def test_video_resume_of_complete_sequence_encodes_all_saved_frames(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    _write_frames(frames, 2)
    source = tmp_path / "source.mp4"
    source.touch()
    output = tmp_path / "out.mp4"

    def publish_silent(temp_out: str, _source: str, final_out: str) -> str:
        deepdream._publish_media(temp_out, final_out)
        return final_out

    with (
        mock.patch.object(deepdream.iio, "immeta", return_value={"fps": 30, "duration": 2 / 30}),
        mock.patch.object(
            deepdream.iio,
            "imiter",
            side_effect=lambda *a, **k: iter([np.zeros((32, 32, 3), dtype=np.uint8)] * 2),
        ),
        mock.patch.object(deepdream, "_remux_audio", side_effect=publish_silent),
    ):
        deepdream.dream_video_file(**_video_resume_kwargs(source, frames, output))

    assert output.is_file()
    assert len(list(deepdream.iio.imiter(str(output)))) == 2
