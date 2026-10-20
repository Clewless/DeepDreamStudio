"""Media boundary regressions: orientation and preservation of completed output."""

import errno
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest
from PIL import Image

from app import media_io


def test_load_image_reports_oriented_native_size_and_accepts_path(tmp_path: Path) -> None:
    source = tmp_path / "oriented.jpg"
    exif = Image.Exif()
    exif[274] = 6  # Rotate the decoded image 90 degrees clockwise.
    Image.new("RGB", (12, 8), (25, 100, 200)).save(source, exif=exif)

    tensor, native_size = media_io.load_image_with_size(source, 6)
    assert media_io.load_image_native_size(source) == (12, 8)
    assert native_size == (8, 12)
    assert tuple(tensor.shape) == (1, 3, 6, 4)
    output = tmp_path / "roundtrip.png"
    media_io.save_image(tensor, output)
    with Image.open(output) as image:
        assert image.size == (4, 6)
        assert image.mode == "RGB"


def test_failed_image_encode_preserves_existing_destination(tmp_path: Path) -> None:
    destination = tmp_path / "image.png"
    destination.write_bytes(b"original output")
    tensor = media_io.pil_to_tensor(Image.new("RGB", (4, 4)))

    def fail_encode(_image: Image.Image, path: media_io.MediaPath, **_options: str) -> None:
        Path(path).write_bytes(b"incomplete encode")
        raise OSError("disk full")

    with mock.patch.object(Image.Image, "save", fail_encode):
        with pytest.raises(OSError, match="disk full"):
            media_io.save_image(tensor, destination)
    assert destination.read_bytes() == b"original output"
    assert not Path(media_io.atomic_tmp_path(destination)).exists()


@pytest.mark.parametrize("kind", ["image", "video", "preview"])
def test_locked_destination_preserves_output_according_to_policy(tmp_path: Path, kind: str) -> None:
    destination = tmp_path / ("video.mp4" if kind == "video" else "image.png")
    destination.write_bytes(b"original output")
    temporary = Path(media_io.atomic_tmp_path(destination))
    frame = np.full((4, 4, 3), 100, dtype=np.uint8)
    if kind == "video":
        temporary.write_bytes(b"completed encode")

    with mock.patch.object(media_io.os, "replace", side_effect=PermissionError("locked")):
        if kind == "preview":
            assert not media_io.save_preview_frame(frame, destination)
        else:
            with pytest.raises(PermissionError, match="locked"):
                if kind == "video":
                    media_io.publish_media(temporary, destination)
                else:
                    media_io.save_image(media_io.pil_to_tensor(Image.fromarray(frame)), destination)

    assert destination.read_bytes() == b"original output"
    if kind == "preview":
        assert not temporary.exists()
    elif kind == "video":
        assert temporary.read_bytes() == b"completed encode"
    else:
        with Image.open(temporary) as completed:
            np.testing.assert_array_equal(np.asarray(completed), frame)

    if kind != "preview":
        # The GUI sweeps on failure, stale completion, Stop, and close using
        # this same method. Its sweep must keep the engine's recovery output.
        import os

        from app.gui import DeepDreamGUI

        partial = tmp_path / f"unfinished.p{os.getpid()}.tmp"
        partial.write_bytes(b"partial")
        DeepDreamGUI._cleanup_job_temps(mock.Mock(), str(tmp_path), os.getpid())
        assert temporary.exists()
        assert not partial.exists()


def test_successful_publication_leaves_no_recovery_marker(tmp_path: Path) -> None:
    destination = tmp_path / "image.png"
    media_io.save_image(media_io.pil_to_tensor(Image.new("RGB", (4, 4))), destination)
    assert list(tmp_path.iterdir()) == [destination]


def test_sidecar_failure_does_not_discard_a_completed_video(tmp_path: Path) -> None:
    destination = tmp_path / "video.mp4"
    temporary = Path(media_io.atomic_tmp_path(destination))
    temporary.write_bytes(b"completed encode")

    name_too_long = OSError(errno.ENAMETOOLONG, "File name too long")
    with mock.patch.object(media_io, "open", side_effect=name_too_long, create=True):
        media_io.publish_media(temporary, destination)

    assert destination.read_bytes() == b"completed encode"
    assert list(tmp_path.iterdir()) == [destination]


def test_locked_destination_retries_a_failed_recovery_marker(tmp_path: Path) -> None:
    destination = tmp_path / "video.mp4"
    temporary = Path(media_io.atomic_tmp_path(destination))
    temporary.write_bytes(b"completed encode")
    marker = Path(f"{temporary}.complete")
    real_open = open
    attempts = {"count": 0}

    def flaky_open(*args: Any, **kwargs: Any):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise OSError(errno.ENAMETOOLONG, "File name too long")
        return real_open(*args, **kwargs)

    with (
        mock.patch.object(media_io, "open", side_effect=flaky_open, create=True),
        mock.patch.object(media_io.os, "replace", side_effect=PermissionError("locked")),
    ):
        with pytest.raises(PermissionError, match="locked"):
            media_io.publish_media(temporary, destination)

    # The retry must leave the sidecar cleanup looks for, so the preserved
    # output survives a later GUI sweep.
    assert marker.exists()
    assert temporary.read_bytes() == b"completed encode"
