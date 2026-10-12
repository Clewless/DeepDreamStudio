"""Media boundary regressions: orientation and preservation of completed output."""

from pathlib import Path
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
                    media_io.publish_video(temporary, destination)
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
