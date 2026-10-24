"""Regression checks for incomplete or unsafe downloadable release archives."""

import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from scripts.release_bundle import REQUIRED_FILES, build_bundle, verify_bundle


def test_bundle_rejects_missing_launcher_before_running_setup(tmp_path: Path) -> None:
    bundle = tmp_path / "incomplete.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        for name in REQUIRED_FILES - {"Launch.bat"}:
            archive.writestr(f"deepdream_studio-0.1/{name}", "")
    with pytest.raises(ValueError, match=r"missing: Launch\.bat"):
        verify_bundle(bundle)


@pytest.mark.parametrize(
    "omitted",
    [
        "app/dream_engine.py",
        "app/gui_classifier.py",
        "app/gui_export.py",
        "app/gui_layers.py",
        "app/gui_playback.py",
        "app/gui_process.py",
        "app/gui_video.py",
        "app/gui_viewer.py",
        "app/model_data.py",
        "app/media_io.py",
        "app/models.py",
        "app/settings.py",
    ],
)
def test_bundle_rejects_missing_application_module(tmp_path: Path, omitted: str) -> None:
    root = Path(__file__).resolve().parent.parent
    names = REQUIRED_FILES | {
        path.relative_to(root).as_posix() for path in (root / "app").glob("*.py")
    }
    bundle = tmp_path / "missing-module.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        for name in names - {omitted}:
            archive.write(root / name, f"deepdream_studio-0.1/{name}")
    with pytest.raises(ValueError, match=omitted):
        verify_bundle(bundle)


def test_bundle_rejects_path_traversal(tmp_path: Path) -> None:
    bundle = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("project/../../escaped.txt", "unexpected")
    with pytest.raises(ValueError, match="unsafe archive paths"):
        verify_bundle(bundle)
    assert not (tmp_path / "escaped.txt").exists()


def test_conversion_preserves_launcher_bytes_and_executable_mode(tmp_path: Path) -> None:
    sdist = tmp_path / "project.tar.gz"
    content = b"#!/usr/bin/env bash\necho launch\n"
    with tarfile.open(sdist, "w:gz") as archive:
        member = tarfile.TarInfo("project/Launch.sh")
        member.mode = 0o755
        member.size = len(content)
        archive.addfile(member, io.BytesIO(content))
    bundle = build_bundle(sdist, tmp_path)
    with zipfile.ZipFile(bundle) as archive:
        assert archive.read("project/Launch.sh") == content
        assert (archive.getinfo("project/Launch.sh").external_attr >> 16) & 0o777 == 0o755
