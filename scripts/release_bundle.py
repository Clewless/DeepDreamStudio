"""Create and smoke-check the launcher bundle from a built source distribution.

Uses only the standard library so packaging CI does not need the ML stack.
Run: python scripts/release_bundle.py dist
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

REQUIRED_FILES = {
    "Launch.bat",
    "Launch.sh",
    "CLI/LaunchCLI.bat",
    "CLI/LaunchCLI.sh",
    "CLI/README.md",
    "README.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "pyproject.toml",
    "requirements.txt",
    "requirements-cuda.txt",
    "requirements-dev.txt",
    "constraints-runtime.txt",
    "app/__init__.py",
    "app/bootstrap.py",
    "app/check_deps.py",
    "app/deepdream.py",
    "app/dream_engine.py",
    "app/gui.py",
    "app/gui_classifier.py",
    "app/gui_export.py",
    "app/gui_helpers.py",
    "app/gui_job.py",
    "app/gui_layers.py",
    "app/gui_playback.py",
    "app/gui_process.py",
    "app/gui_video.py",
    "app/gui_viewer.py",
    "app/gui_widgets.py",
    "app/imagenet_categories.py",
    "app/media_io.py",
    "app/model_data.py",
    "app/models.py",
    "app/protocol.py",
    "tests/conftest.py",
    "docs/images/example-before-after.jpg",
    "docs/images/gui-result.png",
    "docs/architecture.md",
    "scripts/release_bundle.py",
    "scripts/smoke_installed.py",
}


def build_bundle(sdist: Path, destination: Path) -> Path:
    """Copy regular sdist files into a ZIP, preserving executable permissions."""
    with tarfile.open(sdist, "r:gz") as source:
        members = [member for member in source.getmembers() if member.isfile()]
        roots = {PurePosixPath(member.name).parts[0] for member in members}
        if len(roots) != 1:
            raise ValueError("Source distribution must have exactly one root directory")
        root = roots.pop()
        destination.mkdir(parents=True, exist_ok=True)
        bundle = destination / f"{root}.zip"
        with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as output:
            for member in members:
                path = PurePosixPath(member.name)
                if path.is_absolute() or ".." in path.parts or len(path.parts) < 2:
                    raise ValueError(f"Unsafe archive path: {member.name}")
                stream = source.extractfile(member)
                if stream is None:
                    raise ValueError(f"Cannot read archive member: {member.name}")
                with stream:
                    info = zipfile.ZipInfo(member.name)
                    info.create_system = 3
                    info.external_attr = (member.mode | 0o100000) << 16
                    info.compress_type = zipfile.ZIP_DEFLATED
                    output.writestr(info, stream.read())
    return bundle


def verify_bundle(bundle: Path) -> None:
    """Check contents and run lightweight entry points outside the checkout."""
    with zipfile.ZipFile(bundle) as archive:
        names = archive.namelist()
        paths = [PurePosixPath(name) for name in names]
        if not paths or any(path.is_absolute() or ".." in path.parts for path in paths):
            raise ValueError("Empty bundle or unsafe archive paths")
        roots = {path.parts[0] for path in paths}
        if len(roots) != 1:
            raise ValueError("Release bundle must have exactly one root directory")
        root_name = roots.pop()
        relative = {str(path.relative_to(root_name)) for path in paths}
        missing = REQUIRED_FILES - relative
        if missing:
            raise ValueError(f"Release bundle is missing: {', '.join(sorted(missing))}")
        with tempfile.TemporaryDirectory(prefix="deepdream-bundle-") as temporary:
            archive.extractall(temporary)
            root = Path(temporary) / root_name
            for path in root.rglob("*.py"):
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for module in ("app.bootstrap", "app.check_deps"):
                subprocess.run(
                    [sys.executable, "-m", module, "--help"],
                    cwd=root,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            subprocess.run(
                [sys.executable, "-c", "import app; print(app.__version__)"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dist", type=Path, help="directory containing one built .tar.gz sdist")
    args = parser.parse_args()
    sources = list(args.dist.glob("*.tar.gz"))
    if len(sources) != 1:
        parser.error("Expected exactly one source distribution; use a clean output directory")
    bundle = build_bundle(sources[0], args.dist)
    verify_bundle(bundle)
    print(f"Verified launcher bundle: {bundle}")


if __name__ == "__main__":
    main()
