"""First-run launcher checks.

The .bat/.sh scripts are the entire new-user experience, so they get behavioural
tests rather than substring greps: the shell launchers are actually executed and
the Windows ones are parsed for the interpreter probe they will run.
"""

import os
import re
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SH_LAUNCHERS = ["Launch.sh", "CLI/LaunchCLI.sh"]
BAT_LAUNCHERS = ["Launch.bat", "CLI/LaunchCLI.bat"]


def _read(name: str) -> str:
    return (REPO_ROOT / name).read_text(encoding="utf-8")


def _working_bash() -> str | None:
    """Return a bash that actually runs, or None.

    On Windows a bare "bash" on PATH is usually the WSL launcher stub, which
    prints an install prompt and exits non-zero. Trying it first would turn a
    clean machine into a false failure, so probe before use.
    """
    candidates = [shutil.which("bash")]
    if os.name == "nt":
        candidates += [
            r"C:\Program Files\Git\bin\bash.exe",
            r"C:\Program Files\Git\usr\bin\bash.exe",
        ]
    for cand in candidates:
        if not cand or not Path(cand).exists():
            continue
        try:
            probe = subprocess.run([cand, "-c", "exit 0"], capture_output=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            continue
        if probe.returncode == 0:
            return cand
    return None


class LauncherSyntaxTests(unittest.TestCase):
    """Shell launchers must at least parse, on the platform that will run them."""

    def test_shell_launchers_parse(self):
        bash = _working_bash()
        if bash is None:
            self.skipTest("no working bash on this machine")
        for name in SH_LAUNCHERS:
            with self.subTest(launcher=name):
                proc = subprocess.run(
                    [bash, "-n", str(REPO_ROOT / name)],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(proc.returncode, 0, proc.stderr)

    @pytest.mark.skipif(os.name != "nt", reason="Windows launchers only run on Windows")
    def test_bat_launchers_have_balanced_quotes(self):
        # An unbalanced quote in a .bat silently swallows the rest of the file,
        # which for a first-time user means nothing appears to happen at all.
        for name in BAT_LAUNCHERS:
            with self.subTest(launcher=name):
                self.assertEqual(_read(name).count('"') % 2, 0)


class InterpreterProbeTests(unittest.TestCase):
    """The launchers must probe the Python versions the project supports."""

    def _expected_bounds(self):
        import app.bootstrap as bootstrap

        return bootstrap.MIN_PY, bootstrap.MAX_PY

    def test_launcher_bounds_match_bootstrap(self):
        """A launcher that probes a different range than bootstrap.py hard-blocks
        (or accepts) an interpreter the other one then rejects."""
        min_py, max_py = self._expected_bounds()
        lo, hi = min_py[1], max_py[1]
        for name in SH_LAUNCHERS + BAT_LAUNCHERS:
            with self.subTest(launcher=name):
                text = _read(name)
                match = re.search(
                    r"\(3,\s*(\d+)\)\s*<=\s*sys\.version_info\[:2\]\s*<=\s*\(3,\s*(\d+)\)", text
                )
                self.assertIsNotNone(match, f"{name} has no version probe")
                assert match is not None  # narrowed for mypy
                self.assertEqual(int(match.group(1)), lo)
                self.assertEqual(int(match.group(2)), hi)

    def test_shell_launchers_exclude_python_314_on_macos(self):
        for name in SH_LAUNCHERS:
            with self.subTest(launcher=name):
                text = _read(name)
                self.assertIn("sys.platform != 'darwin'", text)
                self.assertIn("sys.version_info[:2] <= (3, 13)", text)

    def test_shell_launchers_try_the_newest_supported_version(self):
        _, max_py = self._expected_bounds()
        newest = f"python3.{max_py[1]}"
        for name in SH_LAUNCHERS:
            with self.subTest(launcher=name):
                self.assertIn(newest, _read(name))

    def test_bat_launchers_try_the_newest_supported_version(self):
        _, max_py = self._expected_bounds()
        newest = f"py -3.{max_py[1]}"
        for name in BAT_LAUNCHERS:
            with self.subTest(launcher=name):
                self.assertIn(newest, _read(name))

    def test_launchers_scan_newest_first(self):
        """An unversioned "python" must not shadow a supported versioned one."""
        for name in SH_LAUNCHERS:
            with self.subTest(launcher=name):
                line = next(ln for ln in _read(name).splitlines() if ln.startswith("for candidate"))
                names = line.split("in", 1)[1].split(";")[0].split()
                # The unversioned fallbacks must come last, and the newest
                # supported release must be probed before the older ones.
                self.assertEqual(names[-2:], ["python3", "python"])
                versioned = [n for n in names if n not in ("python", "python3")]
                self.assertEqual(versioned, sorted(versioned, reverse=True))
                self.assertLess(names.index("python3.11"), names.index("python3"))


class GuidanceTests(unittest.TestCase):
    """Install guidance must name a version the project actually supports."""

    def test_no_guidance_points_at_an_unsupported_python(self):
        import app.bootstrap as bootstrap

        unsupported = f"3.{bootstrap.MAX_PY[1] + 1}"
        for name in SH_LAUNCHERS + BAT_LAUNCHERS:
            with self.subTest(launcher=name):
                text = _read(name)
                for found in re.findall(r"python@3\.(\d+)", text):
                    self.assertLessEqual(
                        int(found),
                        bootstrap.MAX_PY[1],
                        f"{name} suggests python@3.{found}, which is unsupported",
                    )
                self.assertNotIn(f"python@3.{unsupported}", text)

    def test_gui_launcher_mentions_tkinter_where_tkinter_is_separate(self):
        """Homebrew's Python ships Tkinter separately, so the GUI launcher must
        name the python-tk formula or macOS users hit an unfixable dead end."""
        import app.bootstrap as bootstrap

        with mock.patch.object(bootstrap.sys, "platform", "darwin"):
            guidance = "\n".join(bootstrap.install_guidance(gui=True))
        self.assertIn("python-tk@", guidance)
        self.assertIn("python-tk@", _read("Launch.sh"))
        self.assertIn("python-tk@3.13", guidance)

    def test_python_not_found_message_names_tk_for_every_distro(self):
        text = _read("Launch.sh")
        for expected in ("python3-tk", "python3-tkinter", "pacman", "python-tk@"):
            with self.subTest(expected=expected):
                self.assertIn(expected, text)


class WritableFolderGuidanceTests(unittest.TestCase):
    """First launch needs a writable app folder for its local virtualenv."""

    def test_all_launchers_explain_read_only_folder_fix(self):
        for name in SH_LAUNCHERS + BAT_LAUNCHERS:
            with self.subTest(launcher=name):
                text = _read(name).lower()
                self.assertIn("not writable", text)
                self.assertIn("documents", text)
                self.assertIn(".venv", text)


class RepoRootRedirectionTests(unittest.TestCase):
    """CLI launchers live in CLI/ but must operate on the repo root."""

    def test_root_launchers_do_not_use_repo_dir(self):
        for name in ("Launch.sh", "Launch.bat"):
            with self.subTest(launcher=name):
                self.assertNotIn("REPO_DIR", _read(name))

    def test_cli_shell_launcher_uses_repo_dir(self):
        text = _read("CLI/LaunchCLI.sh")
        for expected in (
            'REPO_DIR="$(dirname "$SCRIPT_DIR")"',
            'cd "$REPO_DIR"',
            '"$REPO_DIR/app/bootstrap.py"',
            '"$REPO_DIR/.venv/bin/python"',
            '"$REPO_DIR/.deepdream-write-test"',
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, text)
        for forbidden in (
            'cd "$SCRIPT_DIR"',
            '"$SCRIPT_DIR/app/bootstrap',
            '"$SCRIPT_DIR/.venv',
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, text)

    def test_cli_bat_launcher_uses_repo_dir(self):
        text = _read("CLI/LaunchCLI.bat")
        for expected in (
            'cd /d "%REPO_DIR%"',
            '"%REPO_DIR%.deepdream-write-test"',
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, text)
        # %~dp0 may only appear when defining REPO_DIR itself.
        without_repo_dir_def = "\n".join(ln for ln in text.splitlines() if "REPO_DIR" not in ln)
        self.assertNotIn("%~dp0", without_repo_dir_def)
        self.assertNotIn("%SCRIPT_DIR%", without_repo_dir_def)

    def test_cli_bat_launcher_resolves_repo_dir_to_a_real_path(self):
        """The .bat must normalize "...\\CLI\\.." rather than echoing it back.

        REPO_DIR is shown to the user in the not-writable error, so a raw
        "%~dp0..\\" would tell someone to move the folder to a path containing
        a "..", which is not a path they can act on. The .sh gets this for free
        from dirname, so the two launchers must agree on the resolved value.
        """
        text = _read("CLI/LaunchCLI.bat")
        self.assertIn('for %%I in ("%~dp0..") do set "REPO_DIR=%%~fI\\"', text)


class BootstrapSmokeTests(unittest.TestCase):
    """bootstrap.py must run standalone under the system interpreter."""

    def test_bootstrap_is_stdlib_only_and_importable(self):
        # It is invoked by path with whatever interpreter the launcher found, so
        # importing the app package must never be a requirement.
        source = (REPO_ROOT / "app" / "bootstrap.py").read_text(encoding="utf-8")
        self.assertNotIn("import torch", source)
        self.assertNotIn("from app", source)
        self.assertNotIn("from .", source)

    def test_check_deps_is_stdlib_only(self):
        source = (REPO_ROOT / "app" / "check_deps.py").read_text(encoding="utf-8")
        for forbidden in ("import torch", "import numpy", "from app", "from ."):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_bootstrap_help_runs(self):
        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "app" / "bootstrap.py"), "--help"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("--gui", proc.stdout)


if __name__ == "__main__":
    unittest.main()
