import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app.bootstrap as bootstrap


class PythonVersionTests(unittest.TestCase):
    def test_current_interpreter(self):
        version = bootstrap.python_version(Path(sys.executable))
        assert version is not None
        self.assertGreaterEqual(version, (3, 8))

    def test_missing_interpreter(self):
        self.assertIsNone(bootstrap.python_version(Path("no_such_python_binary_xyz")))


class VenvPythonTests(unittest.TestCase):
    def test_platform_layout(self):
        result = bootstrap.venv_python(Path("v"))
        if bootstrap.os.name == "nt":
            self.assertEqual(result, Path("v/Scripts/python.exe"))
        else:
            self.assertEqual(result, Path("v/bin/python"))

    def test_defaults_to_module_venv(self):
        self.assertEqual(bootstrap.venv_python(), bootstrap.venv_python(bootstrap.VENV))


class GuidanceTests(unittest.TestCase):
    def test_guidance_is_nonempty(self):
        lines = bootstrap.install_guidance()
        self.assertTrue(lines)
        self.assertTrue(any("python" in line.lower() for line in lines))

    def test_gui_guidance_includes_tk_packages_on_linux(self):
        with (
            mock.patch.object(bootstrap.sys, "platform", "linux"),
            mock.patch.object(bootstrap, "linux_distro", return_value="ubuntu"),
        ):
            gui_text = "\n".join(bootstrap.install_guidance(gui=True))
            cli_text = "\n".join(bootstrap.install_guidance(gui=False))
        self.assertIn("python3-tk", gui_text)
        self.assertNotIn("python3-tk", cli_text)

    def test_macos_guidance_recommends_supported_python(self):
        with mock.patch.object(bootstrap.sys, "platform", "darwin"):
            text = "\n".join(bootstrap.install_guidance(gui=True))
        self.assertIn("python@3.13", text)
        self.assertIn("python-tk@3.13", text)
        self.assertNotIn("3.14", text)

    def test_linux_distro_returns_string(self):
        self.assertIsInstance(bootstrap.linux_distro(), str)


class VenvIssueTests(unittest.TestCase):
    def test_missing_venv_reports_issue(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(bootstrap, "VENV", Path(tmp) / ".venv"):
                self.assertEqual(bootstrap.venv_issue(), "missing interpreter")

    def test_too_new_venv_python_reports_issue(self):
        with tempfile.TemporaryDirectory() as tmp:
            venv = Path(tmp) / ".venv"
            (venv / "Scripts" if bootstrap.os.name == "nt" else venv / "bin").mkdir(parents=True)
            py = bootstrap.venv_python(venv)
            py.touch()
            with (
                mock.patch.object(bootstrap, "VENV", venv),
                mock.patch.object(bootstrap, "python_version", return_value=(3, 15)),
            ):
                issue = bootstrap.venv_issue()
        self.assertIsNotNone(issue)
        assert issue is not None  # narrowed for mypy
        self.assertIn("3.15", issue)

    def test_supported_maximum_version_venv_is_accepted(self):
        """The newest supported release must not be rejected as "too new"."""
        maximum = bootstrap.max_supported_python()
        with tempfile.TemporaryDirectory() as tmp:
            venv = Path(tmp) / ".venv"
            (venv / "Scripts" if bootstrap.os.name == "nt" else venv / "bin").mkdir(parents=True)
            bootstrap.venv_python(venv).touch()
            with (
                mock.patch.object(bootstrap, "VENV", venv),
                mock.patch.object(bootstrap, "python_version", return_value=maximum),
            ):
                issue = bootstrap.venv_issue()
        self.assertIsNone(issue)

    def test_python_314_venv_is_rejected_on_macos(self):
        with tempfile.TemporaryDirectory() as tmp:
            venv = Path(tmp) / ".venv"
            python = bootstrap.venv_python(venv)
            python.parent.mkdir(parents=True)
            python.touch()
            with (
                mock.patch.object(bootstrap, "VENV", venv),
                mock.patch.object(bootstrap, "python_version", return_value=(3, 14)),
                mock.patch.object(bootstrap.sys, "platform", "darwin"),
            ):
                issue = bootstrap.venv_issue()
        self.assertIn("supported: 3.11-3.13", issue or "")


class BackupTests(unittest.TestCase):
    def test_backup_renames_without_deleting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            venv = root / ".venv"
            venv.mkdir()
            (venv / "marker.txt").write_text("keep me")
            with (
                mock.patch.object(bootstrap, "ROOT", root),
                mock.patch.object(bootstrap, "VENV", venv),
            ):
                target = bootstrap._backup_venv()
            self.assertFalse(venv.exists())
            self.assertTrue(target.exists())
            self.assertEqual((target / "marker.txt").read_text(), "keep me")
            self.assertTrue(target.name.startswith(".venv.bad-"))


class MainTests(unittest.TestCase):
    def test_valid_venv_short_circuits_without_version_gate(self):
        with (
            mock.patch.object(bootstrap, "venv_issue", return_value=None),
            mock.patch.object(bootstrap, "run_check_deps", return_value=0) as run,
        ):
            with mock.patch.object(sys, "argv", ["bootstrap.py", "--gui"]):
                code = bootstrap.main()
        self.assertEqual(code, 0)
        run.assert_called_once_with(bootstrap.venv_python(), True)

    def test_old_interpreter_without_venv_gives_guidance(self):
        fake = (3, 9, 0, "final", 0)
        buf = io.StringIO()
        with mock.patch.object(bootstrap, "venv_issue", return_value="missing interpreter"):
            with mock.patch.object(bootstrap.sys, "version_info", fake):
                with mock.patch.object(sys, "argv", ["bootstrap.py"]):
                    with contextlib.redirect_stdout(buf):
                        code = bootstrap.main()
        self.assertEqual(code, 1)
        self.assertIn("Python 3.9", buf.getvalue())

    def test_too_new_interpreter_without_venv_gives_guidance(self):
        fake = (3, 15, 0, "final", 0)
        buf = io.StringIO()
        with mock.patch.object(bootstrap, "venv_issue", return_value="missing interpreter"):
            with mock.patch.object(bootstrap.sys, "version_info", fake):
                with mock.patch.object(sys, "argv", ["bootstrap.py"]):
                    with contextlib.redirect_stdout(buf):
                        code = bootstrap.main()
        self.assertEqual(code, 1)
        self.assertIn("Python 3.15", buf.getvalue())

    def test_supported_interpreter_proceeds_to_venv_setup(self):
        fake = (3, 13, 0, "final", 0)
        with (
            mock.patch.object(bootstrap, "venv_issue", return_value="missing interpreter"),
            mock.patch.object(bootstrap.sys, "version_info", fake),
            mock.patch.object(bootstrap, "ensure_venv", return_value=Path("py")) as ensure,
            mock.patch.object(bootstrap, "run_check_deps", return_value=0),
        ):
            with mock.patch.object(sys, "argv", ["bootstrap.py"]):
                code = bootstrap.main()
        self.assertEqual(code, 0)
        ensure.assert_called_once()

    def test_newest_supported_interpreter_is_not_rejected(self):
        """A user whose only Python is the newest supported one must not be told
        to downgrade: the launcher has to accept it and set up the venv."""
        maximum = bootstrap.max_supported_python()
        fake = (maximum[0], maximum[1], 0, "final", 0)
        with (
            mock.patch.object(bootstrap, "venv_issue", return_value="missing interpreter"),
            mock.patch.object(bootstrap.sys, "version_info", fake),
            mock.patch.object(bootstrap, "ensure_venv", return_value=Path("py")) as ensure,
            mock.patch.object(bootstrap, "run_check_deps", return_value=0),
        ):
            with mock.patch.object(sys, "argv", ["bootstrap.py"]):
                code = bootstrap.main()
        self.assertEqual(code, 0)
        ensure.assert_called_once()

    def test_python_314_without_venv_is_rejected_on_macos(self):
        fake = (3, 14, 0, "final", 0)
        buf = io.StringIO()
        with (
            mock.patch.object(bootstrap, "venv_issue", return_value="missing interpreter"),
            mock.patch.object(bootstrap.sys, "version_info", fake),
            mock.patch.object(bootstrap.sys, "platform", "darwin"),
            mock.patch.object(bootstrap, "ensure_venv") as ensure,
        ):
            with mock.patch.object(sys, "argv", ["bootstrap.py"]):
                with contextlib.redirect_stdout(buf):
                    code = bootstrap.main()
        self.assertEqual(code, 1)
        self.assertIn("Python 3.14", buf.getvalue())
        self.assertIn("python@3.13", buf.getvalue())
        ensure.assert_not_called()


class TkinterGuidanceTests(unittest.TestCase):
    def test_guidance_names_tkinter_and_a_fix(self):
        lines = bootstrap.tkinter_guidance()
        self.assertTrue(lines)
        text = "\n".join(lines)
        self.assertIn("kinter", text)
        self.assertTrue(
            any(
                "apt" in line
                or "dnf" in line
                or "pacman" in line
                or "python.org" in line.lower()
                or "reinstall" in line.lower()
                for line in lines
            )
        )

    def test_macos_tk_formula_matches_selected_python(self):
        with (
            mock.patch.object(bootstrap.sys, "platform", "darwin"),
            mock.patch.object(bootstrap, "python_version", return_value=(3, 12)),
        ):
            text = "\n".join(bootstrap.tkinter_guidance(Path("python")))
        self.assertIn("python-tk@3.12", text)


if __name__ == "__main__":
    unittest.main()
