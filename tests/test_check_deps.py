import sys
import unittest
import unittest.mock as mock
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app.check_deps as check_deps


class MissingTests(unittest.TestCase):
    def test_present_and_absent(self):
        absent = check_deps.missing(["os", "definitely_not_a_real_module_xyz"])
        self.assertEqual(absent, ["definitely_not_a_real_module_xyz"])

    def test_empty_when_all_present(self):
        self.assertEqual(check_deps.missing(["os", "sys"]), [])


class PickRequirementsTests(unittest.TestCase):
    def test_cuda_when_nvidia_gpu(self):
        with mock.patch.object(check_deps, "has_nvidia_gpu", return_value=True):
            self.assertEqual(check_deps.pick_requirements(), "requirements-cuda.txt")

    def test_cpu_otherwise(self):
        with mock.patch.object(check_deps, "has_nvidia_gpu", return_value=False):
            self.assertEqual(check_deps.pick_requirements(), "requirements.txt")


class TkinterHintTests(unittest.TestCase):
    def test_macos_formula_matches_python_version(self):
        with (
            mock.patch.object(check_deps.sys, "platform", "darwin"),
            mock.patch.object(check_deps.sys, "version_info", (3, 12, 0, "final", 0)),
        ):
            hint = "\n".join(check_deps.tkinter_hint())
        self.assertIn("python-tk@3.12", hint)


class InVenvTests(unittest.TestCase):
    def test_in_venv_when_prefixes_differ(self):
        fake = SimpleNamespace(prefix="/venv", base_prefix="/usr")
        with mock.patch.object(check_deps, "sys", fake):
            self.assertTrue(check_deps.in_venv())

    def test_not_in_venv_when_prefixes_equal(self):
        fake = SimpleNamespace(prefix="/usr", base_prefix="/usr")
        with mock.patch.object(check_deps, "sys", fake):
            self.assertFalse(check_deps.in_venv())


class DeepCheckTests(unittest.TestCase):
    def test_timeout_returns_false(self):
        marker = mock.MagicMock()
        marker.exists.return_value = False
        with (
            mock.patch.object(check_deps, "MARKER", marker),
            mock.patch.object(check_deps, "pick_requirements", return_value="requirements.txt"),
            mock.patch.object(
                check_deps.subprocess,
                "run",
                side_effect=check_deps.subprocess.TimeoutExpired("cmd", 300),
            ),
        ):
            self.assertFalse(check_deps.deep_check_ok(["torch"], quiet=True))
        marker.unlink.assert_called_once_with(missing_ok=True)


class MainInstallTests(unittest.TestCase):
    def _main(self):
        with mock.patch.object(sys, "argv", ["check_deps.py"]):
            return check_deps.main()

    def test_deep_check_runs_after_first_launch_install(self):
        with (
            mock.patch.object(check_deps, "missing", return_value=["torch"]),
            mock.patch.object(check_deps, "install_with_fallback", return_value=0) as install,
            mock.patch.object(check_deps, "deep_check_ok", return_value=True) as deep,
        ):
            code = self._main()
        self.assertEqual(code, 0)
        install.assert_called_once()
        deep.assert_called_once()

    def test_fails_when_deep_check_fails_after_install(self):
        with (
            mock.patch.object(check_deps, "missing", return_value=["torch"]),
            mock.patch.object(check_deps, "install_with_fallback", return_value=0),
            mock.patch.object(check_deps, "deep_check_ok", return_value=False),
        ):
            self.assertEqual(self._main(), 1)


class CudaFallbackTests(unittest.TestCase):
    """A failed CUDA install must not dead-end a first-time user."""

    def _run(self, preferred, results):
        calls = []

        def fake_install(requirements, required, **kwargs):
            calls.append((requirements, kwargs))
            return results.pop(0)

        with (
            mock.patch.object(check_deps, "pick_requirements", return_value=preferred),
            mock.patch.object(check_deps, "confirm_install", return_value=True),
            mock.patch.object(check_deps, "install", side_effect=fake_install),
        ):
            code = check_deps.install_with_fallback(True, check_deps.REQUIRED)
        return code, calls

    def test_falls_back_to_cpu_when_cuda_install_fails(self):
        code, calls = self._run("requirements-cuda.txt", [1, 0])
        self.assertEqual(code, 0)
        self.assertEqual(
            calls,
            [
                ("requirements-cuda.txt", {}),
                ("requirements.txt", {"gpu_detected": False}),
            ],
        )

    def test_no_second_attempt_when_cpu_install_fails(self):
        code, calls = self._run("requirements-cuda.txt", [1, 1])
        self.assertEqual(code, 1)
        self.assertEqual(
            calls,
            [
                ("requirements-cuda.txt", {}),
                ("requirements.txt", {"gpu_detected": False}),
            ],
        )

    def test_cuda_success_does_not_touch_cpu_requirements(self):
        code, calls = self._run("requirements-cuda.txt", [0])
        self.assertEqual(code, 0)
        self.assertEqual(calls, [("requirements-cuda.txt", {})])

    def test_cpu_machine_does_not_retry_itself(self):
        code, calls = self._run("requirements.txt", [1])
        self.assertEqual(code, 1)
        self.assertEqual(calls, [("requirements.txt", {})])

    def test_declined_install_is_not_retried_on_cpu(self):
        with (
            mock.patch.object(
                check_deps, "pick_requirements", return_value="requirements-cuda.txt"
            ),
            mock.patch.object(check_deps, "confirm_install", return_value=False),
            mock.patch.object(check_deps, "install") as install,
        ):
            code = check_deps.install_with_fallback(False, check_deps.REQUIRED)
        self.assertEqual(code, 1)
        install.assert_not_called()


class MarkerResilienceTests(unittest.TestCase):
    """A read-only install directory must not crash first-launch setup."""

    def test_write_marker_ignores_unwritable_location(self):
        with mock.patch.object(check_deps, "MARKER", Path("/definitely/not/writable/.deps_ok")):
            check_deps.write_marker("abc\n")  # must not raise

    def test_clear_marker_ignores_unwritable_location(self):
        with mock.patch.object(check_deps, "MARKER", Path("/definitely/not/writable/.deps_ok")):
            check_deps.clear_marker()  # must not raise


class DeepCheckProbeTests(unittest.TestCase):
    def test_gui_probe_imports_app_gui(self):
        # pythonw.exe has no console: an app.gui import error must be caught by
        # the dependency check, not surface as a silent, windowless exit.
        marker = mock.MagicMock()
        marker.exists.return_value = False
        with (
            mock.patch.object(check_deps, "MARKER", marker),
            mock.patch.object(check_deps, "pick_requirements", return_value="requirements.txt"),
            mock.patch.object(check_deps.subprocess, "run") as run,
        ):
            run.return_value = SimpleNamespace(returncode=0, stdout="", stderr="")
            self.assertTrue(check_deps.deep_check_ok(["torch", "tkinter"], quiet=True))
        probe = run.call_args[0][0][2]
        self.assertIn("import app.gui", probe)

    def test_cli_check_still_probes_torch(self):
        marker = mock.MagicMock()
        marker.exists.return_value = False
        with (
            mock.patch.object(check_deps, "MARKER", marker),
            mock.patch.object(check_deps, "pick_requirements", return_value="requirements.txt"),
            mock.patch.object(check_deps.subprocess, "run") as run,
        ):
            run.return_value = SimpleNamespace(returncode=0, stdout="", stderr="")
            self.assertTrue(check_deps.deep_check_ok(check_deps.REQUIRED, quiet=True))
        self.assertIn("import torch", run.call_args[0][0][2])
        self.assertNotIn("import app.gui", run.call_args[0][0][2])

    def test_multiline_marker_recognized(self):
        fp_cli = check_deps.fingerprint("requirements.txt", check_deps.REQUIRED)
        fp_gui = check_deps.fingerprint("requirements.txt", check_deps.REQUIRED_GUI)
        marker = mock.MagicMock()
        marker.exists.return_value = True
        marker.read_text.return_value = f"{fp_cli}\n{fp_gui}\n"
        with (
            mock.patch.object(check_deps, "MARKER", marker),
            mock.patch.object(check_deps, "pick_requirements", return_value="requirements.txt"),
        ):
            self.assertTrue(check_deps.deep_check_ok(check_deps.REQUIRED, quiet=True))
            self.assertTrue(check_deps.deep_check_ok(check_deps.REQUIRED_GUI, quiet=True))

    def test_gui_check_records_both_gui_and_cli_fingerprints(self):
        fp_cli = check_deps.fingerprint("requirements.txt", check_deps.REQUIRED)
        fp_gui = check_deps.fingerprint("requirements.txt", check_deps.REQUIRED_GUI)
        marker = mock.MagicMock()
        marker.exists.return_value = False
        with (
            mock.patch.object(check_deps, "MARKER", marker),
            mock.patch.object(check_deps, "pick_requirements", return_value="requirements.txt"),
            mock.patch.object(check_deps.subprocess, "run") as run,
        ):
            run.return_value = SimpleNamespace(returncode=0, stdout="", stderr="")
            self.assertTrue(check_deps.deep_check_ok(check_deps.REQUIRED_GUI, quiet=True))
        marker.write_text.assert_called_once()
        written = marker.write_text.call_args[0][0]
        self.assertIn(fp_gui, written)
        self.assertIn(fp_cli, written)


class FingerprintTests(unittest.TestCase):
    def test_changes_with_requirements_file(self):
        a = check_deps.fingerprint("requirements.txt", check_deps.REQUIRED)
        with mock.patch.object(Path, "read_bytes", return_value=b"different"):
            b = check_deps.fingerprint("requirements.txt", check_deps.REQUIRED)
        self.assertNotEqual(a, b)

    def test_changes_with_required_modules(self):
        a = check_deps.fingerprint("requirements.txt", check_deps.REQUIRED)
        b = check_deps.fingerprint("requirements.txt", check_deps.REQUIRED_GUI)
        self.assertNotEqual(a, b)


if __name__ == "__main__":
    unittest.main()
