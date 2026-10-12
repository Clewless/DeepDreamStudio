import gc
import os
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest

import app.gui as gui
from app.gui_helpers import (
    MEDIA_EXTENSIONS,
    PROC_DONE_PREFIX,
    ZOOM_MAX,
    ZOOM_MIN,
    FrameCache,
    VideoFrameLoader,
    _dialog_pattern_for_ext,
    build_reveal_command,
    clamp_zoom,
    compute_zoom_view,
    count_saved_video_frames,
    create_split_image,
    extract_output_path,
    fit_zoom_for,
    format_frame_label,
    list_saved_video_frames,
    newest_media,
    newest_png,
    render_zoomed_preview,
    resolve_out_dir,
    slider_t_to_val,
    slider_val_to_t,
    terminate_process,
    timeline_index_to_t,
    timeline_t_to_index,
)
from app.gui_job import GenerationJob, ProgressTracker
from app.gui_widgets import ModernButton, Tooltip
from app.model_data import (
    MAX_NAME_LEN,
    build_systematic_video_name,
    read_video_frames_meta,
    resolve_name_conflict,
    write_video_frames_meta,
)
from app.protocol import (
    EV_DONE,
    EV_FRAME,
    EV_ITERATION,
    EV_OCTAVE,
    EV_PREVIEW,
    EVENT_PREFIX,
    format_event,
    parse_event,
)


class TestVideoFrameLoader(unittest.TestCase):
    """Exercises the real worker thread, not a mock of it.

    No Tk needed: this is the thread that keeps mp4 decoding off the UI thread.
    """

    _video_available: bool = False

    @classmethod
    def setUpClass(cls):
        try:
            import imageio  # noqa: F401

            cls._video_available = True
        except ImportError:
            cls._video_available = False

    def setUp(self):
        if not self._video_available:
            self.skipTest("imageio/imageio-ffmpeg not installed")
        self.results: list[Any] = []
        self.loader = VideoFrameLoader(self.results.append, max_pending=1)
        # Own the temp dir so tearDown can stop the worker before Windows
        # refuses to delete a file the decoder still holds open.
        self.temp_dir = tempfile.mkdtemp(prefix="dd_loader_")

    def tearDown(self):
        self.loader.wait_idle(10)
        self.loader.close()
        self.results.clear()
        gc.collect()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _make_video(self, td=None, frames=12, size=(64, 48)):
        import imageio.v2 as imageio_v2

        td = td or self.temp_dir
        path = os.path.join(td, f"clip_{frames}_{size[0]}.mp4")
        with imageio_v2.get_writer(path, fps=10, codec="libx264") as writer:
            for i in range(frames):
                writer.append_data(  # type: ignore[attr-defined]
                    np.full((size[1], size[0], 3), (i * 10) % 256, dtype="uint8")
                )
        return path

    def test_frame_request_decodes_off_thread(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._make_video(td)
            t0 = time.perf_counter()
            self.loader.request_frame("mp4", path, 3, token=1)
            submit_cost = time.perf_counter() - t0
            self.assertLess(submit_cost, 0.05, "request_frame must not decode inline")
            self.loader.wait_idle(30)
            self.loader.poll()
            frames = [e for e in self.results if e[0] == "frame"]
            self.assertEqual(len(frames), 1)
            self.assertEqual(frames[0][3], 3)
            self.assertIsNotNone(frames[0][4])
            self.assertEqual(frames[0][4].size, (64, 48))
            self.assertEqual(frames[0][5], 1)

    def test_frame_and_probe_submitted_together_both_survive(self):
        # _setup_mp4_preview issues both back to back; the frame request must
        # not be trimmed away by the probe (or vice versa).
        with tempfile.TemporaryDirectory() as td:
            path = self._make_video(td)
            self.loader.request_frame("mp4", path, 0, token=1)
            self.loader.request_probe(path, token=1)
            self.loader.wait_idle(30)
            self.loader.poll()
            kinds = [e[0] for e in self.results]
            self.assertIn("frame", kinds)
            self.assertIn("probe", kinds)

    def test_scrubbing_drops_the_backlog_and_keeps_the_newest(self):
        path = self._make_video(frames=40)
        for i in range(20):
            self.loader.request_frame("mp4", path, i, token=1)
        self.loader.wait_idle(30)
        self.loader.poll()
        indices = [e[3] for e in self.results if e[0] == "frame"]
        self.assertTrue(indices)
        self.assertEqual(indices[-1], 19, "the newest scrub position must be decoded")
        self.assertLess(len(indices), 20, "intermediate frames must be dropped, not queued")

    def test_worker_survives_a_decode_failure(self):
        with tempfile.TemporaryDirectory() as td:
            missing = os.path.join(td, "gone.mp4")
            self.loader.request_frame("mp4", missing, 0, token=1)
            self.loader.wait_idle(10)
            self.loader.poll()
            # A bad path must not take the worker down with it.
            real = self._make_video(td)
            self.loader.request_frame("mp4", real, 1, token=2)
            self.loader.wait_idle(30)
            self.loader.poll()
            ok = [e for e in self.results if e[0] == "frame" and e[3] == 1]
            self.assertTrue(ok)

    def test_invalidate_discards_pending_and_marks_generation(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._make_video(td)
            before = self.loader.generation()
            self.loader.request_frame("mp4", path, 0, token=1)
            self.loader.invalidate()
            self.assertGreater(self.loader.generation(), before)
            self.assertTrue(self.loader._requests.empty())

    def test_close_joins_the_worker(self):
        loader = VideoFrameLoader(lambda r: None)
        loader.request_probe(os.devnull, 1)
        loader.close()
        self.assertFalse(loader._thread.is_alive())

    def test_frame_cache_evicts_oldest(self):
        cache = FrameCache(max_entries=3)
        for i in range(5):
            cache.put(("mp4", "v", i), f"frame{i}")
        self.assertEqual(len(cache), 3)
        self.assertIsNone(cache.get(("mp4", "v", 0)))
        self.assertEqual(cache.get(("mp4", "v", 4)), "frame4")
        cache.clear()
        self.assertEqual(len(cache), 0)


class TestStructuredEvents(unittest.TestCase):
    def test_round_trip(self):
        line = format_event(EV_FRAME, current=3, total=10)
        event = parse_event(line)
        assert event is not None
        self.assertEqual(event["event"], EV_FRAME)
        self.assertEqual(event["current"], 3)
        self.assertEqual(event["total"], 10)

    def test_ordinary_output_is_not_an_event(self):
        self.assertIsNone(parse_event("Octave 1/2  image 512x512"))
        self.assertIsNone(parse_event(""))
        self.assertIsNone(parse_event(f"{EVENT_PREFIX}not json"))

    def test_tracker_prefers_events(self):
        t = ProgressTracker()
        frac, status = t.update(format_event(EV_OCTAVE, current=2, total=4))
        assert frac is not None
        self.assertAlmostEqual(frac, 0.25)
        self.assertEqual(status, "Octave 2/4")
        frac, status = t.update(format_event(EV_ITERATION, current=1, total=2))
        assert frac is not None
        self.assertAlmostEqual(frac, 0.375)

    def test_tracker_frame_without_total(self):
        t = ProgressTracker()
        frac, status = t.update(format_event(EV_FRAME, current=42))
        self.assertIsNone(frac)
        self.assertEqual(status, "Frame 42")

    def test_tracker_falls_back_to_text(self):
        t = ProgressTracker()
        frac, status = t.update("Octave 1/2  image 512x512")
        assert frac is not None
        self.assertAlmostEqual(frac, 0.0)
        self.assertEqual(status, "Octave 1/2")

    def test_event_line_is_hidden_from_log(self):
        # An event must never be echoed as text, or the console fills with noise.
        line = format_event(EV_DONE, path="out.png")
        self.assertTrue(line.startswith(EVENT_PREFIX))


class TestProgressTracker(unittest.TestCase):
    def test_octave_start_sets_progress_and_status(self):
        t = ProgressTracker()
        frac, status = t.update("Octave 1/4  image 512x512")
        self.assertEqual(frac, 0.0)
        self.assertEqual(status, "Octave 1/4")

    def test_iteration_advances_within_first_octave(self):
        t = ProgressTracker()
        t.update("Octave 1/4  image 512x512")
        frac, status = t.update("  iteration 5/10   loss      3.2")
        self.assertIsNone(status)
        assert frac is not None
        self.assertAlmostEqual(frac, 0.0 + (5 / 10) / 4)

    def test_iteration_after_later_octave_is_offset(self):
        t = ProgressTracker()
        t.update("Octave 2/4  image 256x256")
        frac, _ = t.update("  iteration 1/2   loss      1.0")
        assert frac is not None
        self.assertAlmostEqual(frac, 0.25 + (1 / 2) / 4)

    def test_iteration_before_any_octave_is_ignored(self):
        t = ProgressTracker()
        frac, status = t.update("  iteration 1/2   loss 1.0")
        self.assertIsNone(frac)
        self.assertIsNone(status)

    def test_unrelated_lines_are_ignored(self):
        t = ProgressTracker()
        for line in (
            "Device: cuda  (NVIDIA RTX)",
            "Loading GoogLeNet...",
            "Done in 12.3s -> out/x.png",
        ):
            self.assertEqual(t.update(line), (None, None))

    def test_malformed_lines_do_not_crash(self):
        t = ProgressTracker()
        self.assertEqual(t.update("Octave /4"), (None, None))
        self.assertEqual(t.update("Octave 1/x"), (None, None))
        self.assertEqual(t.update("Octave 1/0"), (None, None))
        t.update("Octave 1/2")
        self.assertEqual(t.update("  iteration x/y   loss 1.0"), (None, None))
        self.assertEqual(t.update("  iteration 1/0   loss 1.0"), (None, None))

    def test_frame_progress_tracking(self):
        t = ProgressTracker()
        frac, status = t.update("Frame 5/20  (loss = 2.45)")
        self.assertEqual(frac, 0.25)
        self.assertEqual(status, "Frame 5/20")

    def test_encoding_frame_progress_tracking(self):
        t = ProgressTracker()
        frac, status = t.update("Encoding frame 5/20 (25%)")
        self.assertEqual(frac, 0.25)
        self.assertEqual(status, "Encoding 5/20")
        frac, status = t.update("Encoding frame 20/20 (100%)")
        self.assertEqual(frac, 1.0)
        self.assertEqual(status, "Encoding 20/20")

    def test_remux_audio_progress_tracking(self):
        t = ProgressTracker()
        frac, status = t.update("Remuxing audio track...")
        self.assertEqual(frac, 0.98)
        self.assertEqual(status, "Remuxing audio...")

    def test_compiling_progress_tracking(self):
        t = ProgressTracker()
        frac, status = t.update("Compiling 50 frames into video: out.mp4 (512x512 @ 30 fps)...")
        self.assertEqual(frac, 0.0)
        self.assertEqual(status, "Compiling...")

    def test_encoding_malformed_lines(self):
        t = ProgressTracker()
        self.assertEqual(t.update("Encoding frame /20"), (None, None))
        self.assertEqual(t.update("Encoding frame 5/0"), (None, None))
        self.assertEqual(t.update("Encoding frame abc/def"), (None, None))


class TestExtractOutputPath(unittest.TestCase):
    def test_extracts_path_from_done_line(self):
        line = "Done in 12.3s -> C:\\out\\photo_inception4c.png"
        self.assertEqual(extract_output_path(line), "C:\\out\\photo_inception4c.png")

    def test_path_with_spaces(self):
        line = "Done in 1.0s -> E:\\My Dream Images\\photo inception5b.png"
        self.assertEqual(
            extract_output_path(line),
            "E:\\My Dream Images\\photo inception5b.png",
        )

    def test_non_done_line_returns_none(self):
        self.assertIsNone(extract_output_path("Octave 1/4  image 512x512"))
        self.assertIsNone(extract_output_path("  iteration 1/10   loss 5.0"))


class TestResolveOutDir(unittest.TestCase):
    def test_absolute_is_unchanged(self):
        samples = (
            [str(Path(__file__).resolve().parent), "C:\\out", "E:\\My Dreams\\out"]
            if os.name == "nt"
            else [str(Path(__file__).resolve().parent), "/tmp/out", "/var/log/dreams"]
        )
        for v in samples:
            self.assertEqual(resolve_out_dir(v), v)

    def test_relative_is_anchored_at_base(self):
        base = Path(__file__).resolve().parent
        self.assertEqual(resolve_out_dir("out", base=base), str(base / "out"))
        self.assertEqual(
            resolve_out_dir(os.path.join("sub", "dir"), base=base),
            str(base / "sub" / "dir"),
        )

    def test_dot_and_empty_default_to_script_out_dir(self):
        # An empty field must not reach deepdream as --out "" (it would fail
        # with "could not create output directory ''"); fall back to the script
        # out dir instead. Whitespace-only values behave the same.
        base = Path(__file__).resolve().parent
        self.assertEqual(resolve_out_dir("", base=base), str(base / "out"))
        self.assertEqual(resolve_out_dir("   ", base=base), str(base / "out"))

    def test_surrounding_whitespace_is_stripped(self):
        base = Path(__file__).resolve().parent
        self.assertEqual(resolve_out_dir("  out  ", base=base), str(base / "out"))
        self.assertEqual(
            resolve_out_dir(f" {os.path.join('sub', 'dir')} ", base=base),
            str(base / "sub" / "dir"),
        )

    @unittest.skipUnless(os.name == "nt", "Drive-relative paths are Windows-specific")
    def test_drive_relative_is_made_absolute(self):
        # "C:out" is drive-relative, not absolute: os.path.isabs() is False on
        # Windows and os.path.join(base, ...) leaves it unchanged. Every result
        # must be absolute so GUI preview/cleanup and deepdream agree.
        base = Path(__file__).resolve().parent
        for v in ("C:out", "D:", "E:sub\\dir"):
            result = resolve_out_dir(v, base=base)
            self.assertTrue(
                os.path.isabs(result),
                f"{v!r} resolved to non-absolute {result!r}",
            )

    @unittest.skipUnless(os.name == "nt", "Drive-relative paths are Windows-specific")
    def test_drive_relative_is_anchored_at_base(self):
        # A drive-relative path ("C:out", "E:sub\\dir") must be anchored at
        # the base instead of leaking through os.path.join and resolving
        # against the process cwd on that drive.
        base = Path(__file__).resolve().parent
        for v in ("C:out", "E:sub\\dir"):
            result = resolve_out_dir(v, base=base)
            self.assertTrue(os.path.isabs(result), f"{v!r} resolved to non-absolute {result!r}")
            self.assertEqual(
                result,
                os.path.abspath(str(base / v.split(":", 1)[1])),
                f"{v!r} was not anchored at the base",
            )
        # A bare drive ("D:") has nothing to anchor: fall back to the base.
        self.assertEqual(resolve_out_dir("D:", base=base), os.path.abspath(str(base)))


class TestNewestPng(unittest.TestCase):
    def test_returns_most_recent_png(self):
        with tempfile.TemporaryDirectory() as d:
            older = Path(d) / "older.png"
            newer = Path(d) / "newer.png"
            older.write_bytes(b"png")
            newer.write_bytes(b"png")
            os.utime(older, (1000, 1000))
            os.utime(newer, (2000, 2000))
            self.assertEqual(newest_png(d), str(newer))

    def test_ignores_non_png_and_missing_dir(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "note.txt").write_bytes(b"x")
            self.assertIsNone(newest_png(d))
            self.assertIsNone(newest_png(os.path.join(d, "does_not_exist")))

    def test_unstatable_file_is_skipped(self):
        # A PNG that vanishes between glob and stat must not crash the lookup:
        # newest_png runs inside a Tk callback where an unhandled OSError would
        # kill the event loop.
        with tempfile.TemporaryDirectory() as d:
            ghost = Path(d) / "ghost.png"
            real = Path(d) / "real.png"
            ghost.write_bytes(b"png")
            real.write_bytes(b"png")
            os.utime(ghost, (1000, 1000))
            os.utime(real, (2000, 2000))
            real_stat = Path.stat

            def fake_stat(self, **kwargs):
                if self == ghost:
                    raise OSError("No such file")
                return real_stat(self, **kwargs)

            with mock.patch.object(Path, "stat", fake_stat):
                self.assertEqual(newest_png(d), str(real))


class TestNewestMedia(unittest.TestCase):
    def test_returns_most_recent_media_and_ignores_preview(self):
        with tempfile.TemporaryDirectory() as d:
            older = Path(d) / "older.png"
            newer_video = Path(d) / "newer.mp4"
            preview = Path(d) / "_gui_preview.png"
            older.write_bytes(b"png")
            newer_video.write_bytes(b"mp4")
            preview.write_bytes(b"preview")
            os.utime(older, (1000, 1000))
            os.utime(newer_video, (2000, 2000))
            os.utime(preview, (3000, 3000))
            self.assertEqual(newest_media(d), str(newer_video))

    def test_ignores_non_media(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "log.txt").write_bytes(b"text")
            self.assertIsNone(newest_media(d))


class TestPresets(unittest.TestCase):
    def test_all_presets_have_valid_structure(self):
        self.assertGreaterEqual(len(gui.PRESETS), 5)
        for _name, p in gui.PRESETS.items():
            if p is None:
                continue
            self.assertIn("model", p)
            self.assertIn(p["model"], gui.MODEL_META)
            model_key = p["model"]
            valid_layers = [k for k, _ in gui.MODEL_META[model_key]["layers"]]
            self.assertIn("layer", p)
            self.assertIn(p["layer"], valid_layers)
            self.assertIn("iterations", p)
            self.assertGreater(p["iterations"], 0)
            self.assertIn("octaves", p)
            self.assertGreaterEqual(p["octaves"], 0)
            self.assertIn("octave_scale", p)
            self.assertGreater(p["octave_scale"], 1.0)
            self.assertIn("lr", p)
            self.assertGreater(p["lr"], 0)
            self.assertIn("jitter", p)
            self.assertGreaterEqual(p["jitter"], 0)
            self.assertIn("size", p)
            self.assertGreater(p["size"], 0)
            self.assertIn("desc", p)
            self.assertTrue(len(p["desc"]) > 0)

    def test_all_models_have_curated_presets(self):
        models_with_presets = set()
        for _name, p in gui.PRESETS.items():
            if p:
                models_with_presets.add(p["model"])
        for k in gui.MODEL_KEYS:
            self.assertIn(k, models_with_presets)
            count = sum(1 for p in gui.PRESETS.values() if p and p["model"] == k)
            self.assertGreaterEqual(count, 2, f"Model {k} should have at least 2 presets")


class TestTerminateProcess(unittest.TestCase):
    def test_none_is_noop(self):
        terminate_process(None)
        terminate_process(None, force=True)

    def test_calls_terminate_or_kill(self):
        calls = []

        class Proc:
            def terminate(self):
                calls.append("terminate")

            def kill(self):
                calls.append("kill")

        terminate_process(Proc())
        self.assertEqual(calls, ["terminate"])
        terminate_process(Proc(), force=True)
        self.assertEqual(calls, ["terminate", "kill"])

    def test_swallows_oserror_on_already_exited_proc(self):
        # A process that exits between poll() and terminate() raises OSError
        # on Windows; terminate_process must never propagate it from a Tk
        # callback.
        class FailingProc:
            def terminate(self):
                raise OSError("invalid handle")

            def kill(self):
                raise OSError("invalid handle")

        terminate_process(FailingProc())
        terminate_process(FailingProc(), force=True)


@pytest.mark.gui
class TestDeepDreamGUIInit(unittest.TestCase):
    def test_gui_init_and_refresh_slider_labels(self):
        try:
            app = gui.DeepDreamGUI()
        except Exception as exc:
            # Skip if running in headless environment without display
            if "no display name" in str(exc).lower() or "tcl" in str(exc).lower():
                self.skipTest("Tk display not available")
            raise
        try:
            app._refresh_slider_labels()
            self.assertGreater(len(app._slider_registry), 0)
        finally:
            app.destroy()

    def test_gui_init_with_initial_image(self):
        from PIL import Image

        try:
            app = gui.DeepDreamGUI()
        except Exception as exc:
            if "no display name" in str(exc).lower() or "tcl" in str(exc).lower():
                self.skipTest("Tk display not available")
            raise
        app.destroy()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            Image.new("RGB", (64, 64), color=(100, 150, 200)).save(f.name)
            img_path = f.name
        try:
            app = gui.DeepDreamGUI(initial_image=img_path)
            try:
                app.update_idletasks()
                self.assertEqual(app._image_path.get(), img_path)
            finally:
                app.destroy()
        finally:
            if os.path.exists(img_path):
                os.remove(img_path)


class TestCreateSplitImage(unittest.TestCase):
    def setUp(self):
        from PIL import Image

        self.red = Image.new("RGB", (100, 80), color=(255, 0, 0))
        self.blue = Image.new("RGB", (100, 80), color=(0, 0, 255))

    def test_split_ratio_zero_returns_right_image(self):
        comp = create_split_image(self.red, self.blue, split_ratio=0.0)
        self.assertEqual(comp.size, (100, 80))
        self.assertEqual(comp.getpixel((10, 40)), (0, 0, 255))
        self.assertEqual(comp.getpixel((90, 40)), (0, 0, 255))

    def test_split_ratio_one_returns_left_image(self):
        comp = create_split_image(self.red, self.blue, split_ratio=1.0)
        self.assertEqual(comp.size, (100, 80))
        self.assertEqual(comp.getpixel((10, 40)), (255, 0, 0))
        self.assertEqual(comp.getpixel((90, 40)), (255, 0, 0))

    def test_split_ratio_half_composites_and_draws_divider(self):
        comp = create_split_image(self.red, self.blue, split_ratio=0.5, divider_width=2)
        self.assertEqual(comp.size, (100, 80))
        # Left of divider should be red (img_left)
        self.assertEqual(comp.getpixel((10, 40)), (255, 0, 0))
        # Right of divider should be blue (img_right)
        self.assertEqual(comp.getpixel((90, 40)), (0, 0, 255))
        # Divider center line should be white (255, 255, 255)
        self.assertEqual(comp.getpixel((50, 40)), (255, 255, 255))

    def test_split_ratio_clamps_negative_and_excessive(self):
        comp_neg = create_split_image(self.red, self.blue, split_ratio=-0.5)
        self.assertEqual(comp_neg.getpixel((50, 40)), (0, 0, 255))
        comp_high = create_split_image(self.red, self.blue, split_ratio=1.5)
        self.assertEqual(comp_high.getpixel((50, 40)), (255, 0, 0))

    def test_different_resolutions_are_aligned(self):
        from PIL import Image

        large_red = Image.new("RGB", (300, 200), color=(255, 0, 0))
        small_blue = Image.new("RGB", (150, 100), color=(0, 0, 255))
        comp = create_split_image(large_red, small_blue, split_ratio=0.5)
        self.assertEqual(comp.size, (150, 100))
        self.assertEqual(comp.getpixel((20, 50)), (255, 0, 0))
        self.assertEqual(comp.getpixel((120, 50)), (0, 0, 255))


@pytest.mark.gui
class TestGUIComparison(unittest.TestCase):
    def setUp(self):
        try:
            self.app = gui.DeepDreamGUI()
        except Exception as exc:
            if "no display name" in str(exc).lower() or "tcl" in str(exc).lower():
                self.skipTest("Tk display not available")
            raise

    def tearDown(self):
        if hasattr(self, "app"):
            self.app.destroy()

    def test_toggle_compare_flips_between_dream_and_original(self):
        from PIL import Image

        self.app._dream_pil = Image.new("RGB", (100, 100), color=(255, 0, 0))
        self.app._aligned_input_pil = Image.new("RGB", (100, 100), color=(0, 255, 0))
        self.app._compare_mode = "Dream"

        self.app._toggle_compare()
        self.assertEqual(self.app._compare_mode, "Original")
        self.assertEqual(self.app._compare_seg.get(), "Original")

        self.app._toggle_compare()
        self.assertEqual(self.app._compare_mode, "Dream")
        self.assertEqual(self.app._compare_seg.get(), "Dream")

    def test_slide_reveal_mode_updates_preview_title(self):
        from PIL import Image

        self.app._dream_pil = Image.new("RGB", (100, 100), color=(255, 0, 0))
        self.app._aligned_input_pil = Image.new("RGB", (100, 100), color=(0, 255, 0))

        self.app._set_compare_mode("Slide Reveal")
        self.assertEqual(self.app._compare_mode, "Slide Reveal")
        self.assertIn("Slide to Reveal", self.app._preview_title.cget("text"))

        self.app._set_compare_mode("Dream")
        self.assertEqual(self.app._compare_mode, "Dream")

    def test_resolution_alignment_ensures_identical_on_screen_sizing(self):
        from PIL import Image

        # Original input is high res (1920x1080), output dream is low res (512x288)
        input_img = Image.new("RGB", (1920, 1080), color=(10, 20, 30))
        dream_img = Image.new("RGB", (512, 288), color=(40, 50, 60))

        self.app._input_pil = input_img
        self.app._dream_pil = dream_img
        dw, dh = dream_img.size

        # Align input
        self.app._aligned_input_pil = self.app._input_pil.resize((dw, dh), Image.Resampling.LANCZOS)
        self.assertEqual(self.app._aligned_input_pil.size, self.app._dream_pil.size)

        # The split view is built from the aligned pair, so it shares their size
        split_img = create_split_image(self.app._aligned_input_pil, self.app._dream_pil, 0.5)
        self.assertEqual(split_img.size, self.app._dream_pil.size)

    def test_space_key_in_entry_widget_does_not_toggle(self):
        import tkinter as tk

        from PIL import Image

        self.app._dream_pil = Image.new("RGB", (100, 100))
        self.app._aligned_input_pil = Image.new("RGB", (100, 100))
        self.app._compare_mode = "Dream"

        # Simulate keypress from an Entry widget
        entry = tk.Entry(self.app)
        mock_event = mock.Mock()
        mock_event.widget = entry

        self.app._on_space_pressed(mock_event)
        self.assertEqual(self.app._compare_mode, "Dream")

        # Simulate keypress from a non-entry widget (e.g. root frame)
        mock_event.widget = self.app
        self.app._on_space_pressed(mock_event)
        self.assertEqual(self.app._compare_mode, "Original")

    def test_load_input_image_for_job_reuses_same_loaded_input(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "input.png")
            Image.new("RGB", (12, 8), color=(10, 20, 30)).save(path)
            self.app._load_image_preview(path)
            loaded = self.app._input_pil
            self.assertIsNotNone(loaded)

            job = mock.Mock()
            job.image_path = path
            self.app._load_input_image_for_job(job)
            # The same file is already loaded: reuse it instead of clearing
            # and re-decoding (which also protects against a mid-run delete).
            self.assertIs(self.app._input_pil, loaded)

    def test_load_input_image_for_job_clears_missing_file(self):
        self.app._input_pil = object()
        self.app._input_pil_path = "stale-key"
        self.app._input_pil_mtime = 123
        job = mock.Mock()
        job.image_path = os.path.join("no", "such", "file.png")
        self.app._load_input_image_for_job(job)
        self.assertIsNone(self.app._input_pil)
        self.assertIsNone(self.app._input_pil_path)
        self.assertIsNone(self.app._input_pil_mtime)

    def test_load_input_image_for_job_keeps_cached_input_after_delete(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "input.png")
            Image.new("RGB", (12, 8), color=(10, 20, 30)).save(path)
            self.app._load_image_preview(path)
            loaded = self.app._input_pil
            job = mock.Mock()
            job.image_path = path
            # Input vanished mid-run: the copy loaded at run start is still
            # the right comparison image and must survive.
            os.remove(path)
            self.app._load_input_image_for_job(job)
            self.assertIs(self.app._input_pil, loaded)

    def test_load_input_image_for_job_reloads_modified_file(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "input.png")
            Image.new("RGB", (12, 8), color=(10, 20, 30)).save(path)
            self.app._load_image_preview(path)
            first = self.app._input_pil
            Image.new("RGB", (12, 8), color=(200, 100, 50)).save(path)
            st = os.stat(path)
            os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
            job = mock.Mock()
            job.image_path = path
            self.app._load_input_image_for_job(job)
            self.assertIsNot(self.app._input_pil, first)
            assert self.app._input_pil is not None
            self.assertEqual(self.app._input_pil.getpixel((0, 0)), (200, 100, 50))


class TestSliderMath(unittest.TestCase):
    def test_linear_curve(self):
        self.assertEqual(slider_val_to_t(10, 0, 100, curve=1.0), 0.1)
        self.assertEqual(slider_t_to_val(0.1, 0, 100, curve=1.0, is_int=True), 10)

    def test_iterations_curve_granularity_at_low_values(self):
        # 1 to 250 with curve 2.2
        # Values 1 to 3 should take > 10% of track
        t1 = slider_val_to_t(1, 1, 250, curve=2.2)
        t3 = slider_val_to_t(3, 1, 250, curve=2.2)
        self.assertEqual(t1, 0.0)
        self.assertGreater(t3, 0.10)

        # Values 150 to 250 should take ~20% of track (compressed)
        t150 = slider_val_to_t(150, 1, 250, curve=2.2)
        t250 = slider_val_to_t(250, 1, 250, curve=2.2)
        self.assertEqual(t250, 1.0)
        self.assertGreater(t150, 0.75)
        self.assertLessEqual(t250 - t150, 0.25)

    def test_slider_clamping_and_stepping(self):
        # Clamp out of bounds
        self.assertEqual(slider_val_to_t(-10, 0, 100), 0.0)
        self.assertEqual(slider_val_to_t(200, 0, 100), 1.0)
        self.assertEqual(slider_t_to_val(-0.5, 0, 100, is_int=True), 0)
        self.assertEqual(slider_t_to_val(1.5, 0, 100, is_int=True), 100)

        # Stepping: Max Size snaps to step=32
        val = slider_t_to_val(0.5, 0, 2048, curve=1.0, is_int=True, step=32)
        self.assertEqual(val % 32, 0)

        # Float stepping: Learning rate snaps to step=0.001
        val_f = slider_t_to_val(0.25, 0.001, 0.2, curve=1.0, is_int=False, step=0.001)
        self.assertAlmostEqual(round(val_f, 3), val_f)


@pytest.mark.gui
class TestSliderGUIControls(unittest.TestCase):
    def setUp(self):
        try:
            self.app = gui.DeepDreamGUI()
        except Exception as exc:
            if "no display name" in str(exc).lower() or "tcl" in str(exc).lower():
                self.skipTest("Tk display not available")
            raise

    def tearDown(self):
        if hasattr(self, "app"):
            self.app.destroy()

    def test_nudge_with_buttons_and_arrows(self):
        iter_item = next(
            item for item in self.app._slider_registry if item.var == self.app._iterations
        )
        self.app._iterations.set(20)
        self.app._refresh_slider_labels()
        self.assertEqual(self.app._iterations.get(), 20)

        # Nudge up with + (delta=1)
        iter_item.nudge(1)
        self.assertEqual(self.app._iterations.get(), 21)
        self.assertEqual(iter_item.val_label.cget("text"), "21")

        # Nudge down with - (delta=-1)
        iter_item.nudge(-1)
        self.assertEqual(self.app._iterations.get(), 20)
        self.assertEqual(iter_item.val_label.cget("text"), "20")

        # Hover arrow nudge via window handler
        self.app._active_hover_nudge = iter_item.nudge
        self.app._on_arrow_nudge(1)
        self.assertEqual(self.app._iterations.get(), 21)

        self.app._on_arrow_nudge(-1)
        self.assertEqual(self.app._iterations.get(), 20)

    def test_arrow_nudge_ignored_in_text_entry(self):
        import tkinter as tk

        iter_item = next(
            item for item in self.app._slider_registry if item.var == self.app._iterations
        )
        self.app._iterations.set(20)
        self.app._active_hover_nudge = iter_item.nudge

        entry = tk.Entry(self.app)
        event = mock.Mock()
        event.widget = entry

        self.app._on_arrow_nudge(1, event=event)
        self.assertEqual(self.app._iterations.get(), 20)

    def test_nudge_clamped_at_bounds(self):
        iter_item = next(
            item for item in self.app._slider_registry if item.var == self.app._iterations
        )
        # Lower bound
        self.app._iterations.set(1)
        iter_item.nudge(-1)
        self.assertEqual(self.app._iterations.get(), 1)

        # Upper bound
        self.app._iterations.set(250)
        iter_item.nudge(1)
        self.assertEqual(self.app._iterations.get(), 250)

    def test_preset_updates_sliders_and_knob_positions(self):
        self.app._on_preset_change("GoogLeNet Medium")
        iter_item = next(
            item for item in self.app._slider_registry if item.var == self.app._iterations
        )
        self.assertEqual(self.app._iterations.get(), 20)
        self.assertEqual(iter_item.val_label.cget("text"), "20")
        expected_t = slider_val_to_t(20, 1, 250, curve=2.2)
        self.assertAlmostEqual(iter_item.sl.get(), expected_t, places=2)

    def test_model_change_applies_calibrated_defaults(self):
        # InceptionV3
        self.app._on_model_change(gui.MODEL_META["inceptionv3"]["label"])
        self.assertEqual(self.app._iterations.get(), 30)
        self.assertAlmostEqual(self.app._lr.get(), 0.025, places=3)
        self.assertAlmostEqual(self.app._octave_scale.get(), 1.35, places=2)
        self.assertEqual(self.app._layer_var.get(), gui.MODEL_META["inceptionv3"]["recommended"])
        self.assertEqual(self.app._preset_var.get(), "Select a preset...")
        self.assertEqual(self.app._preset_menu.cget("values"), gui.presets_for_model("inceptionv3"))

        # VGG16
        self.app._on_model_change(gui.MODEL_META["vgg16"]["label"])
        self.assertEqual(self.app._iterations.get(), 20)
        self.assertAlmostEqual(self.app._lr.get(), 0.018, places=3)
        self.assertAlmostEqual(self.app._octave_scale.get(), 1.32, places=2)
        self.assertEqual(self.app._layer_var.get(), gui.MODEL_META["vgg16"]["recommended"])
        self.assertEqual(self.app._preset_menu.cget("values"), gui.presets_for_model("vgg16"))

        # VGG19
        self.app._on_model_change(gui.MODEL_META["vgg19"]["label"])
        self.assertEqual(self.app._iterations.get(), 25)
        self.assertAlmostEqual(self.app._lr.get(), 0.018, places=3)
        self.assertAlmostEqual(self.app._octave_scale.get(), 1.32, places=2)
        self.assertEqual(self.app._layer_var.get(), gui.MODEL_META["vgg19"]["recommended"])
        self.assertEqual(self.app._preset_menu.cget("values"), gui.presets_for_model("vgg19"))

        # GoogLeNet
        self.app._on_model_change(gui.MODEL_META["googlenet"]["label"])
        self.assertEqual(self.app._iterations.get(), 20)
        self.assertAlmostEqual(self.app._lr.get(), 0.03, places=3)
        self.assertAlmostEqual(self.app._octave_scale.get(), 1.4, places=2)
        self.assertEqual(self.app._layer_var.get(), gui.MODEL_META["googlenet"]["recommended"])
        self.assertEqual(self.app._preset_menu.cget("values"), gui.presets_for_model("googlenet"))

    def test_preset_preserves_custom_preset_values_and_name(self):
        self.app._on_preset_change("VGG16 Medium")
        self.assertEqual(self.app._model_var.get(), gui.MODEL_META["vgg16"]["label"])
        self.assertEqual(self.app._iterations.get(), 20)
        self.assertAlmostEqual(self.app._lr.get(), 0.018, places=3)
        self.assertAlmostEqual(self.app._octave_scale.get(), 1.32, places=2)
        self.assertEqual(self.app._preset_var.get(), "VGG16 Medium")
        self.assertEqual(self.app._preset_menu.cget("values"), gui.presets_for_model("vgg16"))

    def test_progress_tracker_single_frame_without_slash(self):
        tracker = ProgressTracker()
        frac, status = tracker.update("  Frame 12   loss 123.4")
        self.assertIsNone(frac)
        self.assertEqual(status, "Frame 12")

    def test_target_class_guidance_omits_layer_arg(self):
        self.app._image_path.set("photo.jpg")
        self.app._guidance_mode_var.set("Target Class")
        self.app._target_class_var.set("flamingo")
        self.app._custom_layers_var.set("")

        with (
            mock.patch("os.path.isfile", return_value=True),
            mock.patch("os.makedirs"),
            mock.patch("subprocess.Popen") as mock_popen,
            mock.patch("threading.Thread"),
        ):
            mock_proc = mock.Mock()
            mock_proc.stdout = []
            mock_popen.return_value = mock_proc
            self.app._run()

            args = mock_popen.call_args[0][0]
            self.assertIn("--class", args)
            self.assertIn("flamingo", args)
            self.assertNotIn("--layer", args)

    def test_target_class_guidance_includes_custom_layer_if_specified(self):
        self.app._image_path.set("photo.jpg")
        self.app._guidance_mode_var.set("Target Class")
        self.app._target_class_var.set("pizza")
        self.app._custom_layers_var.set("inception4c,inception5b")

        with (
            mock.patch("os.path.isfile", return_value=True),
            mock.patch("os.makedirs"),
            mock.patch("subprocess.Popen") as mock_popen,
            mock.patch("threading.Thread"),
        ):
            mock_proc = mock.Mock()
            mock_proc.stdout = []
            mock_popen.return_value = mock_proc
            self.app._run()

            args = mock_popen.call_args[0][0]
            self.assertIn("--class", args)
            self.assertIn("pizza", args)
            self.assertIn("--layer", args)
            self.assertIn("inception4c,inception5b", args)


@pytest.mark.gui
class TestGenerationQueue(unittest.TestCase):
    def setUp(self):
        try:
            self.app = gui.DeepDreamGUI()
        except Exception as exc:
            if "no display name" in str(exc).lower() or "tcl" in str(exc).lower():
                self.skipTest("Tk display not available")
            raise

    def tearDown(self):
        if hasattr(self, "app"):
            self.app.destroy()

    def test_generation_job_build_cmd(self):
        job = GenerationJob(
            id=1,
            image_path="sample.jpg",
            model_key="googlenet",
            layer_spec="inception4c",
            iterations=25,
            octaves=4,
            octave_scale=1.4,
            lr=0.03,
            size=512,
            jitter=32,
            out_dir="out",
            mode="Single Image",
            timelapse=True,
            custom_name="my_dream",
        )
        cmd = job.build_cmd(preview_tmp_path="out/_gui_preview.png")
        self.assertIn("sample.jpg", cmd)
        self.assertIn("--model", cmd)
        self.assertIn("googlenet", cmd)
        self.assertIn("--layer", cmd)
        self.assertIn("inception4c", cmd)
        self.assertIn("--iterations", cmd)
        self.assertIn("25", cmd)
        self.assertIn("--timelapse", cmd)
        self.assertIn("--name", cmd)
        self.assertIn("my_dream", cmd)
        self.assertIn("--preview-file", cmd)

    def test_generation_job_build_cmd_with_seed(self):
        job = GenerationJob(
            id=1,
            image_path="sample.jpg",
            model_key="googlenet",
            layer_spec="inception4c",
            iterations=25,
            octaves=4,
            octave_scale=1.4,
            lr=0.03,
            size=512,
            jitter=32,
            out_dir="out",
            mode="Single Image",
            seed=42,
        )
        cmd = job.build_cmd()
        self.assertIn("--seed", cmd)
        self.assertIn("42", cmd)

    def test_add_to_queue_updates_badge_and_list(self):
        job = GenerationJob(
            id=1,
            image_path="test.jpg",
            model_key="googlenet",
            layer_spec="inception4c",
            iterations=20,
            octaves=0,
            octave_scale=1.4,
            lr=0.03,
            size=512,
            jitter=32,
            out_dir="out",
        )
        self.app._add_to_queue(job, auto_start=False)
        self.assertEqual(len(self.app._job_queue), 1)
        self.assertEqual(self.app._get_queue_badge_text(), "Queue (1)")
        self.assertIn("Queue (1)", self.app._bottom_tab_seg.cget("values"))

    def test_remove_job_and_clear_pending(self):
        job1 = GenerationJob(
            id=1,
            image_path="img1.jpg",
            model_key="googlenet",
            layer_spec="inception4c",
            iterations=20,
            octaves=0,
            octave_scale=1.4,
            lr=0.03,
            size=512,
            jitter=32,
            out_dir="out",
            status="Pending",
        )
        job2 = GenerationJob(
            id=2,
            image_path="img2.jpg",
            model_key="vgg16",
            layer_spec="features.22",
            iterations=20,
            octaves=0,
            octave_scale=1.4,
            lr=0.03,
            size=512,
            jitter=32,
            out_dir="out",
            status="Pending",
        )
        self.app._add_to_queue(job1, auto_start=False)
        self.app._add_to_queue(job2, auto_start=False)
        self.assertEqual(len(self.app._job_queue), 2)

        # Remove job1
        self.app._remove_job(job1)
        self.assertEqual(len(self.app._job_queue), 1)
        self.assertEqual(self.app._job_queue[0].id, 2)

        # Clear remaining pending jobs
        self.app._clear_pending_jobs()
        self.assertEqual(len(self.app._job_queue), 0)

    def test_clear_completed_jobs_preserves_pending(self):
        job1 = GenerationJob(
            id=1,
            image_path="img1.jpg",
            model_key="googlenet",
            layer_spec="inception4c",
            iterations=20,
            octaves=0,
            octave_scale=1.4,
            lr=0.03,
            size=512,
            jitter=32,
            out_dir="out",
            status="Done",
        )
        job2 = GenerationJob(
            id=2,
            image_path="img2.jpg",
            model_key="vgg16",
            layer_spec="features.22",
            iterations=20,
            octaves=0,
            octave_scale=1.4,
            lr=0.03,
            size=512,
            jitter=32,
            out_dir="out",
            status="Pending",
        )
        self.app._job_queue = [job1, job2]
        self.app._clear_completed_jobs()
        self.assertEqual(len(self.app._job_queue), 1)
        self.assertEqual(self.app._job_queue[0].id, 2)

    def test_queue_auto_execution_sequence(self):
        job1 = GenerationJob(
            id=1,
            image_path="img1.jpg",
            model_key="googlenet",
            layer_spec="inception4c",
            iterations=20,
            octaves=0,
            octave_scale=1.4,
            lr=0.03,
            size=512,
            jitter=32,
            out_dir="out",
            status="Pending",
        )
        job2 = GenerationJob(
            id=2,
            image_path="img2.jpg",
            model_key="googlenet",
            layer_spec="inception4c",
            iterations=20,
            octaves=0,
            octave_scale=1.4,
            lr=0.03,
            size=512,
            jitter=32,
            out_dir="out",
            status="Pending",
        )
        self.app._job_queue = [job1, job2]

        executed = []

        def fake_execute(job):
            executed.append(job.id)
            mock_proc = mock.Mock()
            mock_proc.returncode = 0
            mock_proc.stdout = []
            mock_proc.poll.return_value = None
            self.app._proc = mock_proc

        with mock.patch.object(self.app, "_execute_job", side_effect=fake_execute):
            # Start queue
            self.app._start_queue()
            self.assertEqual(executed, [1])
            self.assertEqual(job1.status, "Running")
            self.assertEqual(job2.status, "Pending")

            # Simulate job1 finishes
            assert self.app._proc is not None
            self.app._proc.poll.return_value = 0
            self.app._on_process_finished()
            self.assertEqual(job1.status, "Done")

            # Queue automatically advances to job2
            self.app._process_next_job()
            self.assertEqual(executed, [1, 2])
            self.assertEqual(job2.status, "Running")

            # Simulate job2 finishes
            assert self.app._proc is not None
            self.app._proc.poll.return_value = 0
            self.app._on_process_finished()
            self.assertEqual(job2.status, "Done")
            self.assertFalse(self.app._queue_running)

    def test_stop_halts_queue_auto_execution(self):
        job1 = GenerationJob(
            id=1,
            image_path="img1.jpg",
            model_key="googlenet",
            layer_spec="inception4c",
            iterations=20,
            octaves=0,
            octave_scale=1.4,
            lr=0.03,
            size=512,
            jitter=32,
            out_dir="out",
            status="Pending",
        )
        job2 = GenerationJob(
            id=2,
            image_path="img2.jpg",
            model_key="googlenet",
            layer_spec="inception4c",
            iterations=20,
            octaves=0,
            octave_scale=1.4,
            lr=0.03,
            size=512,
            jitter=32,
            out_dir="out",
            status="Pending",
        )
        self.app._job_queue = [job1, job2]
        self.app._queue_running = True
        self.app._current_job = job1
        job1.status = "Running"

        mock_proc = mock.Mock()
        mock_proc.poll.return_value = None
        self.app._proc = mock_proc

        self.app._stop()
        self.assertFalse(self.app._queue_running)
        self.assertTrue(self.app._aborted)
        mock_proc.terminate.assert_called_once()

    def test_multi_file_drop_batch_queues_all_files(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as td:
            p1 = os.path.join(td, "a.jpg")
            p2 = os.path.join(td, "b.jpg")
            p3 = os.path.join(td, "c.jpg")
            Image.new("RGB", (64, 64)).save(p1)
            Image.new("RGB", (64, 64)).save(p2)
            Image.new("RGB", (64, 64)).save(p3)

            event = mock.Mock()
            event.data = f'"{p1}" "{p2}" "{p3}"'
            with mock.patch.object(self.app, "_start_queue"):
                self.app._on_drop(event)
                self.assertEqual(len(self.app._job_queue), 3)
                self.assertEqual(self.app._image_path.get(), p1)


@pytest.mark.gui
class TestJobCleanup(unittest.TestCase):
    """Coverage for the interrupted-child cleanup helpers used on Stop/close."""

    def setUp(self):
        try:
            self.app = gui.DeepDreamGUI()
        except Exception as exc:
            if "no display name" in str(exc).lower() or "tcl" in str(exc).lower():
                self.skipTest("Tk display not available")
            raise

    def tearDown(self):
        if hasattr(self, "app"):
            self.app.destroy()

    @staticmethod
    def _job(**kw):
        defaults: dict[str, Any] = {
            "id": 1,
            "image_path": "input.jpg",
            "model_key": "googlenet",
            "layer_spec": "inception4c",
            "iterations": 1,
            "octaves": 0,
            "octave_scale": 1.4,
            "lr": 0.03,
            "size": 512,
            "jitter": 0,
            "out_dir": "out",
        }
        defaults.update(kw)
        return GenerationJob(**defaults)

    def test_cleanup_job_temps_removes_only_this_pid(self):
        pid = 4242
        doomed = [
            f"a.png.p{pid}.tmp",
            f"b.png.p{pid}.temp.mp4",
            f"c.png.p{pid}.remux.temp.mp4",
        ]
        kept = ["keep.png", "a.png.p9999.tmp", f"b.png.p{pid}.tmp.bak"]
        with tempfile.TemporaryDirectory() as d:
            for name in doomed + kept:
                Path(d, name).write_bytes(b"x")
            self.app._cleanup_job_temps(d, pid)
            for name in doomed:
                self.assertFalse(os.path.exists(os.path.join(d, name)), name)
            for name in kept:
                self.assertTrue(os.path.exists(os.path.join(d, name)), name)

    def test_cleanup_job_temps_tolerates_missing_input(self):
        self.app._cleanup_job_temps("no/such/dir", 123)  # must not raise
        self.app._cleanup_job_temps("", None)

    def test_cleanup_partial_frames_removes_fresh_dir(self):
        with tempfile.TemporaryDirectory() as d:
            frames = os.path.join(d, "frames")
            os.makedirs(frames)
            Path(frames, "o1_i1.png").write_bytes(b"x")
            self.app._frames_dir = frames
            self.app._frames_preexisting = False
            self.app._cleanup_partial_frames()
            self.assertFalse(os.path.isdir(frames))

    def test_cleanup_partial_frames_keeps_preexisting_dir(self):
        with tempfile.TemporaryDirectory() as d:
            frames = os.path.join(d, "frames")
            os.makedirs(frames)
            frame = Path(frames, "o1_i1.png")
            frame.write_bytes(b"x")
            self.app._frames_dir = frames
            self.app._frames_preexisting = True
            self.app._cleanup_partial_frames()
            self.assertTrue(frame.exists())

    def test_mark_stale_job_stopped_updates_status(self):
        job = self._job(id=7, status="Running")
        self.app._job_queue = [job]
        self.app._aborted_jobs.add(7)
        self.app._mark_stale_job_stopped(7)
        self.assertEqual(job.status, "Stopped")
        self.assertNotIn(7, self.app._aborted_jobs)

    def test_view_job_output_loads_existing_file(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "dream.png")
            Image.new("RGB", (16, 16), color=(1, 2, 3)).save(out)
            job = self._job(id=8, out_dir=d, status="Done", output_path=out)
            self.app._view_job_output(job)
            self.assertEqual(self.app._output_path, out)
            self.assertIsNotNone(self.app._dream_pil)


@pytest.mark.gui
class TestVideoQualitySettings(unittest.TestCase):
    def setUp(self):
        try:
            self.app = gui.DeepDreamGUI()
        except Exception as exc:
            if "no display name" in str(exc).lower() or "tcl" in str(exc).lower():
                self.skipTest("Tk display not available")
            raise

    def tearDown(self):
        if hasattr(self, "app"):
            self.app.destroy()

    def test_video_quality_presets_update_crf_and_labels(self):
        self.assertEqual(self.app._video_quality_var.get(), "Balanced")
        self.assertEqual(self.app._video_crf.get(), 17)
        self.assertIn("CRF 17", self.app._video_quality_info_lbl.cget("text"))

        # Switch to Small
        self.app._on_video_quality_seg_change("Small")
        self.assertEqual(self.app._video_quality_var.get(), "Small")
        self.assertEqual(self.app._video_crf.get(), 28)
        self.assertIn("CRF 28", self.app._video_quality_info_lbl.cget("text"))

        # Switch to Lossless
        self.app._on_video_quality_seg_change("Lossless")
        self.assertEqual(self.app._video_quality_var.get(), "Lossless")
        self.assertEqual(self.app._video_crf.get(), 0)
        self.assertIn("CRF 0", self.app._video_quality_info_lbl.cget("text"))

        # Switch to Custom via CRF variable
        self.app._video_crf.set(12)
        self.assertEqual(self.app._video_quality_var.get(), "Custom")
        self.assertEqual(self.app._video_quality_seg.get(), "Custom")
        self.assertIn("CRF 12", self.app._video_quality_info_lbl.cget("text"))

    def test_video_quality_visibility_toggle(self):
        # Single Image mode without timelapse -> hidden
        self.app._mode_var.set("Single Image")
        self.app._timelapse.set(False)
        self.app._update_video_quality_visibility()
        self.assertEqual(self.app._video_quality_frame.grid_info(), {})

        # Dream Zoom mode -> visible
        self.app._on_mode_change("Dream Zoom")
        self.assertNotEqual(self.app._video_quality_frame.grid_info(), {})

        # Dream Video mode -> visible
        self.app._on_mode_change("Dream Video")
        self.assertNotEqual(self.app._video_quality_frame.grid_info(), {})

        # Single Image mode with timelapse -> visible
        self.app._on_mode_change("Single Image")
        self.assertEqual(self.app._video_quality_frame.grid_info(), {})
        self.app._timelapse.set(True)
        self.app._on_timelapse_toggle()
        self.assertNotEqual(self.app._video_quality_frame.grid_info(), {})

    def test_generation_job_video_quality_build_cmd(self):
        # Dream Zoom with Lossless
        job_zoom = GenerationJob(
            id=1,
            image_path="sample.jpg",
            model_key="googlenet",
            layer_spec="inception4c",
            iterations=20,
            octaves=0,
            octave_scale=1.4,
            lr=0.03,
            size=512,
            jitter=32,
            out_dir="out",
            mode="Dream Zoom",
            video_quality="Lossless",
            video_crf=0,
        )
        cmd_zoom = job_zoom.build_cmd()
        self.assertIn("--video-quality", cmd_zoom)
        self.assertIn("lossless", cmd_zoom)
        self.assertIn("--crf", cmd_zoom)
        self.assertIn("0", cmd_zoom)

        # Single image with Timelapse and Small preset
        job_tl = GenerationJob(
            id=2,
            image_path="sample.jpg",
            model_key="googlenet",
            layer_spec="inception4c",
            iterations=20,
            octaves=0,
            octave_scale=1.4,
            lr=0.03,
            size=512,
            jitter=32,
            out_dir="out",
            mode="Single Image",
            timelapse=True,
            video_quality="Small",
            video_crf=28,
        )
        cmd_tl = job_tl.build_cmd()
        self.assertIn("--timelapse", cmd_tl)
        self.assertIn("--video-quality", cmd_tl)
        self.assertIn("small", cmd_tl)
        self.assertIn("--crf", cmd_tl)
        self.assertIn("28", cmd_tl)

        # Single image without timelapse -> no video flags
        job_img = GenerationJob(
            id=3,
            image_path="sample.jpg",
            model_key="googlenet",
            layer_spec="inception4c",
            iterations=20,
            octaves=0,
            octave_scale=1.4,
            lr=0.03,
            size=512,
            jitter=32,
            out_dir="out",
            mode="Single Image",
            timelapse=False,
        )
        cmd_img = job_img.build_cmd()
        self.assertNotIn("--video-quality", cmd_img)
        self.assertNotIn("--crf", cmd_img)

    def test_mode_switch_decimates_and_restores_iterations(self):
        # Default GoogLeNet iterations: 20
        self.app._mode_var.set("Single Image")
        self.app._previous_mode = "Single Image"
        self.app._iterations.set(20)
        self.app._saved_single_image_iters = 20

        # Switch to Dream Zoom -> should decimate to 2 (20 / 10)
        self.app._on_mode_change("Dream Zoom")
        self.assertEqual(self.app._iterations.get(), 2)

        # Switch between video modes (Dream Zoom -> Dream Video) -> should not re-decimate
        self.app._on_mode_change("Dream Video")
        self.assertEqual(self.app._iterations.get(), 2)

        # Switch back to Single Image -> should restore to 20
        self.app._on_mode_change("Single Image")
        self.assertEqual(self.app._iterations.get(), 20)

        # Custom value: set to 50 in Single Image
        self.app._iterations.set(50)
        self.app._on_mode_change("Dream Zoom")
        self.assertEqual(self.app._iterations.get(), 5)

        # Switch back to Single Image -> restores to 50
        self.app._on_mode_change("Single Image")
        self.assertEqual(self.app._iterations.get(), 50)

        # Small iteration count (e.g. 4) clamped at minimum 1
        self.app._iterations.set(4)
        self.app._on_mode_change("Dream Video")
        self.assertEqual(self.app._iterations.get(), 1)
        self.app._on_mode_change("Single Image")
        self.assertEqual(self.app._iterations.get(), 4)

    def test_video_mode_model_and_preset_decimation(self):
        self.app._mode_var.set("Single Image")
        self.app._previous_mode = "Single Image"
        self.app._iterations.set(20)
        self.app._saved_single_image_iters = 20

        # Switch to video mode
        self.app._on_mode_change("Dream Zoom")
        self.assertEqual(self.app._iterations.get(), 2)

        # Change model to VGG19 (defaults: 25 iters)
        self.app._on_model_change("VGG19")
        # 25 / 10 = 2.5 -> round(2.5) = 2 or 3; int(round(25/10.0)) == 2
        self.assertEqual(self.app._iterations.get(), int(round(25 / 10.0)))
        self.assertEqual(self.app._saved_single_image_iters, 25)

        # Switch back to Single Image -> should restore to 25
        self.app._on_mode_change("Single Image")
        self.assertEqual(self.app._iterations.get(), 25)

        # Switch to Dream Zoom again
        self.app._on_mode_change("Dream Zoom")

        # Select preset "InceptionV3 High" (iterations: 36)
        self.app._on_preset_change("InceptionV3 High")
        self.assertEqual(self.app._iterations.get(), int(round(36 / 10.0)))
        self.assertEqual(self.app._saved_single_image_iters, 36)

        # Switch back to Single Image -> should restore to 36
        self.app._on_mode_change("Single Image")
        self.assertEqual(self.app._iterations.get(), 36)

    def test_tooltip_lifecycle(self):
        lbl = gui.tk.Label(self.app, text="Test Label")
        lbl.grid()
        tip = Tooltip(lbl, "Test tooltip message", delay_ms=0)
        self.assertIsNone(tip.tip)

        # Show tooltip
        tip._show()
        self.assertIsNotNone(tip.tip)
        assert tip.tip is not None
        self.assertTrue(tip.tip.winfo_exists())

        # Hide tooltip
        tip._hide()
        self.assertIsNone(tip.tip)
        lbl.destroy()

    def test_layer_mode_toggle_single_to_multi(self):
        self.assertEqual(self.app._layer_mode_var.get(), "Single Layer")
        self.app._on_layer_mode_change("Multi-Layer")
        self.assertEqual(self.app._layer_mode_var.get(), "Multi-Layer")
        # Should initialize with current single layer
        self.assertTrue(len(self.app._custom_layers_var.get().strip()) > 0)
        self.assertIn("features.22", self.app._custom_layers_var.get())

    def test_layer_mode_toggle_multi_to_single(self):
        self.app._on_layer_mode_change("Multi-Layer")
        self.app._custom_layers_var.set("inception4c,inception5b")
        self.app._on_layer_mode_change("Single Layer")
        self.assertEqual(self.app._layer_mode_var.get(), "Single Layer")
        self.assertEqual(self.app._custom_layers_var.get(), "")

    def test_remove_layer_chip(self):
        self.app._on_layer_mode_change("Multi-Layer")
        self.app._custom_layers_var.set("inception4c,inception5b,inception3a")
        self.app._remove_layer_chip("inception5b")
        self.assertEqual(self.app._custom_layers_var.get(), "inception4c,inception3a")
        self.app._remove_layer_chip("inception4c")
        self.assertEqual(self.app._custom_layers_var.get(), "inception3a")
        self.app._remove_layer_chip("inception3a")
        self.assertEqual(self.app._custom_layers_var.get(), "")

    def test_clear_multi_layers(self):
        self.app._on_layer_mode_change("Multi-Layer")
        self.app._custom_layers_var.set("inception4c,inception5b")
        self.app._clear_multi_layers()
        self.assertEqual(self.app._custom_layers_var.get(), "")

    def test_preset_switches_layer_mode(self):
        self.app._on_layer_mode_change("Multi-Layer")
        self.app._custom_layers_var.set("inception4c,inception5b")
        self.app._on_preset_change("GoogLeNet High")
        self.assertEqual(self.app._layer_mode_var.get(), "Single Layer")
        self.assertEqual(self.app._custom_layers_var.get(), "")

        self.app._on_preset_change("GoogLeNet Low")
        self.assertEqual(self.app._layer_mode_var.get(), "Single Layer")
        self.assertEqual(self.app._custom_layers_var.get(), "")

    def test_model_change_resets_layer_mode(self):
        self.app._on_layer_mode_change("Multi-Layer")
        self.app._custom_layers_var.set("inception4c,inception5b")
        self.app._on_model_change("InceptionV3")
        self.assertEqual(self.app._layer_mode_var.get(), "Single Layer")
        self.assertEqual(self.app._custom_layers_var.get(), "")

    def test_toggle_layer_activate_and_deactivate(self):
        self.app._on_layer_mode_change("Multi-Layer")
        self.app._custom_layers_var.set("inception4c")
        # Click inactive layer to activate
        self.app._toggle_layer("inception5b")
        self.assertIn("inception5b", self.app._custom_layers_var.get())
        self.assertIn("inception4c", self.app._custom_layers_var.get())

        # Click active layer to deactivate
        self.app._toggle_layer("inception4c")
        self.assertNotIn("inception4c", self.app._custom_layers_var.get())
        self.assertIn("inception5b", self.app._custom_layers_var.get())

    def test_select_recommended_and_all_layers(self):
        self.app._on_layer_mode_change("Multi-Layer")
        self.app._select_recommended_layers()
        self.assertEqual(self.app._custom_layers_var.get(), "features.15,features.22")

        self.app._select_all_layers()
        for layer_key, _ in gui.MODEL_META["vgg16"]["layers"]:
            self.assertIn(layer_key, self.app._custom_layers_var.get())

    def test_integrated_layer_buttons_render_active_and_inactive(self):
        self.app._on_layer_mode_change("Multi-Layer")
        self.app._custom_layers_var.set("features.22")
        self.app._refresh_layer_chips()

        # Find buttons inside _layer_chips_container
        buttons = [
            child
            for child in self.app._layer_chips_container.winfo_children()
            if isinstance(child, ModernButton)
        ]
        button_texts = [b.cget("text") for b in buttons]

        # Active layer should have checkmark prefix
        self.assertTrue(any("✓ features.22" in t for t in button_texts))
        # Inactive layers should have plus prefix
        self.assertTrue(any("+ features.3" in t for t in button_texts))
        self.assertTrue(any("+ features.29" in t for t in button_texts))

    def test_target_class_frame_visibility_toggle(self):
        # Initially unguided, class frame should not be in grid
        self.app._on_guidance_change("Unguided")
        self.assertFalse(self.app._class_frame.winfo_ismapped())

        # Switch to Target Class -> class frame should be mapped / visible
        self.app._on_guidance_change("Target Class")
        self.app.update_idletasks()
        self.assertTrue(self.app._class_frame.winfo_ismapped())

        # Switch back to Unguided -> class frame hidden
        self.app._on_guidance_change("Unguided")
        self.app.update_idletasks()
        self.assertFalse(self.app._class_frame.winfo_ismapped())

    def test_target_class_search_and_filter(self):
        # Search by name substring
        self.app._class_search_var.set("pizza")
        dropdown_vals = self.app._class_dropdown.cget("values")
        self.assertTrue(any("pizza" in v.lower() for v in dropdown_vals))
        listbox_size = self.app._class_listbox.size()
        self.assertGreater(listbox_size, 0)
        first_item = self.app._class_listbox.get(0)
        self.assertIn("pizza", first_item.lower())

        # Search by index
        self.app._class_search_var.set("130")
        dropdown_vals = self.app._class_dropdown.cget("values")
        self.assertTrue(any("130" in v and "flamingo" in v for v in dropdown_vals))

        # Clear search
        self.app._clear_class_search()
        self.assertEqual(self.app._class_search_var.get(), "")

        # Pressing enter selects top match
        self.app._class_search_var.set("pizza")
        self.app._on_class_search_enter()
        self.assertIn("pizza", self.app._target_class_var.get().lower())

    def test_target_class_category_filter_segment(self):
        self.app._on_class_category_filter_change("🍕 Food")
        dropdown_vals = self.app._class_dropdown.cget("values")
        self.assertTrue(
            any("pizza" in v or "cheeseburger" in v or "ice cream" in v for v in dropdown_vals)
        )

    def test_target_class_dropdown_and_listbox_selection(self):
        # Select from dropdown
        self.app._on_class_dropdown_select("#130: flamingo")
        self.assertEqual(self.app._target_class_var.get(), "#130: flamingo")
        lbl_text = self.app._class_selected_label.cget("text")
        self.assertIn("flamingo", lbl_text)

        # Select via listbox
        self.app._class_listbox.selection_clear(0, "end")
        self.app._class_listbox.selection_set(0)
        first_text = self.app._class_listbox.get(0).strip()
        self.app._on_class_listbox_select()
        self.assertEqual(self.app._target_class_var.get(), first_text)

    def test_target_class_clear_button(self):
        self.app._target_class_var.set("#130: flamingo")
        self.app._clear_target_class()
        self.assertEqual(self.app._target_class_var.get(), "")
        lbl_text = self.app._class_selected_label.cget("text")
        self.assertIn("No target selected", lbl_text)

    def test_job_creation_with_target_class(self):
        job = GenerationJob(
            id=10,
            image_path="sample.jpg",
            model_key="googlenet",
            layer_spec="",
            iterations=20,
            octaves=0,
            octave_scale=1.4,
            lr=0.03,
            size=512,
            jitter=32,
            out_dir="out",
            mode="Single Image",
            guidance_mode="Target Class",
            target_class="#130: flamingo",
        )
        cmd = job.build_cmd()
        self.assertIn("--class", cmd)
        self.assertIn("#130: flamingo", cmd)
        self.assertNotIn("--layer", cmd)


class TestSavedVideoFramesHelpers(unittest.TestCase):
    def test_list_prefers_numbered_frames(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as d:
            Image.new("RGB", (8, 8)).save(os.path.join(d, "frame_000002.png"))
            Image.new("RGB", (8, 8)).save(os.path.join(d, "frame_000001.png"))
            Image.new("RGB", (8, 8)).save(os.path.join(d, "o1_i1.png"))
            with open(os.path.join(d, "_gui_preview_p1_j1.png"), "wb") as fh:
                fh.write(b"x")
            self.assertEqual(
                [os.path.basename(p) for p in list_saved_video_frames(d)],
                ["frame_000001.png", "frame_000002.png"],
            )
            self.assertEqual(count_saved_video_frames(d), 2)

    def test_list_falls_back_to_generic_pngs(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as d:
            Image.new("RGB", (8, 8)).save(os.path.join(d, "o1_i1.png"))
            Image.new("RGB", (8, 8)).save(os.path.join(d, "o1_i2.png"))
            self.assertEqual(count_saved_video_frames(d), 2)

    def test_list_and_meta_tolerate_missing_dir(self):
        missing = os.path.join("no", "such", "dir")
        self.assertEqual(list_saved_video_frames(missing), [])
        self.assertEqual(count_saved_video_frames(missing), 0)
        self.assertEqual(read_video_frames_meta(missing), {})

    def test_zoom_build_cmd_uses_frames_only_with_frames_dir(self):
        job = GenerationJob(
            id=1,
            image_path="sample.jpg",
            model_key="googlenet",
            layer_spec="inception4c",
            iterations=2,
            octaves=0,
            octave_scale=1.4,
            lr=0.03,
            size=512,
            jitter=32,
            out_dir="out",
            mode="Dream Zoom",
            frames_dir=os.path.join("out", "frames_job1_p1"),
        )
        cmd = job.build_cmd()
        self.assertIn("--zoom", cmd)
        self.assertIn("--frames-dir", cmd)
        self.assertIn("--frames-only", cmd)
        # Quality is an export-time decision now, not a generation flag.
        self.assertNotIn("--video-quality", cmd)
        self.assertNotIn("--crf", cmd)

    def test_video_build_cmd_uses_frames_only_with_frames_dir(self):
        job = GenerationJob(
            id=2,
            image_path="clip.mp4",
            model_key="googlenet",
            layer_spec="inception4c",
            iterations=2,
            octaves=0,
            octave_scale=1.4,
            lr=0.03,
            size=512,
            jitter=32,
            out_dir="out",
            mode="Dream Video",
            frames_dir=os.path.join("out", "frames_job2_p1"),
        )
        cmd = job.build_cmd()
        self.assertIn("--video", cmd)
        self.assertIn("--frames-only", cmd)

    def test_single_image_frames_job_clears_its_own_frames_dir(self):
        frames_dir = os.path.join("out", "frames_job1_p1")
        job = GenerationJob(
            id=1,
            image_path="sample.jpg",
            model_key="googlenet",
            layer_spec="inception4c",
            iterations=2,
            octaves=0,
            octave_scale=1.4,
            lr=0.03,
            size=512,
            jitter=32,
            out_dir="out",
            mode="Single Image",
            timelapse=True,
            frames_dir=frames_dir,
        )
        cmd = job.build_cmd()
        self.assertIn("--frames-dir", cmd)
        self.assertIn(frames_dir, cmd)
        # A retry must be allowed to replace this job's own iteration frames,
        # or the CLI's default safety check would abort it.
        self.assertIn("--clear-frames", cmd)

    def test_single_image_job_without_frames_dir_omits_clear_frames(self):
        job = GenerationJob(
            id=2,
            image_path="sample.jpg",
            model_key="googlenet",
            layer_spec="inception4c",
            iterations=2,
            octaves=0,
            octave_scale=1.4,
            lr=0.03,
            size=512,
            jitter=32,
            out_dir="out",
            mode="Single Image",
            timelapse=True,
        )
        cmd = job.build_cmd()
        self.assertIn("--timelapse", cmd)
        self.assertNotIn("--clear-frames", cmd)


@pytest.mark.gui
class TestVideoExport(unittest.TestCase):
    def setUp(self):
        try:
            self.app = gui.DeepDreamGUI()
        except Exception as exc:
            if "no display name" in str(exc).lower() or "tcl" in str(exc).lower():
                self.skipTest("Tk display not available")
            raise

    def tearDown(self):
        if hasattr(self, "app"):
            loader = getattr(self.app, "_video_loader", None)
            if loader is not None:
                loader.wait_idle()
                loader.close()
            self.app.destroy()

    def _settle_video_loader(self):
        """Wait for async decodes and apply their results.

        Video previews now decode off the Tk thread, so a test that finishes an
        export must let the loader drain before its temp files can be removed.
        """
        loader = getattr(self.app, "_video_loader", None)
        if loader is None:
            return
        loader.wait_idle(timeout=5.0)
        self.app._poll_video_loader()

    @staticmethod
    def _make_frames(d, n=4, fps=24, mode="zoom", source=None, job_id=1):
        from PIL import Image

        fd = os.path.join(d, f"frames_job{job_id}_p1")
        os.makedirs(fd, exist_ok=True)
        for i in range(1, n + 1):
            Image.new("RGB", (64, 48), color=(i * 40, 100, 150)).save(
                os.path.join(fd, f"frame_{i:06d}.png")
            )
        with open(os.path.join(fd, "meta.json"), "w", encoding="utf-8") as fh:
            import json

            json.dump(
                {"mode": mode, "fps": fps, "source": source, "width": 64, "height": 48},
                fh,
            )
        return fd

    def _ready_job(self, d, **kw):
        fd = self._make_frames(d, **kw.pop("frames_kw", {}))
        job = GenerationJob(
            id=1,
            image_path=kw.pop("image_path", "sample.jpg"),
            model_key="googlenet",
            layer_spec="inception4c",
            iterations=2,
            octaves=0,
            octave_scale=1.4,
            lr=0.03,
            size=512,
            jitter=32,
            out_dir=d,
            mode=kw.pop("mode", "Dream Zoom"),
            zoom_fps=24,
            frames_dir=fd,
            status="Running",
        )
        self.app._job_queue = [job]
        self.app._current_job = job
        self.app._load_video_frames_ready(job, fd)
        # The frames preview decodes PNGs on a worker thread. On Windows an
        # open file cannot be deleted, so settle the loader before the caller
        # exits its TemporaryDirectory context.
        self._settle_video_loader()
        return job, fd

    def _save_video_command(self, out_mp4):
        worker_called = threading.Event()
        with (
            mock.patch("tkinter.filedialog.asksaveasfilename", return_value=out_mp4),
            mock.patch.object(self.app, "_start_encode_process", return_value=mock.Mock()) as start,
            mock.patch.object(
                self.app,
                "_run_video_encode",
                side_effect=lambda *_args: worker_called.set(),
            ) as run,
        ):
            self.app._on_save_video_click()
            self.assertTrue(worker_called.wait(timeout=5), "video encode worker did not start")
            command = run.call_args.args[0]
            start.assert_called_once_with(command)
            return command

    def test_frames_ready_enables_save(self):
        with tempfile.TemporaryDirectory() as d:
            job, fd = self._ready_job(d)
            self.assertEqual(self.app._video_frames_dir, fd)
            self.assertEqual(self.app._video_frames_count, 4)
            self.assertAlmostEqual(self.app._video_frames_fps, 24.0)
            self.assertEqual(self.app._save_video_btn.cget("state"), "normal")
            self.assertIn("Frames Ready", self.app._preview_title.cget("text"))
            self.assertIsNone(self.app._output_path)
            self.assertIn("frames ready", self.app._video_frames_label.cget("text").lower())

    def test_save_click_builds_encode_cmd(self):
        with tempfile.TemporaryDirectory() as d:
            self._ready_job(d)
            out_mp4 = os.path.join(d, "export.mp4")
            cmd = self._save_video_command(out_mp4)
            self.assertIn("--encode-frames", cmd)
            self.assertIn("--crf", cmd)
            self.assertIn("--fps", cmd)
            self.assertEqual(cmd[cmd.index("--fps") + 1], "24.00")
            self.assertNotIn("--encode-audio-from", cmd)

    def test_export_completion_keeps_started_quality_when_controls_change(self):
        with tempfile.TemporaryDirectory() as d:
            self._ready_job(d)
            for started_crf, later_crf in ((17, 28), (0, 17)):
                with self.subTest(started_crf=started_crf):
                    self.app._video_crf.set(started_crf)
                    out_mp4 = os.path.join(d, f"export-{started_crf}.mp4")
                    cmd = self._save_video_command(out_mp4)
                    self.assertEqual(cmd[cmd.index("--crf") + 1], str(started_crf))
                    self.app._video_crf.set(later_crf)
                    Path(out_mp4).write_bytes(b"completed export")
                    with mock.patch.object(self.app, "_setup_mp4_preview", return_value=False):
                        self.app._finish_video_encode(0, "", out_mp4, 1)
                    self.assertEqual(self.app._video_last_crf, started_crf)
                    self.assertIn(f"CRF {started_crf}", self.app._video_frames_label.cget("text"))
                    self.assertIsNone(self.app._video_export_request)

    def test_export_start_failure_clears_snapshot_and_keeps_frames(self):
        with tempfile.TemporaryDirectory() as d:
            self._ready_job(d)
            with (
                mock.patch("tkinter.filedialog.asksaveasfilename", return_value=f"{d}/failed.mp4"),
                mock.patch.object(self.app, "_start_encode_process", side_effect=OSError("failed")),
                mock.patch.object(gui.messagebox, "showerror"),
            ):
                self.app._on_save_video_click()
            self.assertIsNone(self.app._video_export_request)
            self.assertFalse(self.app._video_save_in_progress)
            self.assertIsNone(self.app._encode_proc)
            self.assertTrue(self.app._has_ready_video_frames())

    def test_stopped_export_completion_clears_snapshot_and_keeps_frames(self):
        with tempfile.TemporaryDirectory() as d:
            _job, frames_dir = self._ready_job(d)
            existing_frames = set(Path(frames_dir).iterdir())
            out_mp4 = os.path.join(d, "stopped.mp4")
            self._save_video_command(out_mp4)
            proc = self.app._encode_proc
            proc.poll.return_value = None
            with mock.patch.object(gui, "terminate_process") as terminate:
                self.app._stop()
            terminate.assert_called_once_with(proc)
            # Completion remains owned by the queue, not the Stop handler.
            self.assertIs(self.app._encode_proc, proc)
            self.assertIsNotNone(self.app._video_export_request)
            self.app._encode_queue.put(("done", 1, "cancelled", out_mp4, 1))
            with mock.patch.object(gui.messagebox, "showerror"):
                self.app._drain_encode_queue()
            self.assertIsNone(self.app._video_export_request)
            self.assertIsNone(self.app._encode_proc)
            self.assertFalse(self.app._video_save_in_progress)
            self.assertEqual(set(Path(frames_dir).iterdir()), existing_frames)
            self.assertTrue(self.app._has_ready_video_frames())

    def test_save_click_includes_audio_for_video_mode(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "clip.mp4")
            with open(src, "wb") as fh:
                fh.write(b"fake")
            self._ready_job(
                d,
                mode="Dream Video",
                image_path=src,
                frames_kw={"mode": "video", "source": src},
            )
            out_mp4 = os.path.join(d, "export.mp4")
            cmd = self._save_video_command(out_mp4)
            self.assertIn("--encode-audio-from", cmd)
            self.assertEqual(cmd[cmd.index("--encode-audio-from") + 1], src)

    def test_save_cancel_does_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            self._ready_job(d)
            with (
                mock.patch("tkinter.filedialog.asksaveasfilename", return_value=""),
                mock.patch.object(self.app, "_run_video_encode") as mrun,
            ):
                self.app._on_save_video_click()
                mrun.assert_not_called()
                self.assertIsNone(self.app._video_export_request)

    def test_finish_save_marks_output_and_stays_savable(self):
        with tempfile.TemporaryDirectory() as d:
            job, _fd = self._ready_job(d)
            out_mp4 = os.path.join(d, "export.mp4")
            with open(out_mp4, "wb") as fh:
                fh.write(b"x" * 64)
            self.app._video_save_in_progress = True
            self.app._finish_video_encode(0, "Done in 0.5s -> " + out_mp4 + "\n", out_mp4, 1)
            self._settle_video_loader()
            self.assertEqual(self.app._output_path, out_mp4)
            self.assertEqual(job.output_path, out_mp4)
            # Multi-save: frames context retained, button back to normal.
            self.assertEqual(self.app._save_video_btn.cget("state"), "normal")
            self.assertIn("export.mp4", self.app._video_frames_label.cget("text"))

    def test_finish_save_failure_keeps_frames(self):
        with tempfile.TemporaryDirectory() as d:
            self._ready_job(d)
            self.app._video_save_in_progress = True
            with mock.patch.object(gui.messagebox, "showerror"):
                self.app._finish_video_encode(1, "boom\n", os.path.join(d, "x.mp4"), 1)
            self.assertIsNone(self.app._output_path)
            self.assertTrue(self.app._has_ready_video_frames())
            self.assertEqual(self.app._save_video_btn.cget("state"), "normal")

    def test_save_click_resets_progress_during_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            self._ready_job(d)
            existing_mp4 = os.path.join(d, "export.mp4")
            with open(existing_mp4, "wb") as fh:
                fh.write(b"video data")
            # Simulate a previously finished video state.
            self.app._output_path = existing_mp4
            self.app._progress.set(1.0)
            self.app._status_label.configure(text="Video saved ✓")

            with (
                mock.patch("tkinter.filedialog.asksaveasfilename", return_value=existing_mp4),
                mock.patch.object(self.app, "_run_video_encode") as mrun,
            ):
                self.app._on_save_video_click()
                self.assertTrue(mrun.called)
                # Overwriting invalidates the previous output while encoding.
                self.assertIsNone(self.app._output_path)
                self.assertEqual(self.app._progress.get(), 0.0)
                self.assertEqual(self.app._status_label.cget("text"), "Encoding video...")
                self.assertEqual(self.app._save_video_btn.cget("state"), "disabled")
                self.assertIn(
                    "Encoding video -> export.mp4...", self.app._video_frames_label.cget("text")
                )
                self.assertIn(
                    "Encoding Video -> export.mp4...", self.app._preview_title.cget("text")
                )

    def test_poll_log_uses_structured_events(self):
        # Events drive progress and output discovery; they are not shown in the
        # log, and reworded CLI text can no longer break them.
        with tempfile.TemporaryDirectory() as d:
            self.app._tracker = ProgressTracker()
            self.app._log_queue.put(format_event(EV_OCTAVE, current=2, total=4) + "\n")
            self.app._log_queue.put(format_event(EV_PREVIEW, path=os.path.join(d, "p.png")) + "\n")
            self.app._log_queue.put(format_event(EV_DONE, path=os.path.join(d, "out.png")) + "\n")
            self.app._poll_log()
            self.assertAlmostEqual(self.app._progress.get(), 1.0)
            self.assertEqual(self.app._expected_output, os.path.join(d, "out.png"))
            self.assertEqual(self.app._status_label.cget("text"), "Done ✓")
            log_text = (
                self.app._log_text.get("1.0", "end") if hasattr(self.app, "_log_text") else ""
            )
            self.assertNotIn("dd-event", log_text)

    def test_poll_log_still_parses_legacy_text(self):
        # A CLI without events (older version, or a bare subprocess) still
        # reports progress through the text fallback.
        with tempfile.TemporaryDirectory() as d:
            self.app._tracker = ProgressTracker()
            self.app._log_queue.put("Octave 1/2  image 512x512\n")
            self.app._log_queue.put("Done in 1.0s -> " + os.path.join(d, "legacy.png") + "\n")
            self.app._poll_log()
            self.assertEqual(self.app._expected_output, os.path.join(d, "legacy.png"))
            self.assertEqual(self.app._status_label.cget("text"), "Done ✓")

    def test_poll_log_keeps_newest_preview_only(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as d:
            first = os.path.join(d, "a.png")
            last = os.path.join(d, "b.png")
            Image.new("RGB", (8, 8)).save(first)
            Image.new("RGB", (8, 8)).save(last)
            shown: list[Any] = []
            with mock.patch.object(
                type(self.app), "_show_output_preview", side_effect=shown.append
            ):
                self.app._log_queue.put(f"[PREVIEW] {first}\n")
                self.app._log_queue.put(f"[PREVIEW] {last}\n")
                self.app._poll_log()
            self.assertEqual(len(shown), 1)
            self.assertEqual(shown[0].size, (8, 8))

    def test_encode_worker_queues_events_without_touching_tk(self):
        # _run_video_encode runs on a worker thread: it must only put events on
        # the queue, never call Tk, so the Tk thread applies them.
        with tempfile.TemporaryDirectory() as d:
            self._ready_job(d)
            out_mp4 = os.path.join(d, "export.mp4")

            class FakeStdout:
                def __init__(self, lines):
                    self._it = iter(lines)

                def readline(self):
                    return next(self._it, "")

                def close(self):
                    pass

            class FakeProc:
                def __init__(self):
                    self.stdout = FakeStdout(
                        ["Encoding frame 1/2 (50%)\n", "Encoding frame 2/2 (100%)\n"]
                    )

                def wait(self):
                    self.returncode = 0
                    return 0

                def poll(self):
                    return self.returncode

            proc = FakeProc()
            with mock.patch.object(gui.subprocess, "Popen") as mpop:
                started = self.app._start_encode_process(["fake"])
            mpop.assert_called_once()
            self.assertIs(started, mpop.return_value)

            self.app._video_save_in_progress = True
            with mock.patch.object(
                type(self.app), "after", side_effect=AssertionError("Tk called off-thread")
            ):
                self.app._run_video_encode(["fake"], proc, out_mp4, 1, 2, 24.0, 17)
            self.assertEqual(self.app._video_save_in_progress, True)
            events = []
            while not self.app._encode_queue.empty():
                events.append(self.app._encode_queue.get_nowait())
            self.assertEqual([e[0] for e in events], ["line", "line", "done"])
            self.assertEqual(events[-1][1], 0)
            self.assertEqual(events[-1][2], "")

    def test_drain_encode_queue_applies_lines_then_finishes(self):
        with tempfile.TemporaryDirectory() as d:
            self._ready_job(d)
            out_mp4 = os.path.join(d, "export.mp4")
            with open(out_mp4, "wb") as fh:
                fh.write(b"x" * 64)
            self.app._video_save_in_progress = True
            self.app._encode_proc = mock.Mock()
            self.app._encode_queue.put(("line", "Encoding frame 10/20 (50%)\n", out_mp4))
            self.app._encode_queue.put(("done", 0, "", out_mp4, 1))
            self.app._drain_encode_queue()
            self._settle_video_loader()
            self.assertEqual(self.app._output_path, out_mp4)
            self.assertFalse(self.app._video_save_in_progress)
            self.assertIsNone(self.app._encode_proc)

    def test_poll_log_keeps_polling_while_encode_running(self):
        with tempfile.TemporaryDirectory() as d:
            self._ready_job(d)
            self.app._video_save_in_progress = True
            with mock.patch.object(type(self.app), "after", return_value="poll-id") as mafter:
                self.app._poll_log()
            mafter.assert_called()
            self.assertEqual(mafter.call_args[0][1], self.app._poll_log)
            self.app._video_save_in_progress = False

    def test_poll_log_keeps_polling_when_job_finishes_during_encode(self):
        # A finishing dream job must not stop the poll that drains a running
        # video export, or the export's "done" event would be stranded and the
        # GUI would stay on "Encoding..." with Save Video disabled.
        proc = mock.Mock()
        self.app._proc = proc
        self.app._video_save_in_progress = True
        self.app._log_queue.put(f"{PROC_DONE_PREFIX}{id(proc)}:1")
        with (
            mock.patch.object(self.app, "_on_process_finished") as mfin,
            mock.patch.object(type(self.app), "after", return_value="poll-id") as mafter,
        ):
            self.app._poll_log()
        mfin.assert_called_once()
        mafter.assert_called()
        self.assertEqual(mafter.call_args[0][1], self.app._poll_log)
        self.app._video_save_in_progress = False

    def test_encode_process_start_failure_reports_error(self):
        with tempfile.TemporaryDirectory() as d:
            self._ready_job(d)
            out_mp4 = os.path.join(d, "export.mp4")
            with (
                mock.patch("tkinter.filedialog.asksaveasfilename", return_value=out_mp4),
                mock.patch.object(
                    self.app, "_start_encode_process", side_effect=OSError("no ffmpeg")
                ),
                mock.patch.object(gui.messagebox, "showerror"),
            ):
                self.app._on_save_video_click()
            self.assertFalse(self.app._video_save_in_progress)
            self.assertIsNone(self.app._output_path)
            self.assertIsNone(self.app._encode_proc)

    def test_handle_encode_line_updates_progress_and_labels(self):
        with tempfile.TemporaryDirectory() as d:
            self._ready_job(d)
            out_mp4 = os.path.join(d, "export.mp4")
            self.app._video_save_in_progress = True
            self.app._video_export_target = out_mp4
            self.app._tracker = ProgressTracker()

            self.app._handle_encode_line("Encoding frame 10/20 (50%)\n", out_mp4)
            self.assertAlmostEqual(self.app._progress.get(), 0.5)
            self.assertEqual(self.app._status_label.cget("text"), "Encoding 10/20")
            self.assertIn("Encoding 10/20 -> export.mp4", self.app._video_frames_label.cget("text"))

            self.app._handle_encode_line("Remuxing audio track...\n", out_mp4)
            self.assertAlmostEqual(self.app._progress.get(), 0.98)
            self.assertEqual(self.app._status_label.cget("text"), "Remuxing audio...")

            self.app._handle_encode_line("Done in 1.2s -> " + out_mp4 + "\n", out_mp4)
            self.assertAlmostEqual(self.app._progress.get(), 1.0)
            self.assertEqual(self.app._status_label.cget("text"), "Video saved ✓")

    def test_handle_encode_line_hides_structured_events(self):
        with tempfile.TemporaryDirectory() as d:
            self._ready_job(d)
            out_mp4 = os.path.join(d, "export.mp4")
            self.app._tracker = ProgressTracker()
            event_line = format_event(EV_FRAME, current=2, total=4)
            self.app._handle_encode_line(event_line, out_mp4)
            self.assertAlmostEqual(self.app._progress.get(), 0.5)
            log_text = self.app._log.get("1.0", "end")
            self.assertNotIn(EVENT_PREFIX, log_text)

    def test_finish_save_success_and_failure_progress_and_playback_state(self):
        with tempfile.TemporaryDirectory() as d:
            self._ready_job(d)
            out_mp4 = os.path.join(d, "export.mp4")
            with open(out_mp4, "wb") as fh:
                fh.write(b"done")

            # Successful exports are previewed in-window and can be revealed.
            self.app._video_save_in_progress = True
            self.app._finish_video_encode(0, "", out_mp4, 1)
            self._settle_video_loader()
            self.assertEqual(self.app._progress.get(), 1.0)
            self.assertEqual(self.app._status_label.cget("text"), "Video saved ✓")
            self.assertNotEqual(self.app._open_dir_btn.grid_info(), {})
            self.assertEqual(self.app._output_path, out_mp4)

            # Failure test
            self.app._video_save_in_progress = True
            with mock.patch.object(gui.messagebox, "showerror"):
                self.app._finish_video_encode(1, "error", out_mp4, 1)
            self.assertEqual(self.app._progress.get(), 0.0)
            self.assertEqual(self.app._status_label.cget("text"), "Export failed")
            self.assertIsNone(self.app._output_path)

    def test_create_zoom_job_carries_frames_dir(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as d:
            img = os.path.join(d, "in.jpg")
            Image.new("RGB", (64, 48)).save(img)
            self.app._image_path.set(img)
            self.app._out_dir.set(os.path.join(d, "out"))
            self.app._on_mode_change("Dream Zoom")
            job = self.app._create_job_from_inputs()
            self.assertIsNotNone(job)
            self.assertEqual(job.mode, "Dream Zoom")
            self.assertTrue(job.frames_dir)
            cmd = job.build_cmd()
            self.assertIn("--frames-only", cmd)
            self.assertIn(job.frames_dir, cmd)

    def test_view_job_output_reviews_unexported_frames(self):
        with tempfile.TemporaryDirectory() as d:
            fd = self._make_frames(d)
            job = GenerationJob(
                id=7,
                image_path="sample.jpg",
                model_key="googlenet",
                layer_spec="inception4c",
                iterations=2,
                octaves=0,
                octave_scale=1.4,
                lr=0.03,
                size=512,
                jitter=32,
                out_dir=d,
                mode="Dream Zoom",
                frames_dir=fd,
                status="Done",
            )
            self.app._view_job_output(job)
            self.assertEqual(self.app._video_frames_job_id, 7)
            self.assertEqual(self.app._save_video_btn.cget("state"), "normal")
            self._settle_video_loader()

    def test_default_save_name_includes_model_resolution_and_layers(self):
        with tempfile.TemporaryDirectory() as d:
            job, _ = self._ready_job(d)
            default_name = self.app._default_video_save_name(out_dir=d)
            self.assertIn("512px", default_name)
            self.assertIn("googlenet", default_name)
            self.assertIn("inception4c", default_name)
            self.assertIn("zoom", default_name)
            self.assertIn("crf", default_name)
            self.assertTrue(default_name.endswith(".mp4"))

    def test_default_save_name_conflict_resolution(self):
        with tempfile.TemporaryDirectory() as d:
            job, _ = self._ready_job(d)
            name1 = self.app._default_video_save_name(out_dir=d)
            # Create the file on disk to simulate a previously saved video
            (Path(d) / name1).write_bytes(b"mp4")
            # Next save name should auto-increment to prevent naming conflict
            name2 = self.app._default_video_save_name(out_dir=d)
            self.assertNotEqual(name1, name2)
            self.assertEqual(name2, f"{Path(name1).stem}_1.mp4")

    def test_multiple_jobs_same_image_get_distinct_names(self):
        with tempfile.TemporaryDirectory() as d:
            # Job 1 with GoogLeNet / inception4c
            job1, fd1 = self._ready_job(d)
            name1 = self.app._default_video_save_name(out_dir=d)

            # Job 2 with VGG19 / features.26 on the same input image
            fd2 = self._make_frames(d, mode="zoom", job_id=2)
            job2 = GenerationJob(
                id=2,
                image_path="sample.jpg",
                model_key="vgg19",
                layer_spec="features.26",
                iterations=3,
                octaves=0,
                octave_scale=1.4,
                lr=0.018,
                size=1024,
                jitter=32,
                out_dir=d,
                mode="Dream Zoom",
                frames_dir=fd2,
                status="Running",
            )
            self.app._job_queue.append(job2)
            self.app._load_video_frames_ready(job2, fd2)
            self._settle_video_loader()
            name2 = self.app._default_video_save_name(out_dir=d)

            self.assertIn("googlenet", name1)
            self.assertIn("512px", name1)
            self.assertIn("vgg19", name2)
            self.assertIn("1024px", name2)
            self.assertNotEqual(name1, name2)

    def test_default_save_name_from_meta_json_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            # When job is not in memory, meta.json in frames_dir is used
            fd = os.path.join(d, "frames_orphan")
            os.makedirs(fd)
            from PIL import Image

            Image.new("RGB", (64, 48)).save(os.path.join(fd, "frame_000001.png"))
            write_video_frames_meta(
                fd,
                fps=30,
                mode="zoom",
                source="mountain.png",
                width=64,
                height=48,
                model="vgg16",
                layers=["features.22"],
                size=768,
                iterations=4,
            )
            self.app._video_frames_dir = fd
            self.app._video_frames_job = None
            self.app._video_frames_job_id = 999  # not in queue
            name = self.app._default_video_save_name(out_dir=d)
            self.assertIn("mountain", name)
            self.assertIn("768px", name)
            self.assertIn("vgg16", name)
            self.assertIn("features-22", name)
            self.assertIn("i4", name)

    def test_save_video_button_in_main_actions_and_synced_with_sidebar(self):
        # Save video button is positioned next to Run/Dream button in main actions
        run_master = self.app._run_btn.master
        self.assertIs(self.app._save_video_btn.master, run_master)
        run_info = self.app._run_btn.grid_info()
        save_info = self.app._save_video_btn.grid_info()
        self.assertEqual(int(run_info["row"]), int(save_info["row"]))
        self.assertEqual(int(run_info["column"]), 0)
        self.assertEqual(int(save_info["column"]), 2)

        # Sidebar video save button is in video_quality_frame
        self.assertIsNotNone(self.app._sidebar_save_video_btn)
        self.assertIs(self.app._sidebar_save_video_btn.master, self.app._video_quality_frame)

        # Both start disabled when no frames are ready
        self.assertEqual(self.app._save_video_btn.cget("state"), "disabled")
        self.assertEqual(self.app._sidebar_save_video_btn.cget("state"), "disabled")

        # When frames are ready, both are enabled and show "Save Video"
        with tempfile.TemporaryDirectory() as d:
            self._ready_job(d)
            self.assertEqual(self.app._save_video_btn.cget("state"), "normal")
            self.assertEqual(self.app._sidebar_save_video_btn.cget("state"), "normal")
            self.assertEqual(self.app._save_video_btn.cget("text"), "Save Video")
            self.assertEqual(self.app._sidebar_save_video_btn.cget("text"), "Save Video")

            # In progress state: both disabled and show "Encoding..."
            self.app._video_save_in_progress = True
            self.app._refresh_video_save_ui()
            self.assertEqual(self.app._save_video_btn.cget("state"), "disabled")
            self.assertEqual(self.app._sidebar_save_video_btn.cget("state"), "disabled")
            self.assertEqual(self.app._save_video_btn.cget("text"), "Encoding...")
            self.assertEqual(self.app._sidebar_save_video_btn.cget("text"), "Encoding...")

            # Finish: both re-enabled and show "Save Video"
            out_mp4 = os.path.join(d, "out.mp4")
            with open(out_mp4, "wb") as fh:
                fh.write(b"x" * 64)
            with mock.patch.object(self.app, "_setup_mp4_preview", return_value=True):
                self.app._finish_video_encode(0, "Done in 1s\n", out_mp4, 1)
            self.assertEqual(self.app._save_video_btn.cget("state"), "normal")
            self.assertEqual(self.app._sidebar_save_video_btn.cget("state"), "normal")
            self.assertEqual(self.app._save_video_btn.cget("text"), "Save Video")
            self.assertEqual(self.app._sidebar_save_video_btn.cget("text"), "Save Video")
            self._settle_video_loader()


class TestSystematicVideoNaming(unittest.TestCase):
    def test_build_systematic_video_name_full(self):
        name = build_systematic_video_name(
            source="photo.jpg",
            size=1024,
            model="googlenet",
            layers="inception4c",
            iterations=2,
            mode="zoom",
            crf=17,
        )
        self.assertEqual(name, "photo_1024px_googlenet_inception4c_i2_zoom_crf17.mp4")

    def test_build_systematic_video_name_with_custom_name(self):
        name = build_systematic_video_name(
            source="photo.jpg",
            custom_name="trip_dream",
            size=512,
            model="vgg19",
            layers="features.26",
            iterations=3,
            mode="zoom",
            crf=17,
        )
        self.assertEqual(name, "trip_dream_512px_vgg19_features-26_i3_zoom_crf17.mp4")

    def test_build_systematic_video_name_dimensions_fallback(self):
        name = build_systematic_video_name(
            source="clip.mp4",
            size=0,
            width=1280,
            height=720,
            model="vgg16",
            layers=["features.22"],
            mode="video",
            crf=28,
        )
        self.assertEqual(name, "clip_1280x720_vgg16_features-22_video_crf28.mp4")

    def test_build_systematic_video_name_target_class(self):
        name = build_systematic_video_name(
            source="dog.png",
            size=512,
            model="inceptionv3",
            target_class="goldfish",
            mode="zoom",
            crf=17,
        )
        self.assertEqual(name, "dog_512px_inceptionv3_class-goldfish_zoom_crf17.mp4")

    def test_build_systematic_video_name_guide_image(self):
        name = build_systematic_video_name(
            source="input.png",
            size=512,
            model="googlenet",
            layers="inception4c",
            guide="path/to/starry_night.jpg",
            mode="zoom",
            crf=17,
        )
        self.assertEqual(
            name, "input_512px_googlenet_inception4c_guide-starry_night_zoom_crf17.mp4"
        )

    def test_build_systematic_video_name_sanitizes_invalid_chars(self):
        name = build_systematic_video_name(
            source="invalid:file*name?.jpg",
            size=512,
            model="googlenet",
            mode="zoom",
            crf=17,
        )
        self.assertNotIn(":", name)
        self.assertNotIn("*", name)
        self.assertNotIn("?", name)
        self.assertTrue(name.endswith(".mp4"))

    def test_build_systematic_video_name_caps_length(self):
        long_stem = "a" * 200
        name = build_systematic_video_name(
            source=long_stem + ".png",
            size=1024,
            model="googlenet",
            layers="inception4c",
            mode="zoom",
            crf=17,
        )
        stem = Path(name).stem
        self.assertLessEqual(len(stem), MAX_NAME_LEN)

    def test_resolve_name_conflict_empty_or_nonexistent_dir(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(resolve_name_conflict(d, "clip.mp4"), "clip.mp4")
            self.assertEqual(
                resolve_name_conflict(os.path.join(d, "nonexistent"), "clip.mp4"),
                "clip.mp4",
            )

    def test_resolve_name_conflict_increments_when_exists(self):
        with tempfile.TemporaryDirectory() as d:
            f0 = Path(d) / "clip.mp4"
            f0.write_bytes(b"0")
            self.assertEqual(resolve_name_conflict(d, "clip.mp4"), "clip_1.mp4")

            f1 = Path(d) / "clip_1.mp4"
            f1.write_bytes(b"1")
            self.assertEqual(resolve_name_conflict(d, "clip.mp4"), "clip_2.mp4")
            # Passing clip_1 when clip_1 exists rolls to clip_2
            self.assertEqual(resolve_name_conflict(d, "clip_1.mp4"), "clip_2.mp4")

    def test_resolve_name_conflict_case_insensitive(self):
        with tempfile.TemporaryDirectory() as d:
            f0 = Path(d) / "CLIP.MP4"
            f0.write_bytes(b"0")
            self.assertEqual(resolve_name_conflict(d, "clip.mp4"), "clip_1.mp4")


class TestCaseInsensitiveMedia(unittest.TestCase):
    def test_newest_png_finds_uppercase(self):
        with tempfile.TemporaryDirectory() as d:
            upper = Path(d) / "PHOTO.PNG"
            upper.write_bytes(b"png")
            os.utime(upper, (2000, 2000))
            self.assertEqual(newest_png(d), str(upper))

    def test_newest_media_finds_uppercase(self):
        with tempfile.TemporaryDirectory() as d:
            video = Path(d) / "CLIP.MP4"
            photo = Path(d) / "photo.JPG"
            video.write_bytes(b"mp4")
            photo.write_bytes(b"jpg")
            os.utime(photo, (1000, 1000))
            os.utime(video, (2000, 2000))
            self.assertEqual(newest_media(d), str(video))

    def test_newest_media_ignores_directories_named_like_media(self):
        # A directory named "x.png" must not be returned as the newest image.
        with tempfile.TemporaryDirectory() as d:
            sub = Path(d) / "x.png"
            sub.mkdir()
            real = Path(d) / "real.png"
            real.write_bytes(b"png")
            os.utime(sub, (3000, 3000))
            os.utime(real, (2000, 2000))
            self.assertEqual(newest_media(d), str(real))

    def test_dialog_pattern_matches_any_case(self):
        import fnmatch

        for ext in MEDIA_EXTENSIONS:
            pat = _dialog_pattern_for_ext(ext)
            stem = "photo" + ext
            self.assertTrue(fnmatch.fnmatch(stem, pat), f"{pat} should match {stem}")
            self.assertTrue(
                fnmatch.fnmatch(stem.upper(), pat),
                f"{pat} should match {stem.upper()}",
            )


class TestIntegratedViewerHelpers(unittest.TestCase):
    def test_clamp_zoom(self):
        self.assertEqual(clamp_zoom(1.0), 1.0)
        self.assertEqual(clamp_zoom(100.0), ZOOM_MAX)
        self.assertEqual(clamp_zoom(0.0), ZOOM_MIN)
        self.assertEqual(clamp_zoom("bad"), 1.0)
        self.assertEqual(clamp_zoom(float("nan")), 1.0)

    def test_fit_zoom_for(self):
        self.assertAlmostEqual(fit_zoom_for(200, 100, 100, 100), 0.5)
        self.assertAlmostEqual(fit_zoom_for(100, 100, 400, 300), 3.0)
        self.assertEqual(fit_zoom_for(0, 100, 100, 100), 1.0)

    def test_compute_zoom_view_fit_and_zoomed(self):
        v = compute_zoom_view(200, 100, 100, 100, 0.5, 100, 50)
        self.assertEqual(v["crop"], (0, 0, 200, 100))
        self.assertEqual(v["resample"], "lanczos")
        v2 = compute_zoom_view(200, 200, 100, 100, 4.0, 100, 100)
        self.assertEqual(v2["resample"], "nearest")
        left, top, right, bottom = v2["crop"]
        self.assertGreaterEqual(left, 0)
        self.assertGreaterEqual(top, 0)
        self.assertLessEqual(right, 200)
        self.assertLessEqual(bottom, 200)
        self.assertLess(right - left, 200)

    def test_render_zoomed_preview_exact_viewport(self):
        from PIL import Image

        src = Image.new("RGB", (64, 48), color=(10, 200, 30))
        out = render_zoomed_preview(src, 100, 80, 1.0, 32, 24)
        self.assertEqual(out.size, (100, 80))
        out2 = render_zoomed_preview(src, 100, 80, 4.0, 32, 24)
        self.assertEqual(out2.size, (100, 80))

    def test_timeline_mapping_roundtrip(self):
        self.assertEqual(timeline_t_to_index(0.0, 60), 0)
        self.assertEqual(timeline_t_to_index(1.0, 60), 59)
        self.assertEqual(timeline_index_to_t(0, 60), 0.0)
        self.assertEqual(timeline_index_to_t(59, 60), 1.0)
        idx = timeline_t_to_index(0.5, 5)
        self.assertTrue(0 <= idx < 5)
        self.assertIn("3/60", format_frame_label(2, 60, 30))
        self.assertIn("30 fps", format_frame_label(2, 60, 30))
        self.assertEqual(format_frame_label(0, 0), "No frames")

    def test_build_reveal_command(self):
        win = build_reveal_command("C:\\out\\dream.png", platform="win32")
        assert win is not None
        self.assertEqual(win[0], "explorer")
        self.assertTrue(any("select" in c.lower() for c in win))
        mac = build_reveal_command("/tmp/dream.png", platform="darwin")
        assert mac is not None
        self.assertEqual(mac[:2], ["open", "-R"])
        self.assertIsNone(build_reveal_command("/tmp/dream.png", platform="linux"))


@pytest.mark.gui
class TestIntegratedViewerGUI(unittest.TestCase):
    def setUp(self):
        try:
            self.app = gui.DeepDreamGUI()
        except Exception as exc:
            if "no display name" in str(exc).lower() or "tcl" in str(exc).lower():
                self.skipTest("Tk display not available")
            raise

    def tearDown(self):
        if hasattr(self, "app"):
            try:
                self.app._stop_video_playback()
            except Exception:
                pass
            loader = getattr(self.app, "_video_loader", None)
            if loader is not None:
                loader.wait_idle()
                loader.close()
            self.app.destroy()

    def test_zoom_controls_resample_and_label(self):
        from PIL import Image

        self.app._dream_pil = Image.new("RGB", (200, 200), color=(255, 0, 0))
        self.app._aligned_input_pil = None
        self.app._compare_mode = "Dream"
        self.app._reset_viewer()
        self.app._show_output_preview(self.app._dream_pil)
        self.assertEqual(self.app._zoom_label.cget("text"), "Fit")
        self.app._on_zoom_100()
        self.assertIn("100%", self.app._zoom_label.cget("text"))
        self.assertFalse(self.app._viewer_fit)
        self.app._on_zoom_in()
        self.assertGreater(self.app._viewer_zoom, 1.0)
        self.app._on_zoom_fit()
        self.assertTrue(self.app._viewer_fit)

    def test_show_in_folder_highlights_file(self):
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            target = f.name
        try:
            self.app._output_path = target
            self.app._viewer_mode = "image"
            self.assertEqual(self.app._reveal_target(), target)
            with mock.patch("app.gui.subprocess.Popen") as mp:
                with mock.patch.object(gui, "sys") as _:
                    # Force the win32 branch without touching the real platform.
                    cmd = build_reveal_command(target, platform="win32")
                    self.assertIsNotNone(cmd)
                    mp.assert_not_called()
        finally:
            if os.path.isfile(target):
                os.remove(target)

    def test_reaim_shows_input_reticle_over_still_result(self):
        # Regression: "Show Input to Aim" must show the input-with-reticle even
        # when a still dream result is already present, instead of the dream.
        from PIL import Image

        self.app._input_pil = Image.new("RGB", (120, 90), color=(0, 120, 0))
        self.app._preview_pil = self.app._input_pil
        self.app._dream_pil = Image.new("RGB", (120, 90), color=(200, 0, 0))
        self.app._compare_mode = "Dream"
        self.app._viewer_mode = "image"
        self.app._mode_var.set("Dream Zoom")
        self.app._showing_zoom_focus = False

        self.app._show_zoom_focus_for_aiming()
        self.assertTrue(self.app._showing_zoom_focus)
        src = self.app._get_viewer_source()
        # Corner (away from the reticle marker) is the green input, not the
        # red dream result.
        self.assertEqual(src.getpixel((5, 5)), (0, 120, 0))

        self.app._showing_zoom_focus = False
        self.assertEqual(self.app._get_viewer_source().getpixel((5, 5)), (200, 0, 0))

    def test_video_cache_miss_returns_none_not_unrelated_still(self):
        # A resize before the requested frame decodes must leave the previous
        # frame on screen, not swap in the input photo or an older dream result.
        from PIL import Image

        self.app._preview_pil = Image.new("RGB", (8, 8), color=(0, 120, 0))
        self.app._dream_pil = Image.new("RGB", (8, 8), color=(200, 0, 0))
        self.app._viewer_mode = "video"
        self.app._video_kind = "mp4"
        self.app._video_mp4 = "missing.mp4"
        self.app._video_index = 0
        self.assertIsNone(self.app._get_viewer_source())

    def test_click_jitter_does_not_leave_fit(self):
        # Regression: sub-threshold motion during a click must not flip the
        # viewer out of "Fit" (Tk emits such jitter mid-click).
        import types

        from PIL import Image

        self.app._dream_pil = Image.new("RGB", (200, 200), color=(1, 2, 3))
        self.app._aligned_input_pil = None
        self.app._compare_mode = "Dream"
        self.app._viewer_mode = "image"
        self.app._reset_viewer()
        self.app._show_output_preview(self.app._dream_pil)
        self.assertTrue(self.app._viewer_fit)

        self.app._on_viewer_press(types.SimpleNamespace(x=100, y=100))
        self.app._on_viewer_drag(types.SimpleNamespace(x=101, y=101))
        self.assertTrue(self.app._viewer_fit)

        self.app._on_viewer_drag(types.SimpleNamespace(x=120, y=120))
        self.assertFalse(self.app._viewer_fit)

    def test_timelapse_hides_compare_controls(self):
        # Compare controls only apply to stills: hidden during timelapse,
        # restored when the still is shown again.
        from PIL import Image

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            mp4 = f.name
        try:
            self.app._aligned_input_pil = Image.new("RGB", (32, 32), color=(9, 9, 9))
            self.app._dream_pil = Image.new("RGB", (32, 32), color=(1, 1, 1))
            self.app._compare_mode = "Dream"
            self.app._viewer_mode = "image"
            self.app._timelapse_mp4 = mp4
            self.app._compare_frame.grid()

            # The toggle now probes and decodes off the Tk thread; only the
            # viewer state and compare-control visibility are set inline.
            with mock.patch.object(self.app, "_video_loader", mock.Mock()):
                self.app._on_view_timelapse_toggle()
            self.assertEqual(self.app._viewer_mode, "timelapse")
            self.assertEqual(self.app._video_kind, "mp4")
            self.assertEqual(self.app._compare_frame.grid_info(), {})

            self.app._on_view_image_toggle()
            self.assertEqual(self.app._viewer_mode, "image")
            self.assertNotEqual(self.app._compare_frame.grid_info(), {})
        finally:
            if os.path.isfile(mp4):
                os.remove(mp4)

    def test_view_size_matches_label_when_mapped(self):
        # No wasted pixels: once laid out, the view renders at the label's
        # exact size so the image fills the available space.
        from PIL import Image

        self.app._dream_pil = Image.new("RGB", (400, 300), color=(5, 5, 5))
        self.app._aligned_input_pil = None
        self.app._compare_mode = "Dream"
        self.app._viewer_mode = "image"
        self.app._reset_viewer()
        self.app._show_output_preview(self.app._dream_pil)
        self.app.update_idletasks()
        lw = self.app._preview_label.winfo_width()
        lh = self.app._preview_label.winfo_height()
        if lw > 2 and lh > 2:
            self.assertEqual(self.app._get_view_size(), (lw, lh))

    def test_frames_preview_sets_timeline(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as d:
            for i in range(1, 4):
                Image.new("RGB", (32, 24), color=(i * 40, 10, 10)).save(
                    os.path.join(d, f"frame_{i:06d}.png")
                )
            ok = self.app._setup_frames_preview(d, fps=24.0, title="Frames Ready")
            self.assertTrue(ok)
            self.assertEqual(self.app._video_total, 3)
            self.assertEqual(self.app._video_kind, "frames")
            self.assertNotEqual(self.app._video_bar.grid_info(), {})
            self.assertIn("Frame", self.app._video_pos_label.cget("text"))
            self.app._on_video_slider_change(1.0)
            self.assertEqual(self.app._video_index, 2)
            self.app._teardown_video()
            self.assertIsNone(self.app._video_kind if self.app._video_kind else None)

    def test_slide_reveal_drag_updates_split_ratio(self):
        from PIL import Image

        self.app._dream_pil = Image.new("RGB", (200, 200), color=(255, 0, 0))
        self.app._aligned_input_pil = Image.new("RGB", (200, 200), color=(0, 255, 0))
        self.app._set_compare_mode("Slide Reveal")
        self.assertEqual(self.app._compare_mode, "Slide Reveal")

        # Initial ratio is 0.5
        self.assertAlmostEqual(self.app._reveal_split, 0.5, places=2)

        # Start dragging at center
        line_x = self.app._split_ratio_to_label_x(0.5)
        self.assertIsNotNone(line_x)
        self.app._on_viewer_press(mock.MagicMock(x=line_x, y=50))
        self.assertTrue(self.app._reveal_dragging)

        # Drag to 25% position
        target_ratio = 0.25
        target_x = self.app._split_ratio_to_label_x(target_ratio)
        self.assertIsNotNone(target_x)
        self.app._on_viewer_drag(mock.MagicMock(x=target_x, y=50))
        self.assertAlmostEqual(self.app._reveal_split, target_ratio, places=2)
        self.assertIn("25%", self.app._preview_title.cget("text"))

        # Release drag
        self.app._on_viewer_release(mock.MagicMock(x=target_x, y=50))
        self.assertFalse(self.app._reveal_dragging)

    def test_slide_reveal_hover_cursor(self):
        from PIL import Image

        self.app._dream_pil = Image.new("RGB", (200, 200), color=(255, 0, 0))
        self.app._aligned_input_pil = Image.new("RGB", (200, 200), color=(0, 255, 0))
        self.app._set_compare_mode("Slide Reveal")

        line_x = self.app._split_ratio_to_label_x(0.5)
        self.assertIsNotNone(line_x)

        # Hover near the line
        self.app._on_viewer_motion(mock.MagicMock(x=line_x, y=50))
        self.assertEqual(self.app._preview_label.cget("cursor"), "sb_h_double_arrow")

        # Hover far from the line
        self.app._on_viewer_motion(mock.MagicMock(x=0, y=50))
        self.assertEqual(self.app._preview_label.cget("cursor"), "")

        # Leave resets cursor
        self.app._on_viewer_leave(mock.MagicMock())
        self.assertEqual(self.app._preview_label.cget("cursor"), "")

    def test_show_in_folder_windows_selects_resolved_path(self):
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=" image, test.png", delete=False) as f:
            target = f.name
        try:
            self.app._output_path = target
            self.app._viewer_mode = "image"
            resolved = self.app._reveal_target()
            self.assertEqual(resolved, os.path.abspath(target))

            with mock.patch("app.gui.subprocess.Popen") as mp:
                with mock.patch.object(gui.sys, "platform", "win32"):
                    self.app._show_in_folder()
                    mp.assert_called_once()
                    # build_reveal_command returns argv as a list, so a path
                    # with spaces or commas reaches explorer as one argument.
                    cmd = mp.call_args[0][0]
                    self.assertEqual(cmd[0], "explorer")
                    self.assertTrue(cmd[1].startswith("/select,"))
                    self.assertIn(os.path.abspath(target), cmd[1])
        finally:
            if os.path.isfile(target):
                os.remove(target)

    def test_combobox_scroll_wheel_disabled_and_scrollable_frame_works(self):
        import tkinter as tk

        from app.gui_widgets import ModernOptionMenu, ModernScrollableFrame

        cb = ModernOptionMenu(self.app, values=["Model A", "Model B", "Model C"])
        cb.set("Model A")

        # Scroll wheel on combobox directly does not cycle selection
        cb.event_generate("<MouseWheel>", delta=120)
        self.app.update()
        self.assertEqual(cb.get(), "Model A")

        cb.event_generate("<MouseWheel>", delta=-120)
        self.app.update()
        self.assertEqual(cb.get(), "Model A")
        cb.destroy()

        # In ModernScrollableFrame, combobox is traversed and does not block page scroll
        top = tk.Toplevel(self.app)
        try:
            sf = ModernScrollableFrame(top, height=100)
            sf.pack()
            cb2 = ModernOptionMenu(sf.inner, values=["Option 1", "Option 2"])
            cb2.set("Option 1")
            cb2.pack()
            top.update()
            sf._bind_wheel()
            cb2.event_generate("<MouseWheel>", delta=-120)
            top.update()
            self.assertEqual(cb2.get(), "Option 1")
        finally:
            top.destroy()

    def test_failed_video_job_with_saved_frames_recovers_as_stopped(self):
        from PIL import Image

        from app.gui_job import GenerationJob
        from app.model_data import write_video_frames_meta

        with tempfile.TemporaryDirectory() as td:
            frames_dir = os.path.join(td, "seq_failed")
            os.makedirs(frames_dir, exist_ok=True)
            for i in range(1, 4):
                Image.new("RGB", (32, 24), color=(i * 30, 20, 20)).save(
                    os.path.join(frames_dir, f"frame_{i:06d}.png")
                )
            write_video_frames_meta(
                frames_dir, fps=24.0, mode="zoom", source="test.png", total_frames=10
            )

            job = GenerationJob(
                id=100,
                image_path="test.png",
                model_key="googlenet",
                layer_spec="inception4c",
                iterations=10,
                octaves=0,
                octave_scale=1.4,
                lr=0.01,
                size=64,
                jitter=0,
                out_dir=td,
                mode="Dream Zoom",
                status="Running",
                frames_dir=frames_dir,
                zoom_frames=10,
            )

            self.app._current_job = job
            self.app._job_queue = [job]
            self.app._frames_dir = frames_dir
            self.app._aborted = False
            mock_proc = mock.MagicMock()
            mock_proc.returncode = 1
            mock_proc.poll.return_value = 1
            self.app._proc = mock_proc

            self.app._on_process_finished()

            # A crashed job with readable frames recovers as Stopped, not Failed
            self.assertEqual(job.status, "Stopped")
            self.assertEqual(self.app._video_frames_dir, frames_dir)
            self.assertIn("Paused", self.app._preview_title.cget("text"))

    def test_stopped_video_job_preserves_frames_and_enables_save_and_resume(self):
        from PIL import Image

        from app.gui_job import GenerationJob
        from app.model_data import write_video_frames_meta

        with tempfile.TemporaryDirectory() as td:
            frames_dir = os.path.join(td, "seq_stopped")
            os.makedirs(frames_dir, exist_ok=True)
            for i in range(1, 4):
                Image.new("RGB", (32, 24), color=(i * 30, 20, 20)).save(
                    os.path.join(frames_dir, f"frame_{i:06d}.png")
                )
            write_video_frames_meta(
                frames_dir,
                fps=24.0,
                mode="zoom",
                source="test.png",
                width=32,
                height=24,
                model="googlenet",
                total_frames=10,
                seed=42,
            )

            job = GenerationJob(
                id=99,
                image_path="test.png",
                model_key="googlenet",
                layer_spec="inception4c",
                iterations=10,
                octaves=0,
                octave_scale=1.4,
                lr=0.01,
                size=64,
                jitter=0,
                out_dir=td,
                mode="Dream Zoom",
                status="Running",
                frames_dir=frames_dir,
                zoom_frames=10,
            )

            self.app._current_job = job
            self.app._job_queue = [job]
            self.app._frames_dir = frames_dir
            self.app._aborted = True
            mock_proc = mock.MagicMock()
            mock_proc.returncode = 1
            mock_proc.poll.return_value = 1
            self.app._proc = mock_proc

            self.app._on_process_finished()

            # Frames must be preserved on disk!
            self.assertEqual(len(os.listdir(frames_dir)), 4)  # 3 frames + meta.json
            self.assertEqual(job.status, "Stopped")

            # Video export state must be active and Save Video enabled
            self.assertEqual(self.app._video_frames_dir, frames_dir)
            self.assertEqual(self.app._save_video_btn.cget("state"), "normal")
            self.assertIn("Save Video", self.app._save_video_btn.cget("text"))

            # Dream button must ALWAYS remain "▶ Dream" (never replaced or renamed)
            self.assertEqual(self.app._run_btn.cget("text"), "▶ Dream")

            # Dedicated Resume button appears next to Dream button
            self.assertEqual(self.app._resumable_job, job)
            self.assertNotEqual(self.app._main_resume_btn.grid_info(), {})
            self.assertEqual(self.app._main_resume_btn.cget("text"), "▶ Resume")
            self.assertEqual(self.app._main_resume_btn.cget("state"), "normal")

            # Queue card reflects stopped status with frame count and Resume button
            self.app._refresh_queue_ui()
            children = list(self.app._queue_inner.winfo_children())
            self.assertTrue(len(children) > 0)
            card = children[0]
            btn_texts = [w.cget("text") for w in card.winfo_children() if hasattr(w, "cget")]
            self.assertIn("Resume", btn_texts)
            self.assertIn("⏹ Stopped (3/10)", btn_texts)

            # Clicking _main_resume_btn triggers resume
            with mock.patch.object(self.app, "_start_queue") as mock_start:
                self.app._proc = None
                self.app._on_main_resume_click()
                self.assertTrue(job.resume)
                self.assertEqual(job.status, "Pending")
                self.assertIsNone(self.app._resumable_job)
                self.assertEqual(self.app._main_resume_btn.grid_info(), {})
                mock_start.assert_called_once()

            # Verify build_cmd includes --resume and preserved seed
            self.assertEqual(job.seed, 42)
            cmd = job.build_cmd()
            self.assertIn("--resume", cmd)
            self.assertIn("--seed", cmd)
            self.assertIn("42", cmd)

            # And verify that pressing Dream starts fresh from frame 1:
            job.status = "Stopped"
            self.app._resumable_job = job
            self.app._update_resume_button_visibility()
            self.assertNotEqual(self.app._main_resume_btn.grid_info(), {})
            with mock.patch.object(self.app, "_create_job_from_inputs") as mock_create:
                with mock.patch.object(self.app, "_add_to_queue"):
                    self.app._run()
                    self.assertIsNone(self.app._resumable_job)
                    self.assertEqual(self.app._main_resume_btn.grid_info(), {})
                    mock_create.assert_called_once()


class TestFontResolution(unittest.TestCase):
    """Windows-only font families must resolve to something on every platform."""

    def setUp(self):
        from app import gui_widgets

        self.widgets = gui_widgets
        self._saved = dict(gui_widgets._resolved_families)
        gui_widgets._resolved_families.clear()

    def tearDown(self):
        self.widgets._resolved_families.clear()
        self.widgets._resolved_families.update(self._saved)

    def test_unknown_family_is_passed_through(self):
        self.assertEqual(self.widgets.resolve_font_family("Some Custom Face"), "Some Custom Face")

    def test_resolves_to_an_available_family_when_request_is_missing(self):
        fake_available = {"Helvetica Neue", "Menlo", "Courier New"}
        with mock.patch.object(self.widgets.tkfont, "families", return_value=list(fake_available)):
            self.assertEqual(self.widgets.resolve_font_family("Segoe UI"), "Helvetica Neue")
            self.assertEqual(self.widgets.resolve_font_family("Consolas"), "Menlo")

    def test_keeps_request_when_it_is_available(self):
        fake_available = {"Segoe UI", "Consolas"}
        with mock.patch.object(self.widgets.tkfont, "families", return_value=list(fake_available)):
            self.assertEqual(self.widgets.resolve_font_family("Segoe UI"), "Segoe UI")
            self.assertEqual(self.widgets.resolve_font_family("Consolas"), "Consolas")

    def test_no_tk_interpreter_is_not_fatal_and_not_cached(self):
        """Before a root exists the answer is unknown, so do not poison the cache."""
        with mock.patch.object(
            self.widgets.tkfont, "families", side_effect=self.widgets.tk.TclError("no display")
        ):
            self.assertEqual(self.widgets.resolve_font_family("Segoe UI"), "Segoe UI")
        self.assertEqual(self.widgets._resolved_families, {})

    def test_modern_font_uses_resolved_family(self):
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        try:
            fake_available = {"Menlo"}
            with mock.patch.object(
                self.widgets.tkfont, "families", return_value=list(fake_available)
            ):
                font = self.widgets.ModernFont(family="Consolas", size=10)
            # cget is what we asked Tk for; actual() would report Tk's own
            # substitution if the stand-in is not installed on this machine.
            self.assertEqual(font.cget("family"), "Menlo")
        finally:
            root.destroy()


@pytest.mark.gui
class TestOptionDbFontDefaults(unittest.TestCase):
    """option_add font defaults must be Tcl-list-safe on every platform.

    Regression test: an unbraced "DejaVu Sans 11" default splits into
    [DejaVu, Sans, 11] when Tcl parses it, and "Sans" is read as the
    integer size ("expected integer but got ..."), breaking every widget
    created on that interpreter. Multi-word resolved families only appear
    on machines without Segoe UI, so this passed on some dev boxes while
    failing on CI (Windows Server: "UI", Ubuntu: "Sans", macOS: "Neue").
    """

    def test_option_font_spec_braces_multi_word_family(self):
        from app import gui_widgets

        with mock.patch.object(gui_widgets, "resolve_font_family", return_value="DejaVu Sans"):
            self.assertEqual(gui_widgets.tk_option_font(), "{DejaVu Sans} 11")
        with mock.patch.object(gui_widgets, "resolve_font_family", return_value="Arial"):
            self.assertEqual(gui_widgets.tk_option_font(), "{Arial} 11")

    def test_global_font_default_allows_widget_creation(self):
        import tkinter as tk

        from app import gui_widgets

        root = tk.Tk()
        root.withdraw()
        try:
            gui_widgets._apply_modern_style(root)
            # No explicit font: falls back to the *Font option database default.
            tk.Label(root, text="uses default font")
            # Explicit object font alongside the default.
            tk.Label(root, text="uses object font", font=gui_widgets.ModernFont(size=10))
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
