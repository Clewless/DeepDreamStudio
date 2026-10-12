import argparse
import contextlib
import io
import os
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
import unittest.mock as mock
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
import torchvision.models as models
from PIL import Image

import app.deepdream as deepdream

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_module(args: list[str], **kwargs: Any):
    """Run the CLI as ``python -m app.deepdream`` from the repository root."""
    return subprocess.run(args, cwd=str(REPO_ROOT), **kwargs)


def _weights_cached() -> bool:
    """True if the torchvision weight file is already on disk (no download)."""
    try:
        cached = os.listdir(torch.hub.get_dir())
    except (OSError, AttributeError):
        return False
    return Path(models.GoogLeNet_Weights.IMAGENET1K_V1.url).name in cached


@dataclass
class CliResult:
    returncode: int
    stdout: str
    stderr: str


def run_cli(args: list[str]) -> CliResult:
    buf_out = io.StringIO()
    buf_err = io.StringIO()
    code = 0
    with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
        try:
            deepdream.main(args)
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else (1 if exc.code else 0)
            if isinstance(exc.code, str) and exc.code.strip():
                buf_err.write(exc.code + "\n")
    return CliResult(code, buf_out.getvalue(), buf_err.getvalue())


class TestDeepDream(unittest.TestCase):
    device: str
    model: Any

    @classmethod
    def setUpClass(cls):
        if torch.cuda.is_available():
            cls.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            cls.device = "mps"
        else:
            cls.device = "cpu"
        cls.model = deepdream.build_model("googlenet", cls.device)

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.sample_img_path = os.path.join(self.temp_dir, "test_img.jpg")
        img = Image.new("RGB", (100, 80), color=(128, 64, 200))
        img.save(self.sample_img_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_windows_launcher_quotes_single_dropped_path(self):
        with open(REPO_ROOT / "CLI" / "LaunchCLI.bat", encoding="utf-8") as fh:
            launcher = fh.read()
        self.assertIn('-- "%IMG%"', launcher)
        self.assertNotIn("%*", launcher)

    def test_windows_studio_launcher_resolves_image_and_quotes(self):
        with open(REPO_ROOT / "Launch.bat", encoding="utf-8") as fh:
            launcher = fh.read()
        self.assertIn('set "IMG_ARG=%~f1"', launcher)
        self.assertIn('"%IMG_ARG%"', launcher)
        self.assertNotIn("%*", launcher)

    def test_resolve_layer_valid(self):
        self.assertEqual(deepdream.resolve_layer("inception4c", "googlenet"), "inception4c")
        self.assertEqual(deepdream.resolve_layer("mixed4c", "googlenet"), "inception4c")
        self.assertEqual(deepdream.resolve_layer("Inception4C", "googlenet"), "inception4c")
        self.assertEqual(deepdream.resolve_layer("MIXED5B", "googlenet"), "inception5b")
        self.assertEqual(deepdream.resolve_layer("Mixed_6d", "inceptionv3"), "Mixed_6d")
        self.assertEqual(deepdream.resolve_layer("features.22", "vgg16"), "features.22")

    def test_resolve_layer_invalid(self):
        with self.assertRaises(SystemExit):
            deepdream.resolve_layer("invalid_layer_name", "googlenet")
        with self.assertRaises(SystemExit):
            deepdream.resolve_layer("inception4c", "vgg16")  # wrong model

    def test_model_registry_defaults_structure(self):
        expected_models = {"googlenet", "inceptionv3", "vgg16", "vgg19"}
        self.assertTrue(expected_models.issubset(set(deepdream.MODEL_REGISTRY.keys())))
        for _name, meta in deepdream.MODEL_REGISTRY.items():
            self.assertIn("defaults", meta)
            defaults = meta["defaults"]
            self.assertIn("iterations", defaults)
            self.assertGreater(defaults["iterations"], 0)
            self.assertIn("lr", defaults)
            self.assertGreater(defaults["lr"], 0)
            self.assertIn("octave_scale", defaults)
            self.assertGreater(defaults["octave_scale"], 1.0)
            self.assertIn("octaves", defaults)
            self.assertGreaterEqual(defaults["octaves"], 0)
            self.assertIn("jitter", defaults)
            self.assertGreaterEqual(defaults["jitter"], 0)
            self.assertIn("size", defaults)
            self.assertGreater(defaults["size"], 0)

        # Architecture-calibrated specifics
        self.assertEqual(deepdream.MODEL_REGISTRY["googlenet"]["defaults"]["iterations"], 20)
        self.assertEqual(deepdream.MODEL_REGISTRY["inceptionv3"]["defaults"]["iterations"], 30)
        self.assertEqual(deepdream.MODEL_REGISTRY["vgg16"]["defaults"]["iterations"], 20)
        self.assertEqual(deepdream.MODEL_REGISTRY["vgg19"]["defaults"]["iterations"], 25)

        self.assertAlmostEqual(
            deepdream.MODEL_REGISTRY["googlenet"]["defaults"]["lr"], 0.03, places=3
        )
        self.assertAlmostEqual(
            deepdream.MODEL_REGISTRY["inceptionv3"]["defaults"]["lr"], 0.025, places=3
        )
        self.assertAlmostEqual(deepdream.MODEL_REGISTRY["vgg16"]["defaults"]["lr"], 0.018, places=3)
        self.assertAlmostEqual(deepdream.MODEL_REGISTRY["vgg19"]["defaults"]["lr"], 0.018, places=3)

    def test_plan_octaves_respects_model_min_size(self):
        # InceptionV3 needs at least 75px
        # From 150px at scale 1.4: 150 -> 107 -> 76 -> (54 would drop below 75)
        n = deepdream.plan_octaves(150, 150, 10, 1.4, min_octave_size=75)
        # Starting size 150 (count=1), 107 (count=2), 76 (count=3); 54 < 75 so break -> count=3
        self.assertEqual(n, 3)

    def test_parse_version(self):
        self.assertLess(deepdream._parse_version("1.8.0"), (1, 11, 0))
        self.assertLess(deepdream._parse_version("1.9.1+cpu"), (1, 11, 0))
        self.assertLess(deepdream._parse_version("1.10.2"), (1, 11, 0))
        self.assertEqual(deepdream._parse_version("1.11.0"), (1, 11, 0))
        self.assertGreater(deepdream._parse_version("2.0.1"), (1, 11, 0))
        # Pre-release suffixes drop the non-digit component (1.11.0a0 -> (1, 11));
        # padding to a 3-tuple stops the shorter-tuple-is-less comparison from
        # falsely rejecting a torch >= 1.11 pre-release as too old.
        self.assertEqual(deepdream._parse_version("1.11.0a0"), (1, 11, 0))
        self.assertGreaterEqual(deepdream._parse_version("2.11.0.dev20240301+cu128"), (1, 11, 0))

    def test_load_and_save_image(self):
        tensor = deepdream.load_image(self.sample_img_path, max_size=50)
        self.assertEqual(tensor.dim(), 4)
        self.assertEqual(tensor.shape[1], 3)
        self.assertLessEqual(max(tensor.shape[2:]), 50)

        out_path = os.path.join(self.temp_dir, "saved.png")
        deepdream.save_image(tensor, out_path)
        self.assertTrue(os.path.exists(out_path))
        with Image.open(out_path) as loaded_saved:
            self.assertEqual(loaded_saved.size, (tensor.shape[3], tensor.shape[2]))

    def test_resize_dims(self):
        self.assertIsNone(deepdream.resize_dims((100, 80), 0))
        self.assertIsNone(deepdream.resize_dims((100, 80), 512))
        self.assertEqual(deepdream.resize_dims((100, 80), 50), (50, 40))
        self.assertEqual(deepdream.resize_dims((200, 100), 100), (100, 50))

    def test_save_image_immutability(self):
        tensor = torch.zeros((1, 3, 20, 20), dtype=torch.float32)
        clone_before = tensor.clone()
        out_path = os.path.join(self.temp_dir, "immutable_check.png")
        deepdream.save_image(tensor, out_path)
        self.assertTrue(
            torch.equal(tensor, clone_before),
            "save_image mutated the input tensor in place!",
        )

    def test_save_image_3d_tensor(self):
        tensor = torch.zeros((3, 20, 20), dtype=torch.float32)
        out_path = os.path.join(self.temp_dir, "3d_check.png")
        deepdream.save_image(tensor, out_path)
        self.assertTrue(os.path.exists(out_path))

    def test_save_image_multi_batch_raises(self):
        # A 4D tensor with batch > 1 can't be collapsed to one image silently.
        tensor = torch.zeros((2, 3, 20, 20), dtype=torch.float32)
        out_path = os.path.join(self.temp_dir, "batch_check.png")
        with self.assertRaises(ValueError):
            deepdream.save_image(tensor, out_path)

    def test_run_to_layers_single_and_multiple(self):
        x = torch.randn(1, 3, 128, 128).to(self.device)
        acts = deepdream.run_to_layers(self.model, "googlenet", x, ["inception3a"])
        self.assertIn("inception3a", acts)
        self.assertEqual(len(acts), 1)

        acts_multi = deepdream.run_to_layers(
            self.model, "googlenet", x, ["inception3a", "inception4c"]
        )
        self.assertIn("inception3a", acts_multi)
        self.assertIn("inception4c", acts_multi)
        self.assertEqual(len(acts_multi), 2)

    def test_run_to_layers_with_duplicates(self):
        # Duplicate targets collapse via dict key overwriting; only one entry
        # is returned.
        x = torch.randn(1, 3, 128, 128).to(self.device)
        acts = deepdream.run_to_layers(self.model, "googlenet", x, ["inception3a", "inception3a"])
        self.assertIn("inception3a", acts)
        self.assertEqual(len(acts), 1)

    def test_run_to_layers_applies_transform_input(self):
        # The real GoogLeNet.forward() begins with _transform_input(), which is
        # a plain tensor op (not a child module). _forward_googlenet must apply it
        # when enabled so conv1 sees the [-1, 1] range the weights expect.
        model = torch.nn.Module()
        model.transform_input = True  # type: ignore[assignment]
        model._transform_input = lambda x: x * 2.0  # type: ignore[assignment]
        model.block = torch.nn.Identity()
        x = torch.randn(1, 3, 8, 8)
        acts = deepdream._forward_googlenet(model, x, {"block"})
        self.assertTrue(torch.equal(acts["block"], x * 2.0))

    def test_run_to_layers_no_transform_when_disabled(self):
        # Models without transform_input (or with it off) must run unchanged.
        model = torch.nn.Module()
        model.block = torch.nn.Identity()
        x = torch.randn(1, 3, 8, 8)
        acts = deepdream._forward_googlenet(model, x, {"block"})
        self.assertTrue(torch.equal(acts["block"], x))

    def test_run_to_layers_missing_layer_raises(self):
        x = torch.randn(1, 3, 128, 128).to(self.device)
        with self.assertRaises(RuntimeError):
            deepdream.run_to_layers(self.model, "googlenet", x, ["inception3a", "not_a_layer"])

    def test_run_to_layers_no_match_never_runs_head(self):
        # No target is a model child: the missing-layer error must be raised
        # up front, BEFORE any forward pass, so the classifier head is never
        # invoked at an arbitrary resolution (which would crash or waste work).
        model = torch.nn.Module()
        model.block_a = torch.nn.Identity()
        model.classifier = torch.nn.Module()
        x = torch.randn(1, 3, 8, 8)
        head_mock = mock.Mock(side_effect=AssertionError("classifier must not run"))
        model.classifier.forward = head_mock
        with self.assertRaises(RuntimeError):
            deepdream._forward_googlenet(model, x, {"not_a_layer"})
        head_mock.assert_not_called()

    def test_run_to_layers_skips_post_trunk_children(self):
        # GoogLeNet with aux_logits=True registers aux heads as direct
        # children after the trunk; _forward_googlenet must stop at the deepest
        # target and never execute them (or the classifier head).
        model = torch.nn.Module()
        model.block_a = torch.nn.Identity()
        model.block_b = torch.nn.Identity()
        model.aux_head = torch.nn.Identity()
        model.classifier = torch.nn.Identity()
        x = torch.randn(1, 3, 8, 8)
        aux_mock = mock.Mock(side_effect=AssertionError("aux_head must not run"))
        model.aux_head.forward = aux_mock  # type: ignore[method-assign]
        acts = deepdream._forward_googlenet(model, x, {"block_b"})
        self.assertEqual(list(acts), ["block_b"])
        aux_mock.assert_not_called()

    def test_run_to_layers_skips_aux_between_trunk(self):
        # GoogLeNet's auxiliary heads (when present) sit BETWEEN trunk
        # modules (aux1 between inception4b/inception4c) and are side
        # branches. Their output must not be fed into the next trunk layer:
        # feeding logits forward would crash the trunk with a shape mismatch.
        model = torch.nn.Module()
        model.block_a = torch.nn.Identity()
        model.aux1 = torch.nn.Module()
        model.block_b = torch.nn.Identity()
        x = torch.randn(1, 3, 8, 8)
        aux_mock = mock.Mock(return_value=torch.randn(1, 1000))
        model.aux1.forward = aux_mock
        acts = deepdream._forward_googlenet(model, x, {"block_b"})
        self.assertIn("block_b", acts)
        aux_mock.assert_not_called()

    def test_gradient_ascent_none_grad_raises(self):
        img = torch.zeros(1, 3, 32, 32, device=self.device)
        # fake_acts is a new tensor disconnected from 'img' in the autograd graph,
        # so backward() flows to fake_acts and img.grad remains None.
        fake_acts = {"inception3a": torch.randn(1, 3, 4, 4, requires_grad=True, device=self.device)}
        with mock.patch.object(deepdream, "run_to_layers", return_value=fake_acts):
            with self.assertRaises(RuntimeError):
                deepdream.gradient_ascent(
                    self.model,
                    "googlenet",
                    img,
                    ["inception3a"],
                    lr=1.5,
                    jitter=0,
                )

    def test_gradient_ascent_nan_grad_raises(self):
        # A NaN gradient must fail loudly (RuntimeError) instead of silently
        # corrupting pixels, so an extreme --lr can't bake NaNs into the dream.
        img = torch.zeros(1, 3, 32, 32, device=self.device)

        class NaNGradient(torch.autograd.Function):
            @staticmethod
            def forward(ctx, i):
                return i.clone()

            @staticmethod
            def backward(ctx, grad_output):
                return torch.full_like(grad_output, float("nan"))

        def nan_acts(model, model_key, x, targets):
            return {"inception3a": NaNGradient.apply(x)}

        with mock.patch.object(deepdream, "run_to_layers", nan_acts):
            with self.assertRaises(RuntimeError) as ctx:
                deepdream.gradient_ascent(
                    self.model,
                    "googlenet",
                    img,
                    ["inception3a"],
                    lr=1.5,
                    jitter=0,
                )
        self.assertIn("NaN", str(ctx.exception))

    def test_validate_filename_stem(self):
        deepdream.validate_filename_stem("my_dream")
        deepdream.validate_filename_stem("with space")
        deepdream.validate_filename_stem("1920px")
        for bad in [
            "a/b",
            "a\\b",
            "a:b",
            "a?b",
            "a*b",
            "a|b",
            "a<b",
            "a>b",
            'a"b',
            "CON",
            "con.txt",
            "aux",
            "LPT3",
            "nul",
            "trailing. ",
            "trailing ",
            "bad\tname",
            ".",
            "..",
        ]:
            with self.assertRaises(SystemExit, msg=f"{bad!r} should be rejected"):
                deepdream.validate_filename_stem(bad)

    def test_validate_filename_stem_too_long(self):
        # Must be rejected up front (not when the file save hits MAX_PATH).
        with self.assertRaises(SystemExit):
            deepdream.validate_filename_stem("x" * (deepdream.MAX_NAME_LEN + 1))

    def test_validate_filename_stem_optional_length(self):
        # max_len=None skips the length cap: names derived from an existing file
        # are bounded by the filesystem + MAX_PATH instead, so a long-but-valid
        # basename is not falsely rejected.
        deepdream.validate_filename_stem("x" * (deepdream.MAX_NAME_LEN + 50), max_len=None)
        # The character checks still apply without the cap.
        with self.assertRaises(SystemExit):
            deepdream.validate_filename_stem("bad/name", max_len=None)

    def test_validate_filename_stem_label(self):
        # The `what` label is carried into every error message so an auto-derived
        # output name is not blamed on --name.
        for label in ("--name", "output filename"):
            with self.assertRaises(SystemExit) as ctx:
                deepdream.validate_filename_stem("bad/name", what=label)
            self.assertIn(label, str(ctx.exception))
            self.assertIn("invalid filename character", str(ctx.exception))

    def test_validate_output_path(self):
        deepdream.validate_output_path(os.path.join(self.temp_dir, "ok.png"))
        long_path = os.path.join(self.temp_dir, "x" * (deepdream.MAX_PATH + 10) + ".png")
        with self.assertRaises(SystemExit) as ctx:
            deepdream.validate_output_path(long_path)
        self.assertIn("too long", str(ctx.exception))

    def test_validate_output_path_utf16_length(self):
        # MAX_PATH counts UTF-16 code units, not code points: emoji are 2 units
        # each, so len() under-measures and must not let the path slip through.
        emoji_count = deepdream.MAX_PATH // 2
        path = os.path.join(self.temp_dir, "emoji_" + "\U0001f600" * emoji_count + ".png")
        self.assertLess(
            len(path),
            deepdream.MAX_PATH,
            "precondition: under-counts in code points",
        )
        with self.assertRaises(SystemExit) as ctx:
            deepdream.validate_output_path(path)
        self.assertIn("too long", str(ctx.exception))
        # A short multibyte path is still accepted (no over-rejection).
        short = os.path.join(self.temp_dir, "emoji_\U0001f600.png")
        deepdream.validate_output_path(short)

    def test_validate_output_path_lone_surrogate(self):
        # Lone surrogates (possible on NTFS) must not crash the UTF-16
        # measurement with UnicodeEncodeError.
        path = os.path.join(self.temp_dir, "lone_" + "\ud800" * 60 + ".png")
        base = deepdream._path_len_utf16(path)
        self.assertLess(base, deepdream.MAX_PATH)
        deepdream.validate_output_path(path)
        # Enough emoji must still push the path over MAX_PATH regardless of how
        # long temp_dir happens to be.
        surplus = (deepdream.MAX_PATH - base) // 2 + 5
        with self.assertRaises(SystemExit):
            deepdream.validate_output_path(path + "\U0001f600" * surplus)

    def test_validate_input_size(self):
        # GoogLeNet min is 15px (for deep layers).
        deepdream.validate_input_size(15, 100, "googlenet")
        deepdream.validate_input_size(300, 200, "googlenet")
        # Below threshold: friendly SystemExit.
        for h, w, model_key, needle in [
            (14, 100, "googlenet", "15px"),
            (14, 14, "googlenet", "15px"),
            (74, 100, "inceptionv3", "75px"),
            (31, 31, "vgg16", "32px"),
        ]:
            with self.subTest(h=h, w=w, model_key=model_key):
                with self.assertRaises(SystemExit) as ctx:
                    deepdream.validate_input_size(h, w, model_key)
                self.assertIn(needle, str(ctx.exception))
                self.assertIn("too small", str(ctx.exception))

    def test_deepdream_rejects_too_small_input(self):
        # A library caller feeding a sub-threshold tensor must get a friendly
        # error, not a cryptic conv-shape RuntimeError.
        tensor = torch.randn(1, 3, 6, 6)
        with self.assertRaises(SystemExit) as ctx:
            deepdream.deepdream(
                self.model,
                "googlenet",
                tensor,
                targets=["inception5b"],
                iterations=1,
                lr=1.0,
                octave_scale=1.4,
                num_octaves=0,
                jitter=0,
                device=self.device,
            )
        self.assertIn("too small", str(ctx.exception))

    def test_main_rejects_too_small_size(self):
        # --size that shrinks the image below the trunk minimum must fail with
        # a friendly message BEFORE the model is downloaded/loaded.
        result = run_cli(
            [
                self.sample_img_path,
                "--size",
                "4",
                "--iterations",
                "1",
                "--octaves",
                "1",
            ]
        )
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("too small", result.stderr)
        self.assertIn("Use --size 0", result.stderr)
        self.assertNotIn("Loading ", result.stdout)

    def test_main_too_small_native_message(self):
        # A sub-threshold image at NATIVE resolution (no --size downscale) must
        # say the image itself is too small, not suggest --size 0 -- which is
        # already the default and cannot help.
        small = os.path.join(self.temp_dir, "tiny.jpg")
        Image.new("RGB", (6, 6), color=(0, 0, 0)).save(small)
        result = run_cli(
            [
                small,
                "--iterations",
                "1",
                "--octaves",
                "1",
            ]
        )
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("native resolution", result.stderr)
        self.assertNotIn("Use --size 0", result.stderr)

    def test_main_too_small_native_plus_resize_hint(self):
        # Image 10x10 is below the model minimum (32px for VGG16) at NATIVE
        # resolution, and --size 5 downscales it further. The hint must say the
        # native image is too small -- "--size 0" would still fail, so it must
        # not be suggested.
        small = os.path.join(self.temp_dir, "tiny10.jpg")
        Image.new("RGB", (10, 10), color=(0, 0, 0)).save(small)
        result = run_cli(
            [
                small,
                "--size",
                "5",
                "--iterations",
                "1",
                "--octaves",
                "1",
            ]
        )
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("native resolution", result.stderr)
        self.assertNotIn("Use --size 0", result.stderr)

    def test_ensure_output_dir(self):
        good = os.path.join(self.temp_dir, "nested", "dirs")
        deepdream.ensure_output_dir(good)
        self.assertTrue(os.path.isdir(good))

        # A path that exists as a file must fail with a friendly message.
        file_path = os.path.join(self.temp_dir, "is_a_file")
        with open(file_path, "w") as f:
            f.write("x")
        with self.assertRaises(SystemExit) as ctx:
            deepdream.ensure_output_dir(file_path)
        self.assertIn("not a directory", str(ctx.exception))

    def test_main_rejects_out_that_is_a_file(self):
        # --out pointing at an existing file must fail fast with a friendly
        # message BEFORE the model is downloaded/loaded (not a traceback).
        out_file = os.path.join(self.temp_dir, "out_is_a_file")
        with open(out_file, "w") as f:
            f.write("x")
        result = run_cli(
            [
                self.sample_img_path,
                "--out",
                out_file,
                "--iterations",
                "1",
                "--octaves",
                "1",
            ]
        )
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("not a directory", result.stderr)
        self.assertNotIn("Loading ", result.stdout)

    def test_main_rejects_too_long_frames_path(self):
        # --frames writes <out>/frames/o<i>_i<j>.png, so the length pre-check
        # must use the REAL worst-case frame name: a path that passes for the
        # final PNG but overflows for the frames subpath must fail up front.
        # Lengths are derived from deepdream.MAX_PATH so this holds on every
        # platform (260 on Windows, 1024 on macOS, 4096 on Linux), not just
        # the Windows limit the original 241-char constant was tuned for.
        # Still-image runs use temp_suffix ".p<pid>.tmp" (see main flow) and
        # --name "a" with an empty suffix; the worst frame here is
        # "o2_i1000.png" (100x80 img, --octaves 2, --iterations 1000).
        temp_suffix = f".p{os.getpid()}.tmp"
        limit = deepdream.MAX_PATH
        base_len = len(os.path.abspath(self.temp_dir)) + 1
        out_len = limit - len(temp_suffix) - len("/a.png") - 1
        out_dir = os.path.join(self.temp_dir, "d" * (out_len - base_len))
        final_len = len(os.path.abspath(os.path.join(out_dir, "a.png")) + temp_suffix)
        frames_len = len(
            os.path.abspath(os.path.join(out_dir, "frames", "o2_i1000.png")) + temp_suffix
        )
        self.assertLess(final_len, limit, "precondition: final PNG passes the check")
        self.assertGreaterEqual(frames_len, limit, "precondition: frames path overflows")
        result = run_cli(
            [
                self.sample_img_path,
                "--out",
                out_dir,
                "--name",
                "a",
                "--frames",
                "--iterations",
                "1000",
                "--octaves",
                "2",
            ]
        )
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("too long", result.stderr)
        self.assertNotIn("Loading ", result.stdout)

    def test_main_frames_zero_iterations_no_dir(self):
        # --frames with --iterations 0 writes no frames, so the out\frames dir
        # must not be created (an empty one used to be left behind).
        result = run_cli(
            [
                self.sample_img_path,
                "--frames",
                "--iterations",
                "0",
                "--octaves",
                "1",
                "--out",
                self.temp_dir,
            ]
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(os.path.exists(os.path.join(self.temp_dir, "frames")))

    def test_deepdream_octave_truncation_note(self):
        # A 40x32 image with explicit --octaves 3 can only build one octave
        # (next scaled size would fall below MIN_OCTAVE_SIZE); the user must
        # be told instead of silently running fewer octaves.
        tensor = deepdream.load_image(self.sample_img_path, max_size=40)
        self.assertLessEqual(max(tensor.shape[2:]), 40)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            deepdream.deepdream(
                self.model,
                "googlenet",
                tensor,
                targets=["inception3a"],
                iterations=0,
                lr=1.0,
                octave_scale=1.4,
                num_octaves=3,
                jitter=0,
                device=self.device,
            )
        self.assertIn("running 1 instead", buf.getvalue())

    def test_octave_truncation_note(self):
        # Auto mode and exact fits never warrant a note.
        self.assertIsNone(deepdream.octave_truncation_note(0, 3, 1.4))
        self.assertIsNone(deepdream.octave_truncation_note(3, 3, 1.4))
        # Image too small (MIN_OCTAVE_SIZE): the classic message.
        note = deepdream.octave_truncation_note(3, 1, 1.4)
        self.assertIn("image too small", note)
        self.assertIn("running 1 instead", note)
        # Only the MAX_OCTAVES cap binds: must say "capped", not "too small".
        note = deepdream.octave_truncation_note(
            deepdream.MAX_OCTAVES + 10, deepdream.MAX_OCTAVES, 1.01
        )
        self.assertIn("capped", note)
        self.assertNotIn("too small", note)
        # Cap and image size bind together.
        note = deepdream.octave_truncation_note(deepdream.MAX_OCTAVES + 10, 5, 1.4)
        self.assertIn("exceeds the", note)
        self.assertNotIn("too small", note)

    def test_octave_truncation_note_auto_capped(self):
        # Auto mode near octave-scale 1.0: the MAX_OCTAVES cap stops the pyramid
        # short of the ~250px base octave, so a note must explain it.
        note = deepdream.octave_truncation_note(0, deepdream.MAX_OCTAVES, 1.0001, auto_capped=True)
        self.assertIn("capped", note)
        self.assertIn(f"running {deepdream.MAX_OCTAVES} instead", note)
        # Capped AND the image can't fit the full cap: say both.
        note = deepdream.octave_truncation_note(0, 5, 1.4, auto_capped=True)
        self.assertIn("can't fit", note)
        # Auto without the cap still warrants no note.
        self.assertIsNone(deepdream.octave_truncation_note(0, 3, 1.4, auto_capped=False))

    def test_deepdream_auto_cap_note(self):
        # A near-1.0 octave-scale in auto mode wants far more than MAX_OCTAVES
        # levels to reach the ~250px base, so the cap binds and a note explains
        # it (iterations=0, so the model is never invoked).
        tensor = torch.zeros(1, 3, 256, 256)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            deepdream.deepdream(
                None,
                "googlenet",
                tensor,
                targets=["inception3a"],
                iterations=0,
                lr=1.0,
                octave_scale=1.0001,
                num_octaves=0,
                jitter=0,
                device=self.device,
            )
        self.assertIn("capped", buf.getvalue())
        self.assertIn(f"running {deepdream.MAX_OCTAVES} instead", buf.getvalue())

    def test_plan_octaves(self):
        # Small image, auto mode: single octave.
        self.assertEqual(deepdream.plan_octaves(80, 100, 0, 1.4), 1)
        # Auto mode sizes the base octave near the 250px target.
        self.assertEqual(deepdream.plan_octaves(512, 512, 0, 1.4), 3)
        self.assertEqual(deepdream.plan_octaves(4000, 4000, 0, 1.4), 9)
        # Explicit count that fits runs fully.
        self.assertEqual(deepdream.plan_octaves(400, 300, 3, 1.4), 3)
        # Explicit count truncated by MIN_OCTAVE_SIZE (80 -> 57 -> 41 -> 29 <
        # 32).
        self.assertEqual(deepdream.plan_octaves(80, 100, 4, 1.4), 3)

    def test_plan_octaves_caps_near_one_scale(self):
        # octave_scale just above 1.0 makes the auto formula want ~N/ln(scale)
        # octaves (1.0001 on a 4000px image -> ~27k), which would build a
        # multi-thousand-tensor pyramid and OOM. It must be capped instead.
        self.assertEqual(deepdream.plan_octaves(4000, 4000, 0, 1.01), deepdream.MAX_OCTAVES)
        self.assertEqual(
            deepdream.plan_octaves(4000, 4000, 0, 1.0001),
            deepdream.MAX_OCTAVES,
        )
        self.assertEqual(
            deepdream.plan_octaves(4000, 4000, 0, 1.0000001),
            deepdream.MAX_OCTAVES,
        )
        # Explicit requests are capped too, so a huge --octaves with a near-1.0
        # scale cannot OOM; the "running N instead" note tells the user.
        self.assertEqual(
            deepdream.plan_octaves(4000, 4000, 5000, 1.0001),
            deepdream.MAX_OCTAVES,
        )
        # Normal scales are unaffected (cap never binds).
        self.assertEqual(deepdream.plan_octaves(4000, 4000, 0, 1.4), 9)

    def test_plan_octaves_matches_deepdream(self):
        # plan_octaves must predict exactly how many octaves deepdream runs,
        # so the frames-path pre-check always covers the real worst case.
        tensor = deepdream.load_image(self.sample_img_path, max_size=64)
        n = deepdream.plan_octaves(tensor.shape[2], tensor.shape[3], 4, 1.4)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            deepdream.deepdream(
                self.model,
                "googlenet",
                tensor,
                targets=["inception3a"],
                iterations=0,
                lr=1.0,
                octave_scale=1.4,
                num_octaves=4,
                jitter=0,
                device=self.device,
            )
        octave_lines = [line for line in buf.getvalue().splitlines() if line.startswith("Octave ")]
        self.assertEqual(len(octave_lines), n)

    def test_main_missing_input(self):
        result = run_cli(["definitely_missing_image.jpg"])
        self.assertEqual(result.returncode, 1)
        self.assertIn("Input image not found", result.stderr)

    def test_main_corrupt_input(self):
        # A file that exists but is not a decodable image must fail with a
        # friendly message (not a PIL traceback) before the model is loaded.
        bad_path = os.path.join(self.temp_dir, "corrupt.jpg")
        with open(bad_path, "wb") as f:
            f.write(b"this is not an image")
        result = run_cli([bad_path])
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("Could not load image", result.stderr)

    def test_deepdream_validation(self):
        tensor = deepdream.load_image(self.sample_img_path, max_size=64)
        with self.assertRaises(SystemExit):
            deepdream.deepdream(
                self.model,
                "googlenet",
                tensor,
                targets=[],
                iterations=1,
                lr=1.0,
                octave_scale=1.4,
                num_octaves=2,
                jitter=8,
                device=self.device,
            )

        with self.assertRaises(SystemExit):
            deepdream.deepdream(
                self.model,
                "googlenet",
                tensor,
                targets=["inception3a"],
                iterations=1,
                lr=1.0,
                octave_scale=1.0,
                num_octaves=2,
                jitter=8,
                device=self.device,
            )

        # num_octaves must be >= 0 (0 = auto)
        with self.assertRaises(SystemExit):
            deepdream.deepdream(
                self.model,
                "googlenet",
                tensor,
                targets=["inception3a"],
                iterations=1,
                lr=1.0,
                octave_scale=1.4,
                num_octaves=-1,
                jitter=8,
                device=self.device,
            )

        # iterations must be >= 0
        with self.assertRaises(SystemExit):
            deepdream.deepdream(
                self.model,
                "googlenet",
                tensor,
                targets=["inception3a"],
                iterations=-1,
                lr=1.0,
                octave_scale=1.4,
                num_octaves=2,
                jitter=8,
                device=self.device,
            )

        # lr must be > 0
        with self.assertRaises(SystemExit):
            deepdream.deepdream(
                self.model,
                "googlenet",
                tensor,
                targets=["inception3a"],
                iterations=1,
                lr=0.0,
                octave_scale=1.4,
                num_octaves=2,
                jitter=8,
                device=self.device,
            )

        with self.assertRaises(SystemExit):
            deepdream.deepdream(
                self.model,
                "googlenet",
                tensor,
                targets=["inception3a"],
                iterations=1,
                lr=-1.0,
                octave_scale=1.4,
                num_octaves=2,
                jitter=8,
                device=self.device,
            )

        # lr and octave_scale must be finite: NaN would silently bake NaNs into
        # the pixels (the gradient itself is finite, so the NaN-gradient guard
        # can't catch it), and NaN octave_scale crashes round()/log() later with
        # an unhandled traceback.
        with self.assertRaises(SystemExit):
            deepdream.deepdream(
                self.model,
                "googlenet",
                tensor,
                targets=["inception3a"],
                iterations=1,
                lr=float("nan"),
                octave_scale=1.4,
                num_octaves=2,
                jitter=8,
                device=self.device,
            )

        with self.assertRaises(SystemExit):
            deepdream.deepdream(
                self.model,
                "googlenet",
                tensor,
                targets=["inception3a"],
                iterations=1,
                lr=1.0,
                octave_scale=float("nan"),
                num_octaves=2,
                jitter=8,
                device=self.device,
            )

        with self.assertRaises(SystemExit):
            deepdream.deepdream(
                self.model,
                "googlenet",
                tensor,
                targets=["inception3a"],
                iterations=1,
                lr=float("inf"),
                octave_scale=1.4,
                num_octaves=2,
                jitter=8,
                device=self.device,
            )

        # jitter must be >= 0
        with self.assertRaises(SystemExit):
            deepdream.deepdream(
                self.model,
                "googlenet",
                tensor,
                targets=["inception3a"],
                iterations=1,
                lr=1.0,
                octave_scale=1.4,
                num_octaves=2,
                jitter=-1,
                device=self.device,
            )

    def test_main_rejects_bad_flags(self):
        img = self.sample_img_path
        for extra, needle in [
            (["--jitter", "-5"], "jitter"),
            (["--lr", "-1"], "lr"),
            (["--lr", "nan"], "lr"),
            (["--lr", "inf"], "lr"),
            (["--size", "-1"], "size"),
            (["--iterations", "-1"], "iterations"),
            (["--octave-scale", "nan"], "octave-scale"),
            (["--octave-scale", "inf"], "octave-scale"),
            (["--name", "bad/name"], "name"),
        ]:
            result = run_cli([img, *extra])
            self.assertEqual(result.returncode, 1, f"expected failure for {extra}: {result.stderr}")
            self.assertIn(needle, result.stderr)

    @pytest.mark.integration
    def test_main_no_size_suffix_when_not_resized(self):
        # Image (100x80) is smaller than --size 512, so it is NOT downscaled
        # and the output name must not claim a "_512px" cap.
        # Default model=vgg16, default layer=features.22.
        result = run_cli(
            [
                self.sample_img_path,
                "--size",
                "512",
                "--iterations",
                "1",
                "--octaves",
                "1",
                "--out",
                self.temp_dir,
            ]
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "test_img_features-22.png")))
        self.assertFalse(
            os.path.exists(os.path.join(self.temp_dir, "test_img_512px_features-22.png"))
        )

    def test_main_friendly_model_load_error(self):
        # Weight-download failure or a corrupt file must be a friendly message
        # with exit code 1, not a traceback at model load time.
        script = os.path.abspath(__file__)
        with (
            mock.patch.object(
                deepdream,
                "build_model",
                side_effect=OSError("Failed to connect to download.pytorch.org"),
            ),
            mock.patch.object(
                sys,
                "argv",
                [
                    script,
                    self.sample_img_path,
                    "--iterations",
                    "1",
                    "--octaves",
                    "1",
                ],
            ),
        ):
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                with self.assertRaises(SystemExit) as ctx:
                    deepdream.main()
            self.assertEqual(ctx.exception.code, 1)
            self.assertIn("weights", err.getvalue())
            self.assertNotIn("Traceback", err.getvalue())

    def test_main_friendly_model_load_runtime_error(self):
        # A corrupt weight file / CUDA OOM during model load must also be a
        # friendly message, not a traceback.
        script = os.path.abspath(__file__)
        with (
            mock.patch.object(
                deepdream,
                "build_model",
                side_effect=RuntimeError("CUDA out of memory"),
            ),
            mock.patch.object(
                sys,
                "argv",
                [
                    script,
                    self.sample_img_path,
                    "--iterations",
                    "1",
                    "--octaves",
                    "1",
                ],
            ),
        ):
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                with self.assertRaises(SystemExit) as ctx:
                    deepdream.main()
            self.assertEqual(ctx.exception.code, 1)
            self.assertNotIn("Traceback", err.getvalue())

    def test_main_friendly_runtime_error(self):
        # A NaN/Inf gradient must surface as a friendly exit code 1, not a
        # traceback after a long run.
        script = os.path.abspath(__file__)
        with (
            mock.patch.object(deepdream, "build_model", return_value=None),
            mock.patch.object(
                deepdream,
                "deepdream",
                side_effect=RuntimeError("Gradient was NaN/Inf during ascent: try a lower --lr."),
            ),
            mock.patch.object(
                sys,
                "argv",
                [
                    script,
                    self.sample_img_path,
                    "--iterations",
                    "1",
                    "--octaves",
                    "1",
                ],
            ),
        ):
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                with self.assertRaises(SystemExit) as ctx:
                    deepdream.main()
            self.assertEqual(ctx.exception.code, 1)
            self.assertIn("try a lower --lr", err.getvalue())
            self.assertNotIn("Traceback", err.getvalue())

    def test_main_friendly_save_error(self):
        # A failed final save must surface as a friendly exit code 1, not a
        # traceback after the run.
        script = os.path.abspath(__file__)
        with (
            mock.patch.object(deepdream, "build_model", return_value=None),
            mock.patch.object(deepdream, "deepdream", return_value=torch.zeros(1, 3, 8, 8)),
            mock.patch.object(
                deepdream,
                "save_image",
                side_effect=OSError("No space left on device"),
            ),
            mock.patch.object(
                sys,
                "argv",
                [
                    script,
                    self.sample_img_path,
                    "--iterations",
                    "1",
                    "--octaves",
                    "1",
                ],
            ),
        ):
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                with self.assertRaises(SystemExit) as ctx:
                    deepdream.main()
            self.assertEqual(ctx.exception.code, 1)
            self.assertIn("could not save output", err.getvalue())
            self.assertIn("No space left on device", err.getvalue())
            self.assertNotIn("Traceback", err.getvalue())

    def test_main_friendly_load_error(self):
        # A failed image load (removed, unreadable, corrupt) must surface as a
        # friendly exit code 1, not a traceback.
        script = os.path.abspath(__file__)
        with (
            mock.patch.object(
                deepdream,
                "load_image_with_size",
                side_effect=OSError("No such file or directory"),
            ),
            mock.patch.object(
                sys,
                "argv",
                [
                    script,
                    self.sample_img_path,
                    "--iterations",
                    "1",
                    "--octaves",
                    "1",
                    "--size",
                    "32",
                ],
            ),
        ):
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                with self.assertRaises(SystemExit) as ctx:
                    deepdream.main()
            self.assertIn("Could not load image", str(ctx.exception))
            self.assertNotIn("Traceback", err.getvalue())

    def test_main_friendly_frame_save_error(self):
        # A failed frame save must surface as a friendly exit code 1, not a
        # traceback after a long run.
        script = os.path.abspath(__file__)
        with (
            mock.patch.object(deepdream, "build_model", return_value=None),
            mock.patch.object(
                deepdream,
                "deepdream",
                side_effect=OSError("No space left on device"),
            ),
            mock.patch.object(
                sys,
                "argv",
                [
                    script,
                    self.sample_img_path,
                    "--frames",
                    "--iterations",
                    "1",
                    "--octaves",
                    "1",
                    "--out",
                    self.temp_dir,
                ],
            ),
        ):
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                with self.assertRaises(SystemExit) as ctx:
                    deepdream.main()
            self.assertEqual(ctx.exception.code, 1)
            self.assertIn("frames", err.getvalue())
            self.assertIn("No space left on device", err.getvalue())
            self.assertNotIn("Traceback", err.getvalue())

    @pytest.mark.integration
    def test_main_unicode_out_path(self):
        # A CJK output path must not crash the final print: piped stdout would
        # default to cp1252 on Windows and raise UnicodeEncodeError after the
        # dream already finished. Defaults otherwise: googlenet / inception4c.
        script = ["-m", "app.deepdream"]
        out_dir = os.path.join(self.temp_dir, "梦想输出")
        result = run_module(
            [
                sys.executable,
                *script,
                self.sample_img_path,
                "--iterations",
                "1",
                "--octaves",
                "1",
                "--out",
                out_dir,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(os.path.exists(os.path.join(out_dir, "test_img_features-22.png")))
        self.assertNotIn("UnicodeEncodeError", result.stdout + result.stderr)

    def test_deepdream_pipeline(self):
        tensor = deepdream.load_image(self.sample_img_path, max_size=64)
        res = deepdream.deepdream(
            self.model,
            "googlenet",
            tensor,
            targets=["inception3a"],
            iterations=2,
            lr=1.0,
            octave_scale=1.4,
            num_octaves=2,
            jitter=8,
            device=self.device,
        )
        self.assertEqual(res.shape, tensor.shape)
        self.assertFalse(torch.isnan(res).any())

    def test_deepdream_frames(self):
        tensor = deepdream.load_image(self.sample_img_path, max_size=64)
        octave_scale = 1.4
        num_octaves = 2
        iterations = 2
        frames_dir = os.path.join(self.temp_dir, "frames")
        deepdream.deepdream(
            self.model,
            "googlenet",
            tensor,
            targets=["inception3a"],
            iterations=iterations,
            lr=1.0,
            octave_scale=octave_scale,
            num_octaves=num_octaves,
            jitter=8,
            device=self.device,
            frames_dir=frames_dir,
        )
        # Compute how many octaves actually ran (same logic deepdream uses).
        actual_octaves = deepdream.plan_octaves(
            tensor.shape[2], tensor.shape[3], num_octaves, octave_scale
        )
        # Verify every frame from every octave was saved.
        for oi in range(1, actual_octaves + 1):
            for it in range(1, iterations + 1):
                path = os.path.join(frames_dir, f"o{oi}_i{it}.png")
                self.assertTrue(os.path.exists(path), f"Missing frame: {path}")


class TestGuidedAndVideoFeatures(unittest.TestCase):
    _fixture_dir: str
    sample_img_path: str

    @classmethod
    def setUpClass(cls):
        cls._fixture_dir = tempfile.mkdtemp(prefix="deepdream_test_")
        cls.sample_img_path = os.path.join(cls._fixture_dir, "sample.jpg")
        Image.new("RGB", (128, 96), color=(128, 64, 200)).save(cls.sample_img_path)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._fixture_dir, ignore_errors=True)

    def test_resolve_target_class_none(self):
        cid, cname = deepdream.resolve_target_class(None)
        self.assertIsNone(cid)
        self.assertIsNone(cname)
        cid, cname = deepdream.resolve_target_class("")
        self.assertIsNone(cid)

    def test_resolve_target_class_by_index(self):
        cid, cname = deepdream.resolve_target_class("130")
        self.assertEqual(cid, 130)
        self.assertEqual(cname, "flamingo")

    def test_resolve_target_class_by_name(self):
        cid, cname = deepdream.resolve_target_class("flamingo")
        self.assertEqual(cid, 130)
        self.assertEqual(cname, "flamingo")

        cid, cname = deepdream.resolve_target_class("Pizza")
        self.assertEqual(cid, 963)
        self.assertEqual(cname, "pizza")

    def test_resolve_target_class_compound_formats(self):
        cid, cname = deepdream.resolve_target_class("#130: flamingo")
        self.assertEqual(cid, 130)
        self.assertEqual(cname, "flamingo")

        cid, cname = deepdream.resolve_target_class("130: flamingo")
        self.assertEqual(cid, 130)
        self.assertEqual(cname, "flamingo")

        cid, cname = deepdream.resolve_target_class("963 - pizza")
        self.assertEqual(cid, 963)
        self.assertEqual(cname, "pizza")

    def test_imagenet_categories_lookup_and_search(self):
        import app.imagenet_categories as ic

        res = ic.lookup_category("#130: flamingo")
        self.assertEqual(res, (130, "flamingo", "#130: flamingo"))

        res2 = ic.lookup_category("pizza")
        self.assertEqual(res2, (963, "pizza", "#963: pizza"))

        res3 = ic.lookup_category("130")
        self.assertEqual(res3, (130, "flamingo", "#130: flamingo"))

        # Search by substring
        matches = ic.search_categories("shark", max_results=10)
        self.assertTrue(any("great white shark" in m[1] for m in matches))

        # Search with category filter
        food_matches = ic.search_categories("", category_filter="🍕 Food", max_results=50)
        self.assertTrue(all(923 <= m[0] <= 968 for m in food_matches))

    def test_resolve_target_class_invalid(self):
        with self.assertRaises(SystemExit):
            deepdream.resolve_target_class("9999")
        with self.assertRaises(SystemExit):
            deepdream.resolve_target_class("completely_nonexistent_class_xyz")

    def test_apply_zoom_and_rotate(self):
        t = torch.zeros(1, 3, 64, 64)
        zoomed = deepdream.apply_zoom_and_rotate(t, zoom_scale=1.05, zoom_rot=1.0)
        self.assertEqual(zoomed.shape, t.shape)
        self.assertTrue(torch.isfinite(zoomed).all())

    def test_tensor_pil_roundtrip(self):
        t = torch.rand(1, 3, 32, 32)
        pil = deepdream.tensor_to_pil(t)
        self.assertIsInstance(pil, Image.Image)
        t_back = deepdream.pil_to_tensor(pil)
        self.assertEqual(t_back.shape, t.shape)

    def test_zoom_parameter_validation(self):
        img_path = self.sample_img_path
        res = run_cli([img_path, "--zoom", "--zoom-frames", "-5"])
        self.assertEqual(res.returncode, 1)
        self.assertIn("--zoom-frames must be > 0", res.stderr)

        res = run_cli([img_path, "--zoom", "--fps", "0"])
        self.assertEqual(res.returncode, 1)
        self.assertIn("--zoom-fps must be > 0", res.stderr)

        res = run_cli([img_path, "--video", "--video-blend", "2.5"])
        self.assertEqual(res.returncode, 1)
        self.assertIn("--video-blend must be between 0.0 and 1.0", res.stderr)

        res = run_cli([img_path, "--zoom", "--zoom-scale", "0.95"])
        self.assertEqual(res.returncode, 1)
        self.assertIn("--zoom-scale must be >= 1.0", res.stderr)

    def test_extract_guide_features(self):
        sample = self.sample_img_path
        model = deepdream.build_model("googlenet", "cpu")
        gf = deepdream.extract_guide_features(
            model, "googlenet", sample, ["inception3a"], "cpu", max_size=64
        )
        self.assertIn("inception3a", gf)
        self.assertEqual(gf["inception3a"].dim(), 2)

    def test_compile_timelapse_ignores_non_matching_files(self):
        with tempfile.TemporaryDirectory() as td:
            frames = os.path.join(td, "frames")
            os.makedirs(frames)
            Image.new("RGB", (64, 64)).save(os.path.join(frames, "o1_i1.png"))
            Image.new("RGB", (64, 64)).save(os.path.join(frames, "o1_i2.png"))
            Image.new("RGB", (64, 64)).save(os.path.join(frames, "o999_i1.png"))
            # Extra non-frame PNG that should be ignored without raising TypeError
            Image.new("RGB", (64, 64)).save(os.path.join(frames, "preview.png"))
            out_mp4 = os.path.join(td, "tl.mp4")
            deepdream.compile_timelapse(frames, out_mp4, fps=10)
            self.assertTrue(os.path.isfile(out_mp4))
            self.assertGreater(os.path.getsize(out_mp4), 0)

    def test_zoom_animation_preserves_min_size(self):
        model = deepdream.build_model("inceptionv3", "cpu")
        t_75 = torch.randn(1, 3, 75, 75)
        with tempfile.TemporaryDirectory() as td:
            out_mp4 = os.path.join(td, "zoom_inc.mp4")
            deepdream.dream_zoom_animation(
                model=model,
                model_key="inceptionv3",
                img=t_75,
                targets=["Mixed_5b"],
                num_frames=2,
                iterations_per_frame=1,
                lr=0.01,
                jitter=0,
                zoom_scale=1.02,
                zoom_rot=0.0,
                device="cpu",
                out_path=out_mp4,
                fps=10,
            )
            self.assertTrue(os.path.isfile(out_mp4))
            self.assertGreater(os.path.getsize(out_mp4), 0)

    def test_zoom_frames_only_saves_frames_without_mp4(self):
        # Generate/save split: out_path=None persists frame_*.png + meta.json
        # and encodes nothing.
        model = deepdream.build_model("googlenet", "cpu")
        with tempfile.TemporaryDirectory() as td:
            frames_dir = os.path.join(td, "seq")
            result = deepdream.dream_zoom_animation(
                model=model,
                model_key="googlenet",
                img=torch.randn(1, 3, 64, 64),
                targets=["inception4c"],
                num_frames=3,
                iterations_per_frame=0,
                lr=0.01,
                jitter=0,
                zoom_scale=1.02,
                zoom_rot=0.0,
                device="cpu",
                out_path=None,
                fps=12,
                frames_dir=frames_dir,
            )
            self.assertEqual(result, frames_dir)
            for i in (1, 2, 3):
                self.assertTrue(
                    os.path.isfile(os.path.join(frames_dir, f"frame_{i:06d}.png")),
                    f"frame_{i:06d}.png",
                )
            self.assertFalse(
                any(n.lower().endswith(".mp4") for n in os.listdir(td)),
                "frames-only must not encode an mp4",
            )
            meta = deepdream.read_video_frames_meta(frames_dir)
            self.assertEqual(meta.get("mode"), "zoom")
            self.assertAlmostEqual(float(meta["fps"]), 12.0)

    def test_zoom_frames_only_requires_a_destination(self):
        model = deepdream.build_model("googlenet", "cpu")
        with self.assertRaises(ValueError):
            deepdream.dream_zoom_animation(
                model=model,
                model_key="googlenet",
                img=torch.randn(1, 3, 64, 64),
                targets=["inception4c"],
                num_frames=1,
                iterations_per_frame=0,
                lr=0.01,
                jitter=0,
                zoom_scale=1.02,
                zoom_rot=0.0,
                device="cpu",
                out_path=None,
                fps=10,
                frames_dir=None,
            )

    def test_zoom_animation_accepts_video_export_config(self):
        model = deepdream.build_model("googlenet", "cpu")
        with tempfile.TemporaryDirectory() as td:
            frames_dir = os.path.join(td, "seq_cfg")
            cfg = deepdream.VideoExportConfig(fps=15, frames_dir=frames_dir)
            result = deepdream.dream_zoom_animation(
                model=model,
                model_key="googlenet",
                img=torch.randn(1, 3, 64, 64),
                targets=["inception4c"],
                num_frames=2,
                iterations_per_frame=0,
                lr=0.01,
                jitter=0,
                zoom_scale=1.02,
                zoom_rot=0.0,
                device="cpu",
                out_path=None,
                config=cfg,
            )
            self.assertEqual(result, frames_dir)
            meta = deepdream.read_video_frames_meta(frames_dir)
            self.assertAlmostEqual(float(meta["fps"]), 15.0)

    def test_zoom_animation_resume_generates_remaining_frames(self):
        model = deepdream.build_model("googlenet", "cpu")
        with tempfile.TemporaryDirectory() as td:
            frames_dir = os.path.join(td, "seq_resume")
            # Step 1: Generate initial 2 frames
            deepdream.dream_zoom_animation(
                model=model,
                model_key="googlenet",
                img=torch.randn(1, 3, 64, 64),
                targets=["inception4c"],
                num_frames=2,
                iterations_per_frame=0,
                lr=0.01,
                jitter=0,
                zoom_scale=1.02,
                zoom_rot=0.0,
                device="cpu",
                out_path=None,
                fps=10,
                frames_dir=frames_dir,
            )
            f1 = os.path.join(frames_dir, "frame_000001.png")
            f2 = os.path.join(frames_dir, "frame_000002.png")
            self.assertTrue(os.path.isfile(f1))
            self.assertTrue(os.path.isfile(f2))
            f1_mtime = os.path.getmtime(f1)
            f2_mtime = os.path.getmtime(f2)

            # Step 2: Resume to 4 frames
            resumed_video = os.path.join(td, "resumed.mp4")
            deepdream.dream_zoom_animation(
                model=model,
                model_key="googlenet",
                img=torch.randn(1, 3, 64, 64),
                targets=["inception4c"],
                num_frames=4,
                iterations_per_frame=0,
                lr=0.01,
                jitter=0,
                zoom_scale=1.02,
                zoom_rot=0.0,
                device="cpu",
                out_path=resumed_video,
                fps=10,
                frames_dir=frames_dir,
                resume=True,
            )
            f3 = os.path.join(frames_dir, "frame_000003.png")
            f4 = os.path.join(frames_dir, "frame_000004.png")
            self.assertTrue(os.path.isfile(f3))
            self.assertTrue(os.path.isfile(f4))
            self.assertTrue(os.path.isfile(resumed_video))
            import imageio.v2 as imageio_v2

            reader: Any = imageio_v2.get_reader(resumed_video)
            try:
                self.assertEqual(sum(1 for _ in reader), 4)
            finally:
                reader.close()
            # Initial frames were not recomputed / overwritten
            self.assertEqual(os.path.getmtime(f1), f1_mtime)
            self.assertEqual(os.path.getmtime(f2), f2_mtime)

    def test_list_video_frames_prefers_numbered_and_skips_temps(self):
        with tempfile.TemporaryDirectory() as td:
            Image.new("RGB", (16, 16)).save(os.path.join(td, "frame_000002.png"))
            Image.new("RGB", (16, 16)).save(os.path.join(td, "frame_000001.png"))
            Image.new("RGB", (16, 16)).save(os.path.join(td, "o1_i1.png"))
            with open(os.path.join(td, "_gui_preview_p1_j1.png"), "wb") as fh:
                fh.write(b"x")
            files = deepdream.list_video_frames(td)
            self.assertEqual(
                [os.path.basename(p) for p in files],
                ["frame_000001.png", "frame_000002.png"],
            )
            self.assertEqual(deepdream.list_video_frames(os.path.join(td, "missing")), [])

    def test_video_frames_meta_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(deepdream.read_video_frames_meta(td), {})
            self.assertTrue(
                deepdream.write_video_frames_meta(
                    td, fps=24, mode="video", source="clip.mp4", width=64, height=48
                )
            )
            meta = deepdream.read_video_frames_meta(td)
            self.assertEqual(meta["mode"], "video")
            self.assertAlmostEqual(float(meta["fps"]), 24.0)
            self.assertEqual(meta["source"], "clip.mp4")

    def test_compile_video_frames_multi_quality(self):
        # Same saved sequence exports repeatedly at different qualities.
        with tempfile.TemporaryDirectory() as td:
            frames = os.path.join(td, "seq")
            os.makedirs(frames)
            for i in range(3):
                Image.new("RGB", (64, 48), color=(i * 60, 100, 150)).save(
                    os.path.join(frames, f"frame_{i + 1:06d}.png")
                )
            deepdream.write_video_frames_meta(frames, fps=10, mode="zoom")
            small = os.path.join(td, "small.mp4")
            lossless = os.path.join(td, "lossless.mp4")
            deepdream.compile_video_frames(frames, small, fps=10, crf=28, preset="fast")
            deepdream.compile_video_frames(frames, lossless, fps=10, crf=0)
            for p in (small, lossless):
                self.assertTrue(os.path.isfile(p), p)
                self.assertGreater(os.path.getsize(p), 0, p)

    def test_compile_video_frames_rejects_empty_dir(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(SystemExit):
                deepdream.compile_video_frames(td, os.path.join(td, "x.mp4"))

    def test_encode_frames_cli_mode(self):
        with tempfile.TemporaryDirectory() as td:
            frames = os.path.join(td, "seq")
            os.makedirs(frames)
            for i in range(2):
                Image.new("RGB", (64, 48), color=(i * 80, 120, 160)).save(
                    os.path.join(frames, f"frame_{i + 1:06d}.png")
                )
            deepdream.write_video_frames_meta(frames, fps=15, mode="zoom")
            out = os.path.join(td, "enc.mp4")
            res = run_cli(["--encode-frames", frames, "--out", out])
            self.assertEqual(res.returncode, 0, res.stderr)
            self.assertIn("Done in", res.stdout)
            self.assertIn("Encoding frame 1/2 (50%)", res.stdout)
            self.assertIn("Encoding frame 2/2 (100%)", res.stdout)
            self.assertTrue(os.path.isfile(out))
            # float fps from meta is honored without an explicit --fps
            self.assertIn("@ 15", res.stdout)

    def test_encode_frames_explicit_fps_overrides_meta(self):
        with tempfile.TemporaryDirectory() as td:
            frames = os.path.join(td, "seq")
            os.makedirs(frames)
            for i in range(2):
                Image.new("RGB", (64, 48)).save(os.path.join(frames, f"frame_{i + 1:06d}.png"))
            deepdream.write_video_frames_meta(frames, fps=15, mode="zoom")
            out = os.path.join(td, "enc.mp4")
            res = run_cli(["--encode-frames", frames, "--out", out, "--fps", "24"])
            self.assertEqual(res.returncode, 0, res.stderr)
            # An explicit --fps wins over sequence metadata even when main() is
            # invoked in-process, where sys.argv is the test runner's.
            self.assertIn("@ 24", res.stdout)

    def test_encode_frames_cli_out_dir_systematic_name_and_conflict_resolution(self):
        with tempfile.TemporaryDirectory() as td:
            frames = os.path.join(td, "seq")
            os.makedirs(frames)
            for i in range(2):
                Image.new("RGB", (64, 48), color=(i * 80, 120, 160)).save(
                    os.path.join(frames, f"frame_{i + 1:06d}.png")
                )
            deepdream.write_video_frames_meta(
                frames,
                fps=15,
                mode="zoom",
                source="landscape.png",
                model="googlenet",
                layers=["inception4c"],
                size=512,
                iterations=2,
            )
            out_dir = os.path.join(td, "out")
            os.makedirs(out_dir)

            # First export into directory: uses systematic name
            res1 = run_cli(["--encode-frames", frames, "--out", out_dir, "--crf", "17"])
            self.assertEqual(res1.returncode, 0, res1.stderr)
            expected1 = os.path.join(
                out_dir, "landscape_512px_googlenet_inception4c_i2_zoom_crf17.mp4"
            )
            self.assertTrue(os.path.isfile(expected1), f"Expected {expected1}")

            # Second export of the same frames into the same directory: conflict resolution appends _1
            res2 = run_cli(["--encode-frames", frames, "--out", out_dir, "--crf", "17"])
            self.assertEqual(res2.returncode, 0, res2.stderr)
            expected2 = os.path.join(
                out_dir, "landscape_512px_googlenet_inception4c_i2_zoom_crf17_1.mp4"
            )
            self.assertTrue(os.path.isfile(expected2), f"Expected {expected2}")

    def test_encode_frames_cli_rejects_conflicts(self):
        res = run_cli(["--encode-frames", "somedir", "--zoom", self.sample_img_path])
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("--encode-frames cannot be combined", res.stderr)

    def test_frames_only_cli_requires_zoom_or_video(self):
        res = run_cli([self.sample_img_path, "--frames-only"])
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("--frames-only requires --zoom or --video", res.stderr)

    def test_clear_frames_cli_requires_iteration_frame_mode(self):
        res = run_cli([self.sample_img_path, "--clear-frames"])
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("--clear-frames requires --frames or --timelapse", res.stderr)

    @pytest.mark.integration
    def test_cli_multi_image_queue(self):
        with tempfile.TemporaryDirectory() as td:
            img1 = os.path.join(td, "batch1.jpg")
            img2 = os.path.join(td, "batch2.jpg")
            Image.new("RGB", (100, 80), color=(100, 150, 200)).save(img1)
            Image.new("RGB", (100, 80), color=(200, 150, 100)).save(img2)

            out_dir = os.path.join(td, "batch_out")
            result = run_cli(
                [
                    img1,
                    img2,
                    "--out",
                    out_dir,
                    "--iterations",
                    "1",
                    "--octaves",
                    "1",
                    "--no-cuda",
                ]
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Queue completed: 2 generations", result.stdout)
            self.assertTrue(os.path.isfile(os.path.join(out_dir, "batch1_features-22.png")))
            self.assertTrue(os.path.isfile(os.path.join(out_dir, "batch2_features-22.png")))

    @pytest.mark.integration
    def test_cli_multi_image_queue_continues_past_missing_input(self):
        with tempfile.TemporaryDirectory() as td:
            img1 = os.path.join(td, "batch1.jpg")
            Image.new("RGB", (100, 80), color=(100, 150, 200)).save(img1)
            missing = os.path.join(td, "does_not_exist.jpg")

            out_dir = os.path.join(td, "batch_out")
            result = run_cli(
                [
                    img1,
                    missing,
                    "--out",
                    out_dir,
                    "--iterations",
                    "1",
                    "--octaves",
                    "1",
                    "--no-cuda",
                ]
            )
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn(f"Error: Input image not found: {missing}", result.stderr)
            self.assertIn("Queue completed with 1 error(s) out of 2", result.stderr)
            self.assertTrue(os.path.isfile(os.path.join(out_dir, "batch1_features-22.png")))

    @pytest.mark.integration
    def test_cli_multi_image_queue_continues_past_too_small_input(self):
        with tempfile.TemporaryDirectory() as td:
            img1 = os.path.join(td, "batch1.jpg")
            too_small = os.path.join(td, "tiny.png")
            Image.new("RGB", (100, 80), color=(100, 150, 200)).save(img1)
            Image.new("RGB", (14, 14), color=(50, 50, 50)).save(too_small)

            out_dir = os.path.join(td, "batch_out")
            result = run_cli(
                [
                    img1,
                    too_small,
                    "--out",
                    out_dir,
                    "--iterations",
                    "1",
                    "--octaves",
                    "1",
                    "--no-cuda",
                ]
            )
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn("too small", result.stderr)
            self.assertIn("Queue completed with 1 error(s) out of 2", result.stderr)
            self.assertTrue(os.path.isfile(os.path.join(out_dir, "batch1_features-22.png")))

    def test_cli_multi_image_all_inputs_fail_before_model_load(self):
        # Every input fails during planning (missing + too small), so the queue
        # must report the failures and exit 1 WITHOUT touching a model at all.
        with tempfile.TemporaryDirectory() as td:
            missing = os.path.join(td, "does_not_exist.jpg")
            too_small = os.path.join(td, "tiny.png")
            Image.new("RGB", (14, 14), color=(50, 50, 50)).save(too_small)

            result = run_cli(
                [
                    missing,
                    too_small,
                    "--out",
                    os.path.join(td, "batch_out"),
                    "--iterations",
                    "1",
                    "--octaves",
                    "1",
                    "--no-cuda",
                ]
            )
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn("Queue completed with 2 error(s) out of 2", result.stderr)
            self.assertNotIn("Loading VGG16", result.stdout)
            self.assertNotIn("Loading VGG16", result.stderr)

    def test_cli_single_missing_input_fails_fast(self):
        with tempfile.TemporaryDirectory() as td:
            missing = os.path.join(td, "does_not_exist.jpg")
            result = run_cli([missing, "--no-cuda"])
            self.assertEqual(result.returncode, 1)
            self.assertIn(f"Error: Input image not found: {missing}", result.stderr)

    def test_resolve_video_quality(self):
        # Presets
        crf_s, pr_s = deepdream.resolve_video_quality("small")
        self.assertEqual(crf_s, 28)
        self.assertEqual(pr_s, "fast")

        crf_b, pr_b = deepdream.resolve_video_quality("balanced")
        self.assertEqual(crf_b, 17)
        self.assertEqual(pr_b, "medium")

        crf_l, pr_l = deepdream.resolve_video_quality("lossless")
        self.assertEqual(crf_l, 0)
        self.assertEqual(pr_l, "medium")

        # Explicit CRF override
        crf_c, pr_c = deepdream.resolve_video_quality("balanced", crf=10)
        self.assertEqual(crf_c, 10)

        # Out-of-range validation
        with self.assertRaises(ValueError):
            deepdream.resolve_video_quality("balanced", crf=-1)
        with self.assertRaises(ValueError):
            deepdream.resolve_video_quality("balanced", crf=52)

    def test_video_quality_cli_flags_validation(self):
        img_path = self.sample_img_path

        # Invalid CRF > 51
        res = run_cli([img_path, "--crf", "99"])
        self.assertEqual(res.returncode, 1)
        self.assertIn("--crf must be between 0 and 51", res.stderr)

        # Invalid preset choice
        res2 = run_cli([img_path, "--video-quality", "ultra_insane"])
        self.assertNotEqual(res2.returncode, 0)

    def test_compile_timelapse_with_crf(self):
        with tempfile.TemporaryDirectory() as td:
            frames = os.path.join(td, "frames")
            os.makedirs(frames)
            Image.new("RGB", (64, 64), color=(255, 0, 0)).save(os.path.join(frames, "o1_i1.png"))
            Image.new("RGB", (64, 64), color=(0, 255, 0)).save(os.path.join(frames, "o1_i2.png"))

            # Lossless compilation (CRF 0)
            out_lossless = os.path.join(td, "tl_lossless.mp4")
            deepdream.compile_timelapse(frames, out_lossless, fps=10, crf=0)
            self.assertTrue(os.path.isfile(out_lossless))
            self.assertGreater(os.path.getsize(out_lossless), 0)

            # Small / compressed compilation (CRF 28)
            out_small = os.path.join(td, "tl_small.mp4")
            deepdream.compile_timelapse(frames, out_small, fps=10, crf=28, preset="fast")
            self.assertTrue(os.path.isfile(out_small))
            self.assertGreater(os.path.getsize(out_small), 0)

    def test_video_mode_default_iterations_decimation(self):
        img_path = self.sample_img_path
        # When --zoom is passed without --iterations, iterations must decimate to 1/10th of model default
        with (
            mock.patch.object(deepdream, "build_model", return_value=mock.MagicMock()),
            mock.patch.object(deepdream, "dream_zoom_animation") as mock_zoom,
        ):
            with mock.patch.object(sys, "argv", ["deepdream.py", img_path, "--zoom", "--no-cuda"]):
                deepdream.main()
                self.assertTrue(mock_zoom.called)
                _, kwargs = mock_zoom.call_args
                # GoogLeNet default iterations is 20, 1/10th is 2
                self.assertEqual(kwargs["iterations_per_frame"], 2)

        # When --iterations IS passed explicitly, it is preserved
        with (
            mock.patch.object(deepdream, "build_model", return_value=mock.MagicMock()),
            mock.patch.object(deepdream, "dream_zoom_animation") as mock_zoom,
        ):
            with mock.patch.object(
                sys, "argv", ["deepdream.py", img_path, "--zoom", "--iterations", "7", "--no-cuda"]
            ):
                deepdream.main()
                self.assertTrue(mock_zoom.called)
                _, kwargs = mock_zoom.call_args
                self.assertEqual(kwargs["iterations_per_frame"], 7)


class TestForwardWithActivations(unittest.TestCase):
    """Regression coverage for the combined layer+class path."""

    @staticmethod
    def _toy_model():
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.block = torch.nn.Conv2d(3, 4, 3, padding=1)
                self.fc = torch.nn.Linear(4, 10)

            def forward(self, x):
                x = torch.relu(self.block(x))
                x = x.mean(dim=(2, 3))
                return self.fc(x)

        return Toy()

    def test_captures_activations_and_returns_logits(self):
        model = self._toy_model()
        x = torch.rand(1, 3, 16, 16, requires_grad=True)
        acts, logits = deepdream.forward_with_activations(model, x, ["block"])
        self.assertIn("block", acts)
        self.assertEqual(tuple(acts["block"].shape), (1, 4, 16, 16))
        self.assertEqual(tuple(logits.shape), (1, 10))
        grad = torch.autograd.grad(acts["block"].sum(), x)[0]
        self.assertTrue(torch.isfinite(grad).all())

    def test_unknown_target_raises_and_leaves_no_hooks(self):
        model = self._toy_model()
        x = torch.rand(1, 3, 16, 16)
        with self.assertRaises(RuntimeError):
            deepdream.forward_with_activations(model, x, ["block", "does_not_exist"])
        # Hooks must be removed on the early raise: a later forward still
        # captures exactly one activation, not a duplicate.
        acts, _ = deepdream.forward_with_activations(model, x, ["block"])
        self.assertEqual(list(acts), ["block"])

    def test_combined_gradient_ascent_is_finite(self):
        model = self._toy_model()
        img = torch.rand(1, 3, 16, 16) * 0.5
        out, loss = deepdream.gradient_ascent(
            model,
            "googlenet",
            img,
            ["block"],
            lr=0.03,
            jitter=0,
            target_class=3,
        )
        self.assertEqual(tuple(out.shape), tuple(img.shape))
        self.assertTrue(torch.isfinite(out).all())
        self.assertFalse(torch.isnan(torch.tensor(loss)))


class TestPreviewWrite(unittest.TestCase):
    def test_save_preview_frame_writes_and_returns_true(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "prev.png")
            arr = np.zeros((8, 8, 3), dtype="uint8")
            self.assertTrue(deepdream.save_preview_frame(arr, out))
            self.assertTrue(os.path.isfile(out))

    def test_save_preview_frame_failure_returns_false(self):
        # Writing onto an existing directory must fail gracefully (no raise).
        with tempfile.TemporaryDirectory() as d:
            arr = np.zeros((8, 8, 3), dtype="uint8")
            self.assertFalse(deepdream.save_preview_frame(arr, d))


class TestModelExecutionValidation(unittest.TestCase):
    """Validation of layer extraction, loss calculation, and model resolution."""

    def test_run_to_layers_empty_targets_raises(self):
        model = torch.nn.Module()
        model.block = torch.nn.Identity()
        x = torch.randn(1, 3, 8, 8)
        with self.assertRaises(RuntimeError):
            deepdream.run_to_layers(model, "googlenet", x, set())
        with self.assertRaises(RuntimeError):
            deepdream.run_to_layers(model, "googlenet", x, [])

    def test_run_to_layers_unknown_model(self):
        model = torch.nn.Module()
        x = torch.randn(1, 3, 8, 8)
        with self.assertRaises(SystemExit):
            deepdream.run_to_layers(model, "nosuchmodel", x, ["inception3a"])

    def test_activation_loss_empty_acts_raises(self):
        with self.assertRaises(RuntimeError):
            deepdream._activation_loss({}, {"inception3a": 1.0})

    def test_validate_input_size_unknown_model(self):
        with self.assertRaises(SystemExit):
            deepdream.validate_input_size(100, 100, "nosuchmodel")

    def test_resolve_layer_unknown_model(self):
        with self.assertRaises(SystemExit):
            deepdream.resolve_layer("inception4c", "nosuchmodel")


class TestOctaveAndScaleValidation(unittest.TestCase):
    """Octave planning and scale parameter validation."""

    def test_plan_octaves_rejects_bad_scale(self):
        for bad in (1.0, 0.5, float("nan"), float("inf")):
            with self.subTest(scale=bad):
                with self.assertRaises(ValueError):
                    deepdream.plan_octaves(400, 400, 0, bad)
                with self.assertRaises(ValueError):
                    deepdream.auto_octave_count(400, bad)


class TestFrameStorageUtilities(unittest.TestCase):
    """Frame caching, directory safety, and file overwriting logic."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.sample_img_path = os.path.join(self.temp_dir, "test_img.jpg")
        img = Image.new("RGB", (100, 80), color=(128, 64, 200))
        img.save(self.sample_img_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_deepdream_frames_zero_iterations_keeps_stale_and_notes(self):
        frames_dir = os.path.join(self.temp_dir, "frames")
        os.makedirs(frames_dir)
        stale = os.path.join(frames_dir, "o1_i1.png")
        Image.new("RGB", (64, 64)).save(stale)
        tensor = deepdream.load_image(self.sample_img_path, max_size=64)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            deepdream.deepdream(
                None,
                "googlenet",
                tensor,
                targets=["inception3a"],
                iterations=0,
                lr=1.0,
                octave_scale=1.4,
                num_octaves=1,
                jitter=0,
                device="cpu",
                frames_dir=frames_dir,
            )
        self.assertIn("no frames will be written", buf.getvalue())
        self.assertTrue(os.path.isfile(stale))

    def test_clear_stale_frames_refuses_foreign_directory(self):
        frames_dir = os.path.join(self.temp_dir, "shared")
        os.makedirs(frames_dir)
        stale = os.path.join(frames_dir, "o1_i1.png")
        foreign = os.path.join(frames_dir, "holiday.png")
        Image.new("RGB", (8, 8)).save(stale)
        Image.new("RGB", (8, 8)).save(foreign)
        with self.assertRaisesRegex(SystemExit, "Use an empty, dedicated output folder"):
            deepdream.clear_stale_frames(frames_dir)
        self.assertTrue(os.path.isfile(stale))
        self.assertTrue(os.path.isfile(foreign))

    def test_clear_stale_frames_requires_explicit_permission(self):
        frames_dir = os.path.join(self.temp_dir, "frames_owned")
        os.makedirs(frames_dir)
        stale = os.path.join(frames_dir, "o2_i3.png")
        Image.new("RGB", (8, 8)).save(stale)
        with self.assertRaisesRegex(SystemExit, "pass --clear-frames"):
            deepdream.clear_stale_frames(frames_dir)
        self.assertTrue(os.path.isfile(stale))
        deepdream.clear_stale_frames(frames_dir, allow_delete=True)
        self.assertFalse(os.path.exists(stale))

    def test_clear_stale_frames_never_removes_video_frames_or_metadata(self):
        frames_dir = os.path.join(self.temp_dir, "video_frames")
        os.makedirs(frames_dir)
        frame = os.path.join(frames_dir, "frame_000001.png")
        meta = os.path.join(frames_dir, "meta.json")
        Image.new("RGB", (8, 8)).save(frame)
        with open(meta, "w", encoding="utf-8") as fh:
            fh.write("{}")
        with self.assertRaisesRegex(SystemExit, "not recognized as iteration frames"):
            deepdream.clear_stale_frames(frames_dir, allow_delete=True)
        self.assertTrue(os.path.isfile(frame))
        self.assertTrue(os.path.isfile(meta))

    def test_save_image_warns_on_overwrite(self):
        tensor = torch.zeros((1, 3, 8, 8), dtype=torch.float32)
        out_path = os.path.join(self.temp_dir, "overwrite_check.png")
        Image.new("RGB", (8, 8)).save(out_path)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            deepdream.save_image(tensor, out_path, warn_on_overwrite=True)
        self.assertIn("already exists", err.getvalue())
        self.assertTrue(os.path.exists(out_path))


class TestVideoProcessingLogic(unittest.TestCase):
    """Zoom transformations, video quality presets, and video blending."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.sample_img_path = os.path.join(self.temp_dir, "test_img.jpg")
        img = Image.new("RGB", (100, 80), color=(128, 64, 200))
        img.save(self.sample_img_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_apply_zoom_noop_is_exact(self):
        t = torch.rand(1, 3, 32, 32) * 0.5 - 0.25
        out = deepdream.apply_zoom_and_rotate(t, zoom_scale=1.0, zoom_rot=0.0)
        self.assertTrue(torch.equal(out, t))
        self.assertFalse(out.requires_grad)
        zoomed = deepdream.apply_zoom_and_rotate(t, zoom_scale=1.05, zoom_rot=1.0)
        self.assertEqual(zoomed.shape, t.shape)
        self.assertTrue(torch.isfinite(zoomed).all())

    def test_resolve_video_quality_rejects_bad_preset(self):
        with self.assertRaises(ValueError):
            deepdream.resolve_video_quality("balanced", crf=None, preset="ultrafast-invalid")
        for preset in sorted(deepdream.FFMPEG_PRESETS):
            crf, got = deepdream.resolve_video_quality("balanced", crf=None, preset=preset)
            self.assertEqual(crf, 17)
            self.assertEqual(got, preset)

    def test_main_value_error_is_friendly(self):
        script = os.path.abspath(__file__)
        with (
            mock.patch.object(deepdream, "build_model", return_value=None),
            mock.patch.object(
                deepdream,
                "deepdream",
                side_effect=ValueError("unsupported image data"),
            ),
            mock.patch.object(
                sys,
                "argv",
                [
                    script,
                    self.sample_img_path,
                    "--iterations",
                    "1",
                    "--octaves",
                    "1",
                ],
            ),
        ):
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                with self.assertRaises(SystemExit) as ctx:
                    deepdream.main()
            self.assertEqual(ctx.exception.code, 1)
            self.assertIn("unsupported image data", err.getvalue())
            self.assertNotIn("Traceback", err.getvalue())

    def test_apply_seed_makes_jitter_reproducible(self):
        # Jitter draws from Python's random, so torch seeding alone would not
        # make a dream repeatable.
        deepdream.apply_seed(1234)
        first = [random.random() for _ in range(5)]
        deepdream.apply_seed(1234)
        second = [random.random() for _ in range(5)]
        self.assertEqual(first, second)

    def test_dream_is_reproducible_with_same_seed(self):
        model = deepdream.build_model("googlenet", "cpu")
        img = Image.new("RGB", (96, 96), color=(90, 120, 200))
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "seeded.jpg")
            img.save(path)
            digests = []
            for run in range(2):
                out_dir = os.path.join(td, f"out{run}")
                with mock.patch.object(
                    deepdream, "_init_device_and_model", return_value=("cpu", model, None)
                ):
                    deepdream.main(
                        [
                            path,
                            "--model",
                            "googlenet",
                            "--iterations",
                            "2",
                            "--octaves",
                            "1",
                            "--jitter",
                            "32",
                            "--seed",
                            "7",
                            "--out",
                            out_dir,
                            "--name",
                            "dreamed",
                        ]
                    )
                produced = os.path.join(out_dir, "dreamed_inception4c.png")
                with Image.open(produced) as result:
                    digests.append(result.tobytes())
            self.assertEqual(digests[0], digests[1])

    def test_batch_results_independent_of_input_count(self):
        # A batch seeds each image with --seed + position, so one image alone and
        # the same image first in a pair must come out identical.
        model = deepdream.build_model("googlenet", "cpu")
        img = Image.new("RGB", (96, 96), color=(200, 60, 90))
        with tempfile.TemporaryDirectory() as td:
            a = os.path.join(td, "a.jpg")
            b = os.path.join(td, "b.jpg")
            img.save(a)
            img.save(b)
            base = [
                "--model",
                "googlenet",
                "--iterations",
                "2",
                "--octaves",
                "1",
                "--jitter",
                "32",
                "--seed",
                "11",
                "--out",
                td,
            ]
            with mock.patch.object(
                deepdream, "_init_device_and_model", return_value=("cpu", model, None)
            ):
                deepdream.main([a, *base, "--name", "solo"])
                deepdream.main([a, b, *base, "--name", "pair"])
            solo_path = next(
                p for p in os.listdir(td) if p.startswith("solo") and p.endswith(".png")
            )
            # Batch output is numbered; the first image keeps position 1.
            first_in_batch = os.path.join(td, "pair_1_inception4c.png")
            with (
                Image.open(os.path.join(td, solo_path)) as solo,
                Image.open(first_in_batch) as first,
            ):
                self.assertEqual(solo.tobytes(), first.tobytes())

    def test_video_final_preview_written(self):
        if not deepdream._VIDEO_AVAILABLE:
            self.skipTest("imageio/imageio-ffmpeg not installed")
        import imageio.v2 as imageio_v2

        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "one.mp4")
            with imageio_v2.get_writer(src, fps=10, codec="libx264") as writer:
                writer.append_data(np.zeros((64, 64, 3), dtype="uint8"))  # type: ignore[attr-defined]
            out = os.path.join(td, "one_dreamed.mp4")
            preview = os.path.join(td, "preview.png")
            frames_dir = os.path.join(td, "frames")
            deepdream.dream_video_file(
                model=None,
                model_key="googlenet",
                video_in_path=src,
                video_out_path=out,
                targets=["inception3a"],
                iterations_per_frame=0,
                lr=0.01,
                jitter=0,
                blend_alpha=1.0,
                device="cpu",
                max_size=64,
                preview_file=preview,
                frames_dir=frames_dir,
            )
            self.assertTrue(os.path.isfile(out))
            self.assertTrue(os.path.isfile(preview))
            self.assertEqual(deepdream.read_video_frames_meta(frames_dir)["total_frames"], 1)

    def test_video_blend_weights_previous_dream(self):
        if not deepdream._VIDEO_AVAILABLE:
            self.skipTest("imageio/imageio-ffmpeg not installed")
        import imageio.v2 as imageio_v2

        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "two.mp4")
            with imageio_v2.get_writer(src, fps=10, codec="libx264") as writer:
                writer.append_data(np.zeros((64, 64, 3), dtype="uint8"))  # type: ignore[attr-defined]
                writer.append_data(np.full((64, 64, 3), 255, dtype="uint8"))  # type: ignore[attr-defined]
            out = os.path.join(td, "two_dreamed.mp4")
            deepdream.dream_video_file(
                model=None,
                model_key="googlenet",
                video_in_path=src,
                video_out_path=out,
                targets=["inception3a"],
                iterations_per_frame=0,
                lr=0.01,
                jitter=0,
                blend_alpha=0.25,
                device="cpu",
                max_size=64,
                crf=0,
            )
            reader = imageio_v2.get_reader(out)
            try:
                frames = [np.asarray(frame) for frame in reader]  # type: ignore[attr-defined]
            finally:
                reader.close()

        self.assertGreaterEqual(len(frames), 2)
        self.assertGreater(float(frames[1].mean()), 150.0)


class TestOfflineWeights(unittest.TestCase):
    """Tests that must run without network access or a warm weight cache."""

    model: Any

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.model = deepdream.build_googlenet("cpu", weights=False)

    def test_activations_have_the_documented_shape(self):
        acts = deepdream.run_to_layers(
            self.model, "googlenet", torch.randn(1, 3, 64, 64), {"inception4c"}
        )
        self.assertEqual(tuple(acts["inception4c"].shape), (1, 512, 4, 4))

    def test_gradient_ascent_changes_the_image_and_repeats(self):
        img = torch.randn(1, 3, 64, 64)
        first, loss_a = deepdream.gradient_ascent(
            self.model, "googlenet", img, ["inception4c"], 0.03, 0
        )
        self.assertTrue(torch.isfinite(first).all())
        self.assertGreater(float((first - img).abs().max()), 0.0)
        second, loss_b = deepdream.gradient_ascent(
            self.model, "googlenet", img, ["inception4c"], 0.03, 0
        )
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(loss_a, loss_b)

    def test_class_target_works_without_pretrained_weights(self):
        out, loss = deepdream.gradient_ascent(
            self.model, "googlenet", torch.randn(1, 3, 96, 96), [], 0.03, 0, target_class=130
        )
        self.assertTrue(torch.isfinite(out).all())
        self.assertIsInstance(loss, float)

    def test_build_model_registry_still_exposes_every_model(self):
        for key in deepdream.MODEL_REGISTRY:
            self.assertIn("build", deepdream.MODEL_REGISTRY[key])
            self.assertIn("forward_fn", deepdream.MODEL_REGISTRY[key])


class TestDreamNumerics(unittest.TestCase):
    """Checks that dreaming does what it claims, using real pretrained weights."""

    model: Any

    @classmethod
    def setUpClass(cls):
        if not deepdream.torch.cuda.is_available() and not _weights_cached():
            raise unittest.SkipTest("pretrained GoogLeNet weights are not available offline")
        cls.model = deepdream.build_model("googlenet", "cpu")

    def test_activation_loss_increases_over_iterations(self):
        # The point of gradient ascent: the targeted activation must go up, or
        # the run is not dreaming at all.
        img = torch.randn(1, 3, 96, 96)
        losses = []
        for _ in range(15):
            img, loss = deepdream.gradient_ascent(
                self.model, "googlenet", img, ["inception4c"], 0.03, 0
            )
            losses.append(loss)
        self.assertGreater(losses[-1], losses[0])
        self.assertTrue(all(loss > 0 for loss in losses))

    def test_output_stays_inside_the_normalised_pixel_range(self):
        img, _loss = deepdream.gradient_ascent(
            self.model, "googlenet", torch.randn(1, 3, 96, 96), ["inception4c"], 0.5, 0
        )
        for c, (mean, std) in enumerate(zip(deepdream.MEAN, deepdream.STD, strict=True)):
            self.assertGreaterEqual(float(img[:, c].min()), -mean / std - 1e-5)
            self.assertLessEqual(float(img[:, c].max()), (1 - mean) / std + 1e-5)

    def test_dream_changes_the_image_without_jitter(self):
        start = torch.randn(1, 3, 96, 96)
        out, _loss = deepdream.gradient_ascent(
            self.model, "googlenet", start, ["inception4c"], 0.03, 0
        )
        self.assertGreater(float((out - start).abs().max()), 0.0)


class TestResumeCompatibility(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.frames_dir = os.path.join(self.temp_dir, "seq")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _zoom(self, num_frames, size=64, resume=False, **kw):
        params: dict[str, Any] = {
            "model": None,
            "model_key": "googlenet",
            "img": torch.zeros(1, 3, size, size),
            "targets": ["inception4c"],
            "num_frames": num_frames,
            "iterations_per_frame": 0,
            "lr": 0.01,
            "jitter": 0,
            "zoom_scale": 1.02,
            "zoom_rot": 0.0,
            "device": "cpu",
            "out_path": None,
            "fps": 10,
            "frames_dir": self.frames_dir,
            "resume": resume,
        }
        params.update(kw)
        return deepdream.dream_zoom_animation(**params)

    def test_resume_mismatches_reports_changed_fields(self):
        saved = {"model": "googlenet", "layers": ["inception4c"], "width": 64, "zoom_scale": 1.02}
        self.assertEqual(
            deepdream.resume_mismatches(
                saved,
                {"model": "googlenet", "layers": ["inception4c"], "width": 64, "zoom_scale": 1.02},
            ),
            [],
        )
        problems = deepdream.resume_mismatches(
            saved, {"model": "vgg16", "layers": "inception4c", "width": 128, "zoom_scale": 1.03}
        )
        self.assertEqual(len(problems), 3)
        self.assertTrue(any(p.startswith("model:") for p in problems))
        self.assertTrue(any(p.startswith("width:") for p in problems))
        self.assertTrue(any(p.startswith("zoom_scale:") for p in problems))

    def test_resume_mismatches_treats_absent_optional_keys_as_none(self):
        saved = {"model": "googlenet"}
        self.assertEqual(deepdream.resume_mismatches(saved, {"target_class": None}), [])
        self.assertEqual(len(deepdream.resume_mismatches(saved, {"target_class": 130})), 1)
        self.assertEqual(deepdream.resume_mismatches({}, {"target_class": 130}), [])

    def test_zoom_resume_with_same_settings_continues(self):
        self._zoom(2)
        f1 = os.path.join(self.frames_dir, "frame_000001.png")
        mtime = os.path.getmtime(f1)
        self._zoom(3, resume=True)
        self.assertTrue(os.path.isfile(os.path.join(self.frames_dir, "frame_000003.png")))
        self.assertEqual(os.path.getmtime(f1), mtime)

    def test_zoom_resume_refuses_different_size_and_keeps_meta(self):
        self._zoom(2)
        meta_before = deepdream.read_video_frames_meta(self.frames_dir)
        err = io.StringIO()
        with self.assertRaises(SystemExit) as ctx, contextlib.redirect_stderr(err):
            self._zoom(3, size=96, resume=True)
        msg = str(ctx.exception.code)
        self.assertIn("cannot resume", msg)
        self.assertIn("width", msg)
        self.assertFalse(os.path.exists(os.path.join(self.frames_dir, "frame_000003.png")))
        self.assertEqual(deepdream.read_video_frames_meta(self.frames_dir), meta_before)

    def test_zoom_resume_refuses_different_zoom_params(self):
        self._zoom(2)
        with self.assertRaises(SystemExit) as ctx:
            self._zoom(3, resume=True, zoom_scale=1.05)
        self.assertIn("zoom_scale", str(ctx.exception.code))

    def test_zoom_resume_refuses_changed_dream_settings(self):
        self._zoom(2, seed=17)
        for key, value in (
            ("lr", 0.02),
            ("jitter", 1),
            ("zoom_cx", 0.6),
            ("zoom_cy", 0.4),
            ("seed", 18),
        ):
            with self.subTest(setting=key), self.assertRaises(SystemExit) as ctx:
                self._zoom(3, resume=True, **{key: value})
            self.assertIn(key, str(ctx.exception.code))

    def test_seeded_zoom_resume_continues_jitter_sequence(self):
        draws = []

        def fake_gradient_ascent(_model, _model_key, img, _targets, _lr, jitter, **_kw):
            draws.append((random.randint(-jitter, jitter), random.randint(-jitter, jitter)))
            return img, 0.0

        full_frames = os.path.join(self.temp_dir, "full")
        self.frames_dir = full_frames
        with mock.patch.object(deepdream, "gradient_ascent", side_effect=fake_gradient_ascent):
            self._zoom(3, iterations_per_frame=1, jitter=5, seed=123)
        expected_next_draw = draws[2]

        self.frames_dir = os.path.join(self.temp_dir, "resumed")
        with mock.patch.object(deepdream, "gradient_ascent", side_effect=fake_gradient_ascent):
            self._zoom(2, iterations_per_frame=1, jitter=5, seed=123)
            draws.clear()
            self._zoom(3, resume=True, iterations_per_frame=1, jitter=5, seed=123)
        self.assertEqual(draws, [expected_next_draw])

    def test_has_audio_stream_skips_still_images(self):
        # A zoom animation's source is a still image, so ffmpeg must not be
        # invoked to remux audio from it.
        with tempfile.TemporaryDirectory() as td:
            still = os.path.join(td, "photo.jpg")
            Image.new("RGB", (16, 16)).save(still)
            self.assertFalse(deepdream.has_audio_stream(still))
            self.assertFalse(deepdream.has_audio_stream(os.path.join(td, "missing.mp4")))
            self.assertFalse(deepdream.has_audio_stream(None))

    def test_zoom_encode_frames_skips_remux_for_still_source(self):
        with tempfile.TemporaryDirectory() as td:
            fd = os.path.join(td, "seq")
            os.makedirs(fd)
            for i in (1, 2):
                Image.new("RGB", (32, 32)).save(os.path.join(fd, f"frame_{i:06d}.png"))
            src = os.path.join(td, "photo.jpg")
            Image.new("RGB", (32, 32)).save(src)
            deepdream.write_video_frames_meta(
                fd, fps=10, mode="zoom", source=src, width=32, height=32
            )
            out = os.path.join(td, "out.mp4")
            with mock.patch.object(deepdream, "_remux_audio") as mremux:
                deepdream.compile_video_frames(fd, out, fps=10, crf=30)
            mremux.assert_not_called()
            self.assertTrue(os.path.isfile(out))

    def test_zoom_resume_checks_frame_size_without_meta(self):
        self._zoom(2)
        os.remove(os.path.join(self.frames_dir, "meta.json"))
        with self.assertRaises(SystemExit) as ctx:
            self._zoom(3, size=96, resume=True)
        self.assertIn("frame_000001.png is 64x64", str(ctx.exception.code))

    def test_video_resume_refuses_different_size(self):
        if not deepdream._VIDEO_AVAILABLE:
            self.skipTest("imageio/imageio-ffmpeg not installed")
        import imageio.v2 as imageio_v2

        src = os.path.join(self.temp_dir, "clip.mp4")
        with imageio_v2.get_writer(src, fps=10, codec="libx264") as writer:
            for _ in range(3):
                writer.append_data(np.zeros((64, 64, 3), dtype="uint8"))  # type: ignore[attr-defined]
        common: dict[str, Any] = {
            "model": None,
            "model_key": "googlenet",
            "video_in_path": src,
            "video_out_path": None,
            "targets": ["inception3a"],
            "iterations_per_frame": 0,
            "lr": 0.01,
            "jitter": 0,
            "blend_alpha": 0.5,
            "device": "cpu",
            "frames_dir": self.frames_dir,
        }
        deepdream.dream_video_file(max_size=64, **common)
        os.remove(os.path.join(self.frames_dir, "frame_000003.png"))
        with self.assertRaises(SystemExit) as ctx:
            deepdream.dream_video_file(max_size=32, resume=True, **common)
        self.assertIn("cannot resume", str(ctx.exception.code))
        deepdream.dream_video_file(max_size=64, resume=True, **common)
        self.assertTrue(os.path.isfile(os.path.join(self.frames_dir, "frame_000003.png")))

    def test_video_resume_refuses_changed_dream_settings(self):
        if not deepdream._VIDEO_AVAILABLE:
            self.skipTest("imageio/imageio-ffmpeg not installed")
        import imageio.v2 as imageio_v2

        src = os.path.join(self.temp_dir, "clip_settings.mp4")
        with imageio_v2.get_writer(src, fps=10, codec="libx264") as writer:
            for _ in range(3):
                writer.append_data(np.zeros((64, 64, 3), dtype="uint8"))  # type: ignore[attr-defined]
        common: dict[str, Any] = {
            "model": None,
            "model_key": "googlenet",
            "video_in_path": src,
            "video_out_path": None,
            "targets": ["inception3a"],
            "iterations_per_frame": 0,
            "lr": 0.01,
            "jitter": 0,
            "blend_alpha": 0.5,
            "device": "cpu",
            "frames_dir": self.frames_dir,
            "seed": 17,
        }
        deepdream.dream_video_file(max_size=64, **common)
        for key, value in (("lr", 0.02), ("jitter", 1), ("seed", 18)):
            changed = {**common, key: value}
            with self.subTest(setting=key), self.assertRaises(SystemExit) as ctx:
                deepdream.dream_video_file(max_size=64, resume=True, **changed)
            self.assertIn(key, str(ctx.exception.code))


class TestOutOfMemoryHandling(unittest.TestCase):
    def test_is_oom_error(self):
        self.assertTrue(deepdream._is_oom_error(RuntimeError("CUDA out of memory. Tried ...")))
        self.assertFalse(deepdream._is_oom_error(RuntimeError("Gradient was NaN/Inf")))
        self.assertFalse(deepdream._is_oom_error(ValueError("out of memory")))
        oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
        if isinstance(oom_type, type):
            self.assertTrue(deepdream._is_oom_error(oom_type("boom")))

    def test_oom_message_suggests_smaller_size(self):
        msg = deepdream._oom_message("cuda", "vgg16", 1600)
        self.assertIn("CUDA ran out of memory", msg)
        self.assertIn("--size 1184", msg)
        self.assertIn("--cpu", msg)
        generic = deepdream._oom_message("cuda", "vgg16", None)
        self.assertIn("--size", generic)

    def test_min_torch_version_matches_requirements(self):
        # The runtime check and the installer's floor must not drift apart.
        req = Path(__file__).resolve().parent.parent / "requirements.txt"
        floors = [
            line.split(">=")[1].strip()
            for line in req.read_text(encoding="utf-8").splitlines()
            if line.strip().startswith("torch>=")
        ]
        self.assertEqual(len(floors), 1, "expected exactly one torch floor")
        self.assertEqual(deepdream.MIN_TORCH_VERSION, deepdream._parse_version(floors[0]))

    def test_remux_failure_warning_includes_ffmpeg_message(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(err):
            deepdream._warn_remux_failure(
                mock.Mock(stderr="frame= 1\n[flv] moov atom not found\n", returncode=1)
            )
        self.assertIn("no audio track", err.getvalue())
        self.assertIn("moov atom not found", err.getvalue())

    def test_remux_failure_warning_without_stderr(self):
        err = io.StringIO()
        with contextlib.redirect_stdout(err):
            deepdream._warn_remux_failure(mock.Mock(stderr="", returncode=1))
        self.assertIn("no audio track", err.getvalue())

    def test_main_reports_oom_with_hint(self):
        with tempfile.TemporaryDirectory() as td:
            img_path = os.path.join(td, "img.jpg")
            Image.new("RGB", (100, 80)).save(img_path)
            with (
                mock.patch.object(deepdream, "build_model", return_value=None),
                mock.patch.object(
                    deepdream,
                    "deepdream",
                    side_effect=RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"),
                ),
            ):
                err = io.StringIO()
                with contextlib.redirect_stderr(err):
                    with self.assertRaises(SystemExit) as ctx:
                        deepdream.main(
                            [img_path, "--iterations", "1", "--octaves", "1", "--out", td]
                        )
                self.assertEqual(ctx.exception.code, 1)
                self.assertIn("ran out of memory", err.getvalue())
                self.assertIn("--size", err.getvalue())
                self.assertNotIn("Traceback", err.getvalue())


class DeviceFallbackTests(unittest.TestCase):
    """A detected accelerator that cannot start must not end the run."""

    def _args(self):
        return argparse.Namespace(no_cuda=False, guide=None, size=512)

    def test_falls_back_to_cpu_when_accelerator_fails_to_start(self):
        args = self._args()
        attempted: list[str] = []

        def build(_key, device):
            attempted.append(device)
            if device == "cuda":
                raise RuntimeError("CUDA error: no kernel image is available")
            return "model-on-cpu"

        with (
            mock.patch.object(deepdream, "build_model", side_effect=build),
            mock.patch.object(deepdream.torch.cuda, "is_available", return_value=True),
            mock.patch.object(deepdream.torch.cuda, "get_device_name", return_value="Fake GPU"),
        ):
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                device, model, guide = deepdream._init_device_and_model(
                    args, "googlenet", ["inception4c"]
                )
        self.assertEqual(device, "cpu")
        self.assertEqual(model, "model-on-cpu")
        self.assertIsNone(guide)
        self.assertEqual(attempted, ["cuda", "cpu"])
        self.assertIn("Falling back to the CPU", err.getvalue())
        self.assertNotIn("Traceback", err.getvalue())

    def test_exits_after_one_cpu_retry(self):
        args = self._args()
        attempted: list[str] = []

        def build(_key, device):
            attempted.append(device)
            raise RuntimeError("no kernel image is available")

        with (
            mock.patch.object(deepdream, "build_model", side_effect=build),
            mock.patch.object(deepdream.torch.cuda, "is_available", return_value=True),
            mock.patch.object(deepdream.torch.cuda, "get_device_name", return_value="Fake GPU"),
        ):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as ctx:
                    deepdream._init_device_and_model(args, "googlenet", ["inception4c"])
        self.assertEqual(ctx.exception.code, 1)
        # Exactly one fallback attempt: never loop.
        self.assertEqual(attempted, ["cuda", "cpu"])

    def test_cpu_only_run_is_not_retried(self):
        """With no accelerator to blame, the original error must still surface."""
        args = self._args()
        attempted: list[str] = []

        def build(_key, device):
            attempted.append(device)
            raise RuntimeError("Model file not found")

        mps = SimpleNamespace(is_available=lambda: False)
        with (
            mock.patch.object(deepdream, "build_model", side_effect=build),
            mock.patch.object(deepdream.torch.cuda, "is_available", return_value=False),
            mock.patch.object(deepdream.torch.backends, "mps", mps, create=True),
        ):
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                with self.assertRaises(SystemExit) as ctx:
                    deepdream._init_device_and_model(args, "googlenet", ["inception4c"])
        self.assertEqual(ctx.exception.code, 1)
        self.assertEqual(attempted, ["cpu"])

    def test_forced_cpu_flag_skips_accelerator_entirely(self):
        args = argparse.Namespace(no_cuda=True, guide=None, size=512)
        with (
            mock.patch.object(deepdream, "build_model", return_value="model") as build,
            mock.patch.object(deepdream.torch.cuda, "is_available", return_value=True),
        ):
            device, model, _ = deepdream._init_device_and_model(args, "googlenet", ["inception4c"])
        self.assertEqual(device, "cpu")
        build.assert_called_once_with("googlenet", "cpu")

    def test_release_device_memory_never_raises(self):
        """Cleanup must not replace a real error with a backend complaint.

        torch.mps.empty_cache() raises when the MPS backend is not usable, which
        is reachable exactly when we are falling back away from a broken device.
        """
        with mock.patch.object(
            deepdream.torch.mps,
            "empty_cache",
            side_effect=RuntimeError("Cannot execute emptyCache() without MPS backend."),
        ):
            deepdream._release_device_memory("mps")  # must not raise
        deepdream._release_device_memory("cuda")


class RemuxResilienceTests(unittest.TestCase):
    """A missing ffmpeg binary must warn, not traceback."""

    def test_missing_ffmpeg_binary_warns_instead_of_crashing(self):
        with tempfile.TemporaryDirectory() as td:
            silent = os.path.join(td, "silent.mp4")
            source = os.path.join(td, "source.mp4")
            final = os.path.join(td, "final.mp4")
            for p in (silent, source):
                Path(p).write_bytes(b"not-really-a-video")
            with mock.patch.object(
                deepdream.imageio_ffmpeg,
                "get_ffmpeg_exe",
                side_effect=RuntimeError("No ffmpeg exe could be found."),
            ):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    deepdream._remux_audio(silent, source, final)
            self.assertIn("no audio track", out.getvalue())
            self.assertNotIn("Traceback", out.getvalue())
            # The silent track must survive as the deliverable.
            self.assertTrue(os.path.isfile(final))


class TestPathLengthLimits(unittest.TestCase):
    """The pre-flight path check must use the real per-OS limit."""

    def test_limit_matches_platform(self):
        if sys.platform == "win32":
            self.assertEqual(deepdream.MAX_PATH, 260)
        elif sys.platform == "darwin":
            # macOS PATH_MAX is 1024, not the POSIX-typical 4096.
            self.assertEqual(deepdream.MAX_PATH, 1024)
        else:
            self.assertEqual(deepdream.MAX_PATH, 4096)

    def test_limit_is_not_above_the_platform_maximum(self):
        ceiling = {"win32": 260, "darwin": 1024}.get(sys.platform, 4096)
        self.assertLessEqual(deepdream.MAX_PATH, ceiling)


if __name__ == "__main__":
    unittest.main()
