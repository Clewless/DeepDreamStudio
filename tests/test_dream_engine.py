"""Numerical regression checks using a deterministic, download-free model."""

import random
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from app import deepdream as cli
from app import dream_engine
from app.media_io import tensor_to_np
from app.protocol import EV_ITERATION, EV_OCTAVE, EV_PREVIEW, parse_event


class NumericalModel(nn.Module):
    """Fixed spatial features and classifier, independent of model weights/RNG."""

    def __init__(self) -> None:
        super().__init__()
        conv = nn.Conv2d(3, 2, 3, padding=1, bias=False, dtype=torch.float64)
        with torch.no_grad():
            conv.weight.copy_(
                torch.linspace(-0.2, 0.3, 54, dtype=torch.float64).reshape(2, 3, 3, 3)
            )
        self.features = nn.Sequential(conv, nn.Tanh())
        self.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.features(x).mean(dim=(2, 3)))


def numerical_input() -> torch.Tensor:
    return torch.linspace(-0.7, 0.9, 3 * 48 * 64, dtype=torch.float64).reshape(1, 3, 48, 64)


@pytest.fixture(autouse=True)
def preserve_random_streams() -> Iterator[None]:
    python_state = random.getstate()
    torch_state = torch.get_rng_state()
    yield
    random.setstate(python_state)
    torch.set_rng_state(torch_state)


# Captured BEFORE extraction from the previous CLI engine with this fixed model,
# float64 input, seed=731, two octaves, two iterations, and jitter=3. These values
# are an independent regression oracle, not recomputed by the implementation.
BASELINES = {
    "layer": (
        [
            -0.7093822562932369,
            -0.583118211251976,
            -0.38068317806161106,
            -0.17627615796258148,
            -0.13614078694139808,
            0.08561917716074066,
            0.30812204346849326,
            0.4966512106889583,
            0.5278587551028857,
            0.7684029787100793,
            1.0088214231491572,
            1.0505142009662611,
        ],
        1742.570690217563,
        3017.610687630292,
    ),
    "guide": (
        [
            -0.5461649382342492,
            -0.3620389682188407,
            -0.33324428137330603,
            -0.2646022993682785,
            0.15138008739078174,
            0.1553296167242404,
            0.20079366219629102,
            0.30851023956927864,
            0.6746800424279693,
            0.6699078886910436,
            0.7581720839959516,
            0.9373240551787834,
        ],
        1826.8872949286663,
        2103.5072498620398,
    ),
    "class": (
        [
            -0.6272034473876203,
            -0.4753514169099313,
            -0.37333418976795113,
            -0.25199613664534676,
            0.11841731607060368,
            0.14862354770594655,
            0.22093786851681146,
            0.3428715088153489,
            0.7399302157381666,
            0.7457425575049613,
            0.8033075421928118,
            1.0044624928460466,
        ],
        1843.4226623787533,
        2528.9449897172917,
    ),
    "combined": (
        [
            -0.6648315146728733,
            -0.5212369015876921,
            -0.380213249899202,
            -0.21829067775017444,
            0.025897118040587974,
            0.12248557791818017,
            0.2586909920139915,
            0.41656575965049836,
            0.6719780506382589,
            0.7562654901163426,
            0.898496311358814,
            1.0322904505214596,
        ],
        1843.307261441574,
        2744.5235870555116,
    ),
}


@pytest.mark.parametrize("mode", BASELINES)
@pytest.mark.parametrize("adapter", [False, True], ids=["core", "cli-adapter"])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
        ),
    ],
)
def test_seeded_multioctave_results_match_pre_extraction_baseline(
    mode: str, adapter: bool, device: str
) -> None:
    model = NumericalModel().to(device)
    img = numerical_input().requires_grad_(True)
    original = img.detach().clone()
    guide = (
        {
            "features.0": torch.linspace(-0.5, 0.8, 14, dtype=torch.float64, device=device).reshape(
                2, 7
            )
        }
        if mode == "guide"
        else None
    )
    targets = [] if mode == "class" else ["features.0"]
    target_class = 1 if mode in ("class", "combined") else None
    random.seed(731)
    run = cli.deepdream if adapter else dream_engine.deepdream
    out = run(
        model,
        "vgg16",
        img,
        targets,
        2,
        0.025,
        1.4,
        2,
        3,
        device,
        guide_features=guide,
        target_class=target_class,
    )
    samples, total, square_total = BASELINES[mode]
    flat = out.flatten()
    actual_samples = flat[torch.linspace(0, flat.numel() - 1, 12).long()]
    torch.testing.assert_close(
        actual_samples,
        torch.tensor(samples, dtype=torch.float64, device=device),
        rtol=1e-8,
        atol=1e-9,
    )
    assert out.sum().item() == pytest.approx(total, rel=1e-8, abs=1e-8)
    assert out.square().sum().item() == pytest.approx(square_total, rel=1e-8, abs=1e-8)
    assert random.randint(-1000, 1000) == -94
    torch.testing.assert_close(img.detach(), original, rtol=0, atol=0)
    assert img.grad is None
    assert not out.requires_grad
    assert out.device.type == device
    assert tuple(out.shape) == tuple(img.shape)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(not module._forward_hooks for module in model.modules())


def test_ascent_matches_analytic_identity_gradient_with_jitter() -> None:
    model = nn.Module()
    model.features = nn.Sequential(nn.Identity())
    img = numerical_input()
    # For 0.5*sum(x**2), the derivative is exactly x. Spatial rolls preserve
    # its mean magnitude, and undoing the roll must recover this same update.
    expected = img + 0.025 * img / (img.abs().mean() + 1e-8)
    out, loss = dream_engine.gradient_ascent(model, "vgg16", img, ["features.0"], 0.025, 3)
    torch.testing.assert_close(out, expected, rtol=1e-12, atol=1e-12)
    assert loss == pytest.approx(0.5 * img.square().sum().item(), rel=1e-12)


def test_octaves_carry_detail_and_observers_run_in_order(
    capsys: pytest.CaptureFixture[str],
) -> None:
    img = numerical_input()
    events: list[tuple[str, int, int]] = []
    callbacks = dream_engine.DreamCallbacks(
        start=lambda: events.append(("start", 0, 0)),
        octave=lambda index, total, current: events.append(("octave", index, total)),
        iteration=lambda octave, index, total, current, loss: events.append(
            ("iteration", octave, index)
        ),
        octave_end=lambda current: events.append(("end", current.shape[2], current.shape[3])),
    )
    out = dream_engine.deepdream(
        None,
        "vgg16",
        img,
        ["features.0"],
        2,
        0.025,
        1.4,
        2,
        0,
        "cpu",
        callbacks=callbacks,
        step=lambda current: (current + 0.125, 0.0),
    )
    torch.testing.assert_close(out, img + 0.5, rtol=1e-12, atol=1e-12)
    assert events == [
        ("start", 0, 0),
        ("octave", 1, 2),
        ("iteration", 1, 1),
        ("iteration", 1, 2),
        ("end", 34, 46),
        ("octave", 2, 2),
        ("iteration", 2, 1),
        ("iteration", 2, 2),
        ("end", 48, 64),
    ]
    assert capsys.readouterr().out == ""


def test_validation_precedes_start_observer() -> None:
    started: list[bool] = []
    with pytest.raises(SystemExit, match="octave_scale must be >"):
        dream_engine.deepdream(
            None,
            "vgg16",
            numerical_input(),
            ["features.0"],
            2,
            0.025,
            1.0,
            2,
            0,
            "cpu",
            callbacks=dream_engine.DreamCallbacks(start=lambda: started.append(True)),
        )
    assert not started


def test_cli_progress_and_frame_output_follow_engine_callbacks(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    frames = tmp_path / "frames"
    preview = tmp_path / "preview.png"
    random.seed(731)
    out = cli.deepdream(
        NumericalModel(),
        "vgg16",
        numerical_input(),
        ["features.0"],
        2,
        0.025,
        1.4,
        2,
        3,
        "cpu",
        frames_dir=frames,
        preview_file=preview,
    )
    assert {path.name for path in frames.iterdir()} == {
        "o1_i1.png",
        "o1_i2.png",
        "o2_i1.png",
        "o2_i2.png",
    }
    with Image.open(frames / "o1_i2.png") as image:
        assert image.size == (46, 34)
    expected = tensor_to_np(out)
    for path in (frames / "o2_i2.png", preview):
        with Image.open(path) as image:
            np.testing.assert_array_equal(np.asarray(image), expected)
    events = [parse_event(line) for line in capsys.readouterr().out.splitlines()]
    structured = [event for event in events if event is not None]
    assert [event["event"] for event in structured] == [
        EV_OCTAVE,
        EV_ITERATION,
        EV_ITERATION,
        EV_PREVIEW,
        EV_OCTAVE,
        EV_ITERATION,
        EV_ITERATION,
        EV_PREVIEW,
    ]
    assert all(
        event["path"] == str(preview) for event in structured if event["event"] == EV_PREVIEW
    )


def test_invalid_run_cannot_clear_existing_frames(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    existing = frames / "o1_i1.png"
    existing.write_bytes(b"previous result")
    with pytest.raises(SystemExit, match="lr must be >"):
        cli.deepdream(
            NumericalModel(),
            "vgg16",
            numerical_input(),
            ["features.0"],
            2,
            -0.025,
            1.4,
            2,
            0,
            "cpu",
            frames_dir=frames,
            clear_frames=True,
        )
    assert existing.read_bytes() == b"previous result"


def test_observer_failure_stops_before_another_ascent_step() -> None:
    steps: list[int] = []

    def step(current: torch.Tensor) -> tuple[torch.Tensor, float]:
        steps.append(1)
        return current + 0.125, 0.0

    def fail(oi: int, it: int, total: int, current: torch.Tensor, loss: float) -> None:
        raise OSError("frame write failed")

    with pytest.raises(OSError, match="frame write failed"):
        dream_engine.deepdream(
            None,
            "vgg16",
            numerical_input(),
            ["features.0"],
            2,
            0.025,
            1.4,
            2,
            0,
            "cpu",
            step=step,
            callbacks=dream_engine.DreamCallbacks(iteration=fail),
        )
    assert steps == [1]
