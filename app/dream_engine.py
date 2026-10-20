"""Numerical DeepDream engine, independent of logging and media I/O.

Callbacks observe completed steps synchronously. The pyramid and carried detail
stay on CPU; only the current octave is moved to the requested device.
"""

import math
import random
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .media_io import MEAN, STD
from .models import (
    MODEL_REGISTRY,
    Activations,
    Device,
    forward_with_activations,
    run_to_layers,
)

MAX_OCTAVES = 25
MIN_OCTAVE_SIZE = 32
TARGET_BASE_SIZE = 250

# Each matching matrix is at most 1024 x 1024 (4 MiB for float32).
GUIDE_MATCH_BLOCK_SIZE = 1024

GuideFeatures = Mapping[str, Tensor]
LayerCapture = Callable[[nn.Module, str, Tensor, Collection[str]], Activations]
FullCapture = Callable[[nn.Module, Tensor, Collection[str]], tuple[Activations, Tensor]]
AscentStep = Callable[[Tensor], tuple[Tensor, float]]


@dataclass(frozen=True)
class DreamCallbacks:
    """Optional synchronous observers; exceptions propagate to the caller.

    start runs after validation, before pyramid construction. Iteration receives
    the one-based octave, iteration, total iterations, current image, and loss.
    Observers must not modify tensors or consume the engine's random stream.
    """

    start: Callable[[], None] | None = None
    note: Callable[[str], None] | None = None
    octave: Callable[[int, int, Tensor], None] | None = None
    iteration: Callable[[int, int, int, Tensor, float], None] | None = None
    octave_end: Callable[[Tensor], None] | None = None


def _validate_guide_features(guide_features: GuideFeatures | None, layers: Collection[str]) -> None:
    """Check matching guide tensors once before an ascent loop starts."""
    if not guide_features or not layers:
        return
    with torch.no_grad():
        for name in dict.fromkeys(layers):
            if name in guide_features and not bool(torch.isfinite(guide_features[name]).all()):
                raise ValueError("Guide features must contain only finite values.")


def _guide_match_loss(features: Tensor, guide: Tensor) -> Tensor:
    """Sum exact nearest-guide similarities without a spatial all-pairs graph.

    Select winners without autograd, then recompute only their dot products.
    Max's derivative is the winning dot product's derivative; strict updates
    retain the first guide index on ties, just like Tensor.max(dim=1).
    """
    if guide.shape[1] == 0:
        raise ValueError("Guide features must contain at least one spatial position.")
    block_size = GUIDE_MATCH_BLOCK_SIZE
    with torch.no_grad():
        if not bool(torch.isfinite(guide).all()):
            raise ValueError("Guide features must contain only finite values.")
        winners = torch.empty(features.shape[1], dtype=torch.long, device=features.device)
        for start in range(0, features.shape[1], block_size):
            query = features[:, start : start + block_size].t()
            best_values = torch.full(
                (query.shape[0],), -torch.inf, dtype=features.dtype, device=features.device
            )
            best_indices = torch.zeros(query.shape[0], dtype=torch.long, device=features.device)
            for guide_start in range(0, guide.shape[1], block_size):
                similarities = torch.mm(query, guide[:, guide_start : guide_start + block_size])
                values, indices = similarities.max(dim=1)
                replace = values > best_values
                best_values = torch.where(replace, values, best_values)
                best_indices = torch.where(replace, indices + guide_start, best_indices)
            winners[start : start + query.shape[0]] = best_indices
    return (features * guide.index_select(1, winners)).sum()


def activation_loss(acts: Mapping[str, Tensor], guide_features: GuideFeatures | None) -> Tensor:
    """Feature amplification or nearest-guide-feature cosine similarity."""
    if not acts:
        raise RuntimeError("No activations captured: no target layers were run.")
    if guide_features:
        total = None
        for name, a in acts.items():
            if name not in guide_features:
                continue
            fx = a[0].reshape(a.shape[1], -1)
            fx_norm = fx / (fx.norm(p=2, dim=0, keepdim=True) + 1e-7)
            matched_loss = _guide_match_loss(fx_norm, guide_features[name])
            total = matched_loss if total is None else total + matched_loss
        if total is not None:
            return total
        raise RuntimeError(
            "Guide features were supplied but none overlap the captured "
            f"layer(s) {sorted(acts)} (guide feature layers: {sorted(guide_features)}). "
            "A guide dream would otherwise run silently with no guide effect."
        )
    return cast(Tensor, 0.5 * sum((a**2).sum() for a in acts.values()))


def resize(t: Tensor, size: Sequence[int]) -> Tensor:
    return F.interpolate(t, size=list(size), mode="bilinear", align_corners=False, antialias=True)


def gradient_ascent(
    model: nn.Module,
    model_key: str,
    img: Tensor,
    targets: Collection[str],
    lr: float,
    jitter: int,
    guide_features: GuideFeatures | None = None,
    target_class: int | None = None,
    *,
    layer_capture: LayerCapture = run_to_layers,
    full_capture: FullCapture = forward_with_activations,
) -> tuple[Tensor, float]:
    """Perform one detached, normalized ascent step without mutating the input."""
    if not targets and target_class is None:
        raise SystemExit("Error: No layer targets or class target specified.")
    sx = sy = 0
    if jitter:
        sx, sy = random.randint(-jitter, jitter), random.randint(-jitter, jitter)
        img = torch.roll(img, shifts=(sx, sy), dims=(2, 3))

    img = img.detach().clone().requires_grad_(True)
    loss = torch.tensor(0.0, device=img.device)
    grad: Tensor | None = None
    if targets and target_class is not None:
        # One full forward captures features and logits. Normalize each gradient
        # separately so the much larger activation sum cannot drown out the class.
        acts, logits = full_capture(model, img, targets)
        layer_loss = activation_loss(acts, guide_features)
        class_loss = logits[0, target_class]
        grad_layer = torch.autograd.grad(layer_loss, img, retain_graph=True)[0]
        grad_class = torch.autograd.grad(class_loss, img)[0]
        grad = grad_layer / (grad_layer.abs().mean() + 1e-8) + grad_class / (
            grad_class.abs().mean() + 1e-8
        )
        loss = loss + layer_loss.detach() + class_loss.detach()
    elif targets:
        acts = layer_capture(model, model_key, img, targets)
        loss = loss + activation_loss(acts, guide_features)
        loss.backward()
        grad = img.grad
    elif target_class is not None:
        logits = model(img)
        loss = loss + logits[0, target_class]
        loss.backward()
        grad = img.grad

    with torch.no_grad():
        if grad is None:
            raise RuntimeError(
                "Gradient was None during ascent: the computational "
                "graph is disconnected (no path from loss to input)."
            )
        if not bool(torch.isfinite(grad).all()):
            raise RuntimeError("Gradient was NaN/Inf during ascent: try a lower --lr.")
        grad = grad / (grad.abs().mean() + 1e-8)
        img = img + lr * grad
        for c in range(3):
            img[:, c] = img[:, c].clamp(-MEAN[c] / STD[c], (1 - MEAN[c]) / STD[c])
        img = img.detach()
        if jitter:
            img = torch.roll(img, shifts=(-sx, -sy), dims=(2, 3))
    return img, float(loss.detach())


def auto_octave_count(
    min_dim: int, octave_scale: float, target_base_size: int = TARGET_BASE_SIZE
) -> int:
    """Auto-mode octave count before the MAX_OCTAVES cap is applied."""
    if not math.isfinite(octave_scale) or octave_scale <= 1.0:
        raise ValueError(f"octave_scale must be > 1.0, got {octave_scale}")
    if min_dim > target_base_size:
        return max(1, round(math.log(min_dim / target_base_size, octave_scale)) + 1)
    return 1


def plan_octaves(
    h: int,
    w: int,
    num_octaves: int,
    octave_scale: float,
    min_octave_size: int = MIN_OCTAVE_SIZE,
    target_base_size: int = TARGET_BASE_SIZE,
) -> int:
    """Predict the number of octaves that fit after scale and count caps."""
    if not math.isfinite(octave_scale) or octave_scale <= 1.0:
        raise ValueError(f"octave_scale must be > 1.0, got {octave_scale}")
    if num_octaves == 0:
        num_octaves = auto_octave_count(min(h, w), octave_scale, target_base_size)
    num_octaves = min(num_octaves, MAX_OCTAVES)
    count = 1
    hh, ww = h, w
    for _ in range(num_octaves - 1):
        hh = round(hh / octave_scale)
        ww = round(ww / octave_scale)
        if min(hh, ww) < min_octave_size:
            break
        count += 1
    return count


def octave_truncation_note(
    requested: int, actual: int, octave_scale: float, auto_capped: bool = False
) -> str | None:
    """Explain truncation while preserving the CLI's existing messages."""
    if auto_capped:
        if actual < MAX_OCTAVES:
            return (
                f"auto octaves would exceed the {MAX_OCTAVES}-octave cap and the image "
                f"can't fit {MAX_OCTAVES} octaves at octave-scale {octave_scale}; "
                f"running {actual} instead."
            )
        return (
            f"auto octaves capped at {MAX_OCTAVES}: octave-scale {octave_scale} is "
            f"so close to 1.0 that reaching the ~{TARGET_BASE_SIZE}px base octave "
            f"would take more than {MAX_OCTAVES} levels; running {actual} instead."
        )
    if requested <= 0 or actual >= requested:
        return None
    cap_bound = requested > MAX_OCTAVES
    size_bound = actual < min(requested, MAX_OCTAVES)
    if cap_bound and size_bound:
        return (
            f"--octaves {requested} exceeds the {MAX_OCTAVES}-octave cap and the "
            f"image can't fit {MAX_OCTAVES} octaves at octave-scale {octave_scale}; "
            f"running {actual} instead."
        )
    if cap_bound:
        return f"--octaves {requested} capped at {MAX_OCTAVES} octaves; running {actual} instead."
    return (
        f"image too small for {requested} octaves at octave-scale {octave_scale}; "
        f"running {actual} instead."
    )


def validate_input_size(
    h: int, w: int, model_key: str, what: str = "Input image", hint: str | None = None
) -> None:
    if model_key not in MODEL_REGISTRY:
        raise SystemExit(f"Unknown model '{model_key}'. Valid models: {sorted(MODEL_REGISTRY)}")
    need = MODEL_REGISTRY[model_key]["min_size"]
    if need and min(h, w) < need:
        msg = (
            f"Error: {what} is too small ({w}x{h}); model '{model_key}' "
            f"needs at least {need}px on the short edge."
        )
        if hint:
            msg += f" {hint}"
        raise SystemExit(msg)


def deepdream(
    model: nn.Module | None,
    model_key: str,
    img: Tensor,
    targets: Collection[str],
    iterations: int,
    lr: float,
    octave_scale: float,
    num_octaves: int,
    jitter: int,
    device: Device,
    guide_features: GuideFeatures | None = None,
    target_class: int | None = None,
    *,
    callbacks: DreamCallbacks | None = None,
    step: AscentStep | None = None,
) -> Tensor:
    """Dream a CPU-backed octave pyramid; callers own progress and file output.

    A model is required for ascent; zero-iteration runs may supply None.
    """
    if not targets and target_class is None:
        raise SystemExit("Error: No layer targets or class target specified.")
    if not math.isfinite(octave_scale):
        raise SystemExit(f"Error: octave_scale must be finite, got {octave_scale}")
    if octave_scale <= 1.0:
        raise SystemExit(f"Error: octave_scale must be > 1.0, got {octave_scale}")
    if num_octaves < 0:
        raise SystemExit(f"Error: num_octaves must be >= 0, got {num_octaves}")
    if iterations < 0:
        raise SystemExit(f"Error: iterations must be >= 0, got {iterations}")
    if not math.isfinite(lr):
        raise SystemExit(f"Error: lr must be finite, got {lr}")
    if lr <= 0:
        raise SystemExit(f"Error: lr must be > 0, got {lr}")
    if jitter < 0:
        raise SystemExit(f"Error: jitter must be >= 0, got {jitter}")
    validate_input_size(img.shape[2], img.shape[3], model_key)
    if iterations > 0:
        _validate_guide_features(guide_features, targets)

    observer = callbacks if callbacks is not None else DreamCallbacks()
    if observer.start is not None:
        observer.start()

    def default_step(current: Tensor) -> tuple[Tensor, float]:
        return gradient_ascent(
            cast(nn.Module, model),
            model_key,
            current,
            targets,
            lr,
            jitter,
            guide_features=guide_features,
            target_class=target_class,
        )

    ascent = step if step is not None else default_step
    requested_octaves = num_octaves
    model_spec = MODEL_REGISTRY.get(model_key)
    model_min = model_spec["min_size"] if model_spec is not None else MIN_OCTAVE_SIZE
    effective_min_octave = max(MIN_OCTAVE_SIZE, model_min)
    effective_target_base = max(TARGET_BASE_SIZE, model_min)
    num_octaves = plan_octaves(
        img.shape[2],
        img.shape[3],
        num_octaves,
        octave_scale,
        min_octave_size=effective_min_octave,
        target_base_size=effective_target_base,
    )
    auto_capped = (
        requested_octaves == 0
        and auto_octave_count(
            min(img.shape[2], img.shape[3]),
            octave_scale,
            target_base_size=effective_target_base,
        )
        > MAX_OCTAVES
    )
    base_cpu = img.detach().to("cpu")
    octaves_cpu = [base_cpu]
    for _ in range(num_octaves - 1):
        cur = octaves_cpu[-1]
        h_scaled = round(cur.shape[2] / octave_scale)
        w_scaled = round(cur.shape[3] / octave_scale)
        if min(h_scaled, w_scaled) < effective_min_octave:
            break
        octaves_cpu.append(resize(cur, [h_scaled, w_scaled]))
    octaves_cpu.reverse()
    note = octave_truncation_note(
        requested_octaves, len(octaves_cpu), octave_scale, auto_capped=auto_capped
    )
    if note and observer.note is not None:
        observer.note(note)

    detail_cpu = torch.zeros_like(octaves_cpu[0])
    img_w = None
    for oi, octave_cpu in enumerate(octaves_cpu):
        if detail_cpu.shape[2:] != octave_cpu.shape[2:]:
            detail_cpu = resize(detail_cpu, [octave_cpu.shape[2], octave_cpu.shape[3]])
        octave = octave_cpu.to(device)
        detail_d = detail_cpu.to(device)
        img_w = (octave + detail_d).detach()
        del detail_d
        if observer.octave is not None:
            observer.octave(oi + 1, len(octaves_cpu), img_w)
        for it in range(iterations):
            img_w, loss = ascent(img_w)
            if observer.iteration is not None:
                observer.iteration(oi + 1, it + 1, iterations, img_w, loss)
        detail_cpu = (img_w - octave).detach().to("cpu")
        del octave
        if observer.octave_end is not None:
            observer.octave_end(img_w)
        if oi < len(octaves_cpu) - 1:
            # Release the current device tensor before allocating the next octave.
            del img_w
            img_w = None
    if img_w is None:
        img_w = base_cpu.to(device)
    return img_w
