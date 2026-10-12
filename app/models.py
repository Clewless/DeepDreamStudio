"""PyTorch model construction and activation capture for DeepDream.

Presentation metadata stays in the torch-free model_data module. This backend
owns model loading, layer resolution, truncated forwards, and temporary hooks;
it does not perform media I/O or orchestrate generation jobs.
"""

from collections.abc import Callable, Iterable
from typing import TypedDict, cast

import torch
import torchvision.models as models
from torch import Tensor, nn

from .model_data import (
    MODEL_DEFAULTS,
    MODEL_DESCRIPTIONS,
    MODEL_LAYERS,
    MODEL_MIN_SIZE,
    MODEL_ORDER,
    MODEL_RECOMMENDED,
)

Device = str | torch.device
Activations = dict[str, Tensor]
ModelBuilder = Callable[[Device], nn.Module]
LayerForward = Callable[[nn.Module, Tensor, set[str]], Activations]


class ModelSpec(TypedDict):
    """Runtime implementation and CLI metadata for a supported model."""

    description: str
    layers: list[tuple[str, str]]
    recommended: str
    build: ModelBuilder
    forward_fn: LayerForward
    min_size: int
    defaults: dict[str, float]


def _freeze(model: nn.Module) -> nn.Module:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def build_googlenet(device: Device, weights: bool = True) -> nn.Module:
    """Build GoogLeNet; weights=False permits network-free model tests."""
    if weights:
        model = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1).to(device)
    else:
        model = models.googlenet(weights=None, aux_logits=True, init_weights=True).to(device)
    model.aux_logits = False
    return _freeze(model)


def _forward_trunk(
    model: nn.Module, x: Tensor, targets: set[str], is_aux: Callable[[str], bool]
) -> Activations:
    """Walk trunk children, restoring input transforms and skipping auxiliary heads."""
    if getattr(model, "transform_input", False) and hasattr(model, "_transform_input"):
        transform = cast(Callable[[Tensor], Tensor], model._transform_input)
        x = transform(x)
    children = list(model.named_children())
    names = [name for name, _ in children]
    missing = targets - set(names)
    if missing:
        raise RuntimeError(f"Target layer(s) not found in model: {sorted(missing)}")
    aux_targets = sorted(name for name in targets if is_aux(name))
    if aux_targets:
        raise RuntimeError(
            f"Target layer(s) are auxiliary heads, not dreamable trunk layers: {aux_targets}"
        )
    deepest = max(i for i, name in enumerate(names) if name in targets)
    stop_at = names[deepest + 1] if deepest + 1 < len(names) else None
    acts: Activations = {}
    for name, module in children:
        if name == stop_at:
            break
        if is_aux(name):
            continue
        x = module(x)
        if name in targets:
            acts[name] = x
            if len(acts) == len(targets):
                break
    missing = targets - set(acts)
    if missing:
        raise RuntimeError(f"Target layer(s) not found in model: {sorted(missing)}")
    return acts


def _forward_googlenet(model: nn.Module, x: Tensor, targets: set[str]) -> Activations:
    return _forward_trunk(model, x, targets, lambda name: name.startswith("aux"))


def _build_inceptionv3(device: Device) -> nn.Module:
    model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1).to(device)
    model.aux_logits = False
    return _freeze(model)


def _forward_inceptionv3(model: nn.Module, x: Tensor, targets: set[str]) -> Activations:
    return _forward_trunk(model, x, targets, lambda name: name == "AuxLogits")


def _build_vgg(variant: str, device: Device) -> nn.Module:
    if variant == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).to(device)
    else:
        model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).to(device)
    return _freeze(model)


def _forward_vgg(model: nn.Module, x: Tensor, targets: set[str]) -> Activations:
    """Run only the features block, stopping at the deepest requested index."""
    idx_to_name: dict[int, str] = {}
    for name in targets:
        if not name.startswith("features."):
            raise RuntimeError(
                f"VGG target '{name}' must be in the form 'features.N' (e.g. 'features.28')."
            )
        parts = name.split(".")
        if len(parts) != 2:
            raise RuntimeError(f"Cannot parse layer index from '{name}'.") from None
        try:
            idx = int(parts[1])
        except ValueError:
            raise RuntimeError(f"Cannot parse layer index from '{name}'.") from None
        idx_to_name[idx] = name
    features = cast(nn.Sequential, model.features)
    try:
        num_features = len(features)
    except TypeError:
        num_features = None
    if num_features is not None:
        bad = sorted(name for idx, name in idx_to_name.items() if idx < 0 or idx >= num_features)
        if bad:
            raise RuntimeError(f"Target layer(s) not found in model.features: {bad}")
    deepest_idx = max(idx_to_name)
    acts: Activations = {}
    for i, layer in enumerate(features):
        x = layer(x)
        if i in idx_to_name:
            acts[idx_to_name[i]] = x
        if i >= deepest_idx:
            break
    missing = targets - set(acts)
    if missing:
        raise RuntimeError(f"Target layer(s) not found in model.features: {sorted(missing)}")
    return acts


_MODEL_IMPLS: dict[str, tuple[ModelBuilder, LayerForward]] = {
    "googlenet": (build_googlenet, _forward_googlenet),
    "inceptionv3": (_build_inceptionv3, _forward_inceptionv3),
    "vgg16": (lambda device: _build_vgg("vgg16", device), _forward_vgg),
    "vgg19": (lambda device: _build_vgg("vgg19", device), _forward_vgg),
}

MODEL_REGISTRY: dict[str, ModelSpec] = {
    key: {
        "description": MODEL_DESCRIPTIONS[key],
        "layers": list(MODEL_LAYERS[key]),
        "recommended": MODEL_RECOMMENDED[key],
        "build": _MODEL_IMPLS[key][0],
        "forward_fn": _MODEL_IMPLS[key][1],
        "min_size": MODEL_MIN_SIZE[key],
        "defaults": dict(MODEL_DEFAULTS[key]),
    }
    for key in MODEL_ORDER
}

TF_LAYER_MAP = {
    "mixed3a": "inception3a",
    "mixed3b": "inception3b",
    "mixed4a": "inception4a",
    "mixed4b": "inception4b",
    "mixed4c": "inception4c",
    "mixed4d": "inception4d",
    "mixed4e": "inception4e",
    "mixed5a": "inception5a",
    "mixed5b": "inception5b",
}


def resolve_layer(name: str, model_key: str) -> str:
    """Resolve canonical names and historical TensorFlow aliases."""
    if model_key not in MODEL_REGISTRY:
        raise SystemExit(f"Unknown model '{model_key}'. Valid models: {sorted(MODEL_REGISTRY)}")
    name = name.strip()
    canonical = TF_LAYER_MAP.get(name.lower(), name)
    valid_keys = {key for key, _ in MODEL_REGISTRY[model_key]["layers"]}
    if canonical in valid_keys:
        return canonical
    for key in valid_keys:
        if key.lower() == canonical.lower():
            return key
    raise SystemExit(
        f"Unknown layer '{name}' for model '{model_key}'. Valid layers: {sorted(valid_keys)}"
    )


def build_model(model_key: str, device: Device) -> nn.Module:
    """Load a supported model with frozen parameters on the selected device."""
    if model_key not in MODEL_REGISTRY:
        raise SystemExit(f"Unknown model '{model_key}'. Valid models: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[model_key]["build"](device)


def run_to_layers(
    model: nn.Module, model_key: str, x: Tensor, targets: Iterable[str]
) -> Activations:
    if model_key not in MODEL_REGISTRY:
        raise SystemExit(f"Unknown model '{model_key}'. Valid models: {sorted(MODEL_REGISTRY)}")
    target_set = set(targets)
    if not target_set:
        raise RuntimeError("No target layers specified.")
    return MODEL_REGISTRY[model_key]["forward_fn"](model, x, target_set)


def _module_for_target(model: nn.Module, name: str) -> nn.Module:
    module = model
    for part in name.split("."):
        module = cast(nn.Sequential, module)[int(part)] if part.isdigit() else getattr(module, part)
    return module


def forward_with_activations(
    model: nn.Module, x: Tensor, targets: Iterable[str]
) -> tuple[Activations, Tensor]:
    """Run a full forward, always removing temporary hooks, including on failure."""
    target_names = list(dict.fromkeys(targets))
    acts: Activations = {}
    handles = []

    def capture(name: str) -> Callable[[nn.Module, tuple[Tensor, ...], Tensor], None]:
        def hook(_module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor) -> None:
            acts[name] = output

        return hook

    unresolved = []
    try:
        for name in target_names:
            try:
                module = _module_for_target(model, name)
            except (AttributeError, IndexError, KeyError, TypeError, ValueError):
                unresolved.append(name)
                continue
            handles.append(module.register_forward_hook(capture(name)))
        if unresolved:
            raise RuntimeError(f"Target layer(s) not found in model: {sorted(unresolved)}")
        logits = model(x)
    finally:
        for handle in handles:
            handle.remove()
    missing = set(target_names) - set(acts)
    if missing:
        raise RuntimeError(f"Target layer(s) not found in model: {sorted(missing)}")
    return acts, logits
