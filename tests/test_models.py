"""Backend boundary tests that do not download pretrained weights."""

import pytest
import torch
from torch import nn

from app.models import forward_with_activations


def test_activation_hooks_are_removed_when_model_forward_fails() -> None:
    class FailingModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block = nn.Identity()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.block(x)
            raise RuntimeError("forward failed")

    model = FailingModel()
    with pytest.raises(RuntimeError, match="forward failed"):
        forward_with_activations(model, torch.ones(1, 2), ["block"])
    assert not model.block._forward_hooks


def test_activation_targets_can_be_a_single_pass_iterable() -> None:
    model = nn.Sequential(nn.Linear(2, 2), nn.ReLU())
    x = torch.ones(1, 2, requires_grad=True)
    acts, logits = forward_with_activations(model, x, iter(["0", "0", "1"]))
    assert set(acts) == {"0", "1"}
    assert acts["1"] is logits
    logits.sum().backward()
    assert x.grad is not None
    assert all(not module._forward_hooks for module in model)
