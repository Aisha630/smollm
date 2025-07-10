from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class LoRAConfig:
    rank: int = 4
    alpha: float = 8.0
    dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj",
        "v_proj",
        "up_proj",
        "down_proj",
        "gate_proj",
    )
    freeze_base: bool = True

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError("rank must be positive")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not self.target_modules:
            raise ValueError("target_modules cannot be empty")


class LoRALinear(nn.Module):
    """A frozen linear projection plus a trainable low-rank update."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0) -> None:
        super().__init__()
        self.base = base
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        self.merged = False
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)
        base.weight.requires_grad_(False)
        if base.bias is not None:
            base.bias.requires_grad_(False)

    @property
    def delta_weight(self) -> Tensor:
        return (self.lora_b @ self.lora_a) * self.scaling

    def forward(self, inputs: Tensor) -> Tensor:
        output = self.base(inputs)
        if self.merged:
            return output
        update = F.linear(F.linear(self.dropout(inputs), self.lora_a), self.lora_b)
        return output + update * self.scaling

    @torch.no_grad()
    def merge(self) -> nn.Linear:
        if not self.merged:
            self.base.weight.add_(self.delta_weight.to(self.base.weight.dtype))
            self.merged = True
        return self.base

    @torch.no_grad()
    def unmerge(self) -> None:
        if self.merged:
            self.base.weight.sub_(self.delta_weight.to(self.base.weight.dtype))
            self.merged = False


def _parent_and_name(model: nn.Module, qualified_name: str) -> tuple[nn.Module, str]:
    parts = qualified_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def inject_lora(model: nn.Module, config: LoRAConfig) -> list[str]:
    """Inject adapters into matching projections and return their qualified names."""

    if config.freeze_base:
        model.requires_grad_(False)
    matches = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in config.target_modules
    ]
    if not matches:
        raise ValueError(f"no linear layers matched target_modules={config.target_modules}")
    for name, module in matches:
        parent, child_name = _parent_and_name(model, name)
        setattr(parent, child_name, LoRALinear(module, config.rank, config.alpha, config.dropout))
    return [name for name, _ in matches]


@torch.no_grad()
def merge_lora(model: nn.Module, unload: bool = True) -> list[str]:
    """Merge every adapter into its base weight and optionally remove adapter wrappers."""

    matches = [
        (name, module) for name, module in model.named_modules() if isinstance(module, LoRALinear)
    ]
    for name, module in matches:
        base = module.merge()
        if unload:
            parent, child_name = _parent_and_name(model, name)
            setattr(parent, child_name, base)
    return [name for name, _ in matches]


def parameter_summary(model: nn.Module) -> dict[str, float | int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "total": total,
        "trainable": trainable,
        "trainable_percent": 100.0 * trainable / total,
    }
