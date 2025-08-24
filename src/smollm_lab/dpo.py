from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class DPOResult:
    loss: Tensor
    logits: Tensor
    chosen_rewards: Tensor
    rejected_rewards: Tensor


def sequence_log_probs(
    logits: Tensor,
    input_ids: Tensor,
    attention_mask: Tensor | None = None,
    response_mask: Tensor | None = None,
    average: bool = False,
) -> Tensor:
    """Compute sequence log-probabilities over non-padding response tokens."""

    if logits.shape[:2] != input_ids.shape:
        raise ValueError("logits and input_ids must share batch and sequence dimensions")
    shifted_logits = logits[:, :-1, :]
    shifted_labels = input_ids[:, 1:]
    token_log_probs = F.log_softmax(shifted_logits.float(), dim=-1)
    token_log_probs = token_log_probs.gather(-1, shifted_labels.unsqueeze(-1)).squeeze(-1)

    mask = torch.ones_like(shifted_labels, dtype=torch.bool)
    if attention_mask is not None:
        mask &= attention_mask[:, 1:].bool()
    if response_mask is not None:
        mask &= response_mask[:, 1:].bool()
    totals = (token_log_probs * mask).sum(dim=-1)
    if average:
        totals = totals / mask.sum(dim=-1).clamp_min(1)
    return totals


def dpo_loss(
    policy_chosen_logps: Tensor,
    policy_rejected_logps: Tensor,
    reference_chosen_logps: Tensor,
    reference_rejected_logps: Tensor,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> DPOResult:
    """Compute the reference-regularized Direct Preference Optimization objective."""

    if beta <= 0:
        raise ValueError("beta must be positive")
    if not 0 <= label_smoothing < 0.5:
        raise ValueError("label_smoothing must be in [0, 0.5)")
    policy_ratio = policy_chosen_logps - policy_rejected_logps
    reference_ratio = reference_chosen_logps - reference_rejected_logps
    logits = beta * (policy_ratio - reference_ratio)
    losses = -(
        (1 - label_smoothing) * F.logsigmoid(logits) + label_smoothing * F.logsigmoid(-logits)
    )
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()
    return DPOResult(
        loss=losses.mean(),
        logits=logits,
        chosen_rewards=chosen_rewards,
        rejected_rewards=rejected_rewards,
    )


def preference_accuracy(chosen_logps: Tensor, rejected_logps: Tensor) -> Tensor:
    if chosen_logps.shape != rejected_logps.shape:
        raise ValueError("chosen and rejected log-probabilities must have equal shapes")
    return (chosen_logps > rejected_logps).float().mean()
