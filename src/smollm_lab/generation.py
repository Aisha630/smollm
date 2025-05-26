from __future__ import annotations

import torch
from torch import Tensor, nn


@torch.inference_mode()
def greedy_generate(
    model: nn.Module,
    input_ids: Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    attention_mask: Tensor | None = None,
) -> Tensor:
    """Generate tokens greedily without mutating the caller's tensors."""

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    generated = input_ids.clone()
    mask = torch.ones_like(generated) if attention_mask is None else attention_mask.clone()
    for _ in range(max_new_tokens):
        output = model(input_ids=generated, attention_mask=mask)
        next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat((generated, next_token), dim=-1)
        mask = torch.cat((mask, torch.ones_like(next_token)), dim=-1)
        if eos_token_id is not None and torch.all(next_token == eos_token_id):
            break
    return generated
