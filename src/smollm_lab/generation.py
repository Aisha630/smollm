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
    use_cache: bool = True,
) -> Tensor:
    """Generate tokens greedily without mutating the caller's tensors."""

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    generated = input_ids.clone()
    mask = torch.ones_like(generated) if attention_mask is None else attention_mask.clone()
    past_key_values = None
    for _ in range(max_new_tokens):
        model_input_ids = generated if past_key_values is None else generated[:, -1:]
        output = model(
            input_ids=model_input_ids,
            attention_mask=mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        past_key_values = output.past_key_values if use_cache else None
        next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat((generated, next_token), dim=-1)
        mask = torch.cat((mask, torch.ones_like(next_token)), dim=-1)
        if eos_token_id is not None and torch.all(next_token == eos_token_id):
            break
    return generated
