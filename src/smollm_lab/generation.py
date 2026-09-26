from __future__ import annotations

import torch
from torch import Tensor, nn

from smollm_lab.modeling import StaticKVCache


@torch.inference_mode()
def greedy_generate(
    model: nn.Module,
    input_ids: Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    attention_mask: Tensor | None = None,
    use_cache: bool = True,
    pad_token_id: int | None = None,
    cache_implementation: str = "dynamic",
) -> Tensor:
    """Generate tokens greedily without mutating the caller's tensors.

    Batches of variable-length prompts must be left-padded and described by ``attention_mask``.
    Once a row emits ``eos_token_id``, its remaining positions are filled with ``pad_token_id``
    (defaulting to the EOS token). ``cache_implementation="static"`` preallocates the key/value
    cache for the whole generation instead of growing it at every step.
    """

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if cache_implementation not in {"dynamic", "static"}:
        raise ValueError("cache_implementation must be 'dynamic' or 'static'")
    generated = input_ids.clone()
    mask = torch.ones_like(generated) if attention_mask is None else attention_mask.clone()
    if pad_token_id is None:
        pad_token_id = eos_token_id
    finished = torch.zeros(generated.shape[0], dtype=torch.bool, device=generated.device)
    past_key_values = None
    if use_cache and cache_implementation == "static" and max_new_tokens > 0:
        parameter = next(model.parameters())
        past_key_values = StaticKVCache(
            model.config,
            batch_size=generated.shape[0],
            max_length=generated.shape[1] + max_new_tokens,
            device=parameter.device,
            dtype=parameter.dtype,
        )
    for step in range(max_new_tokens):
        model_input_ids = generated if step == 0 or not use_cache else generated[:, -1:]
        output = model(
            input_ids=model_input_ids,
            attention_mask=mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            logits_to_keep=1,
        )
        past_key_values = output.past_key_values if use_cache else None
        next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        if eos_token_id is not None:
            next_token = next_token.masked_fill(finished[:, None], pad_token_id)
            finished |= next_token.squeeze(-1) == eos_token_id
        generated = torch.cat((generated, next_token), dim=-1)
        mask = torch.cat((mask, torch.ones_like(next_token)), dim=-1)
        if eos_token_id is not None and torch.all(finished):
            break
    return generated
