from dataclasses import replace

import pytest
import torch

from smollm_lab.modeling import (
    GroupedQueryAttention,
    RMSNorm,
    SmolLMForCausalLM,
    StaticKVCache,
    repeat_kv,
    rotate_half,
)


def test_rotate_half() -> None:
    values = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    assert torch.equal(rotate_half(values), torch.tensor([[-3.0, -4.0, 1.0, 2.0]]))


def test_rmsnorm_matches_reference() -> None:
    torch.manual_seed(0)
    inputs = torch.randn(2, 3, 8)
    layer = RMSNorm(8, eps=1e-5)
    expected = inputs * torch.rsqrt(inputs.pow(2).mean(-1, keepdim=True) + 1e-5)
    assert torch.allclose(layer(inputs), expected, atol=1e-6)


def test_repeat_kv_preserves_group_order() -> None:
    keys = torch.tensor([[[[1.0]], [[2.0]]]])
    repeated = repeat_kv(keys, 2)
    assert repeated[:, :, 0, 0].tolist() == [[1.0, 1.0, 2.0, 2.0]]


def test_grouped_query_attention_shape(tiny_config) -> None:
    attention = GroupedQueryAttention(tiny_config)
    inputs = torch.randn(3, 7, tiny_config.hidden_size)
    output, cache = attention(inputs)
    assert output.shape == inputs.shape
    assert cache is None


def test_causal_mask_blocks_future_information(tiny_config) -> None:
    torch.manual_seed(1)
    attention = GroupedQueryAttention(tiny_config).eval()
    prefix = torch.randn(1, 3, tiny_config.hidden_size)
    first_suffix = torch.randn(1, 2, tiny_config.hidden_size)
    second_suffix = torch.randn(1, 2, tiny_config.hidden_size)
    first = attention(torch.cat((prefix, first_suffix), dim=1))[0][:, :3]
    second = attention(torch.cat((prefix, second_suffix), dim=1))[0][:, :3]
    assert torch.allclose(first, second, atol=1e-6)


def test_sdpa_matches_eager_attention(tiny_config) -> None:
    torch.manual_seed(2)
    eager = GroupedQueryAttention(tiny_config).eval()
    sdpa = GroupedQueryAttention(replace(tiny_config, attention_backend="sdpa")).eval()
    sdpa.load_state_dict(eager.state_dict())
    inputs = torch.randn(2, 9, tiny_config.hidden_size)
    mask = torch.ones(2, 9, dtype=torch.long)
    eager_output = eager(inputs, mask)[0]
    sdpa_output = sdpa(inputs, mask)[0]
    assert torch.allclose(sdpa_output, eager_output, atol=1e-5)


def test_cached_forward_matches_full_forward(tiny_config) -> None:
    config = replace(tiny_config, attention_backend="sdpa")
    model = SmolLMForCausalLM(config).eval()
    prompt = torch.randint(0, config.vocab_size, (1, 7))
    next_token = torch.randint(0, config.vocab_size, (1, 1))
    full_ids = torch.cat((prompt, next_token), dim=-1)

    full_logits = model(full_ids).logits[:, -1]
    prompt_output = model(prompt, use_cache=True)
    cached_logits = model(
        next_token,
        attention_mask=torch.ones_like(full_ids),
        past_key_values=prompt_output.past_key_values,
        use_cache=True,
    ).logits[:, -1]

    assert prompt_output.past_key_values is not None
    assert len(prompt_output.past_key_values) == config.num_hidden_layers
    assert torch.allclose(cached_logits, full_logits, atol=1e-5)


def test_cached_forward_matches_full_forward_with_left_padding(tiny_config) -> None:
    config = replace(tiny_config, attention_backend="sdpa")
    model = SmolLMForCausalLM(config).eval()
    prompt = torch.tensor([[0, 0, 7, 8, 9]])
    prompt_mask = torch.tensor([[0, 0, 1, 1, 1]])
    next_token = torch.tensor([[10]])
    full_ids = torch.cat((prompt, next_token), dim=-1)
    full_mask = torch.cat((prompt_mask, torch.ones_like(next_token)), dim=-1)

    full_logits = model(full_ids, attention_mask=full_mask).logits[:, -1]
    prompt_output = model(prompt, attention_mask=prompt_mask, use_cache=True)
    cached_logits = model(
        next_token,
        attention_mask=full_mask,
        past_key_values=prompt_output.past_key_values,
        use_cache=True,
    ).logits[:, -1]

    assert torch.allclose(cached_logits, full_logits, atol=1e-5)


def test_model_output_loss_and_weight_tying(tiny_config) -> None:
    model = SmolLMForCausalLM(tiny_config)
    input_ids = torch.randint(0, tiny_config.vocab_size, (2, 8))
    output = model(input_ids, labels=input_ids)
    assert output.logits.shape == (2, 8, tiny_config.vocab_size)
    assert output.loss is not None and output.loss.isfinite()
    assert model.lm_head.weight is model.model.embed_tokens.weight


def test_logits_to_keep_matches_final_positions(tiny_config) -> None:
    model = SmolLMForCausalLM(tiny_config).eval()
    input_ids = torch.randint(0, tiny_config.vocab_size, (2, 6))
    full_logits = model(input_ids).logits
    kept_logits = model(input_ids, logits_to_keep=2).logits
    assert kept_logits.shape == (2, 2, tiny_config.vocab_size)
    assert torch.allclose(kept_logits, full_logits[:, -2:], atol=1e-6)


def test_fully_padded_query_rows_stay_finite(tiny_config) -> None:
    input_ids = torch.tensor([[0, 0, 0, 5, 6], [1, 2, 3, 4, 5]])
    attention_mask = torch.tensor([[0, 0, 0, 1, 1], [1, 1, 1, 1, 1]])
    for backend in ("eager", "sdpa"):
        model = SmolLMForCausalLM(replace(tiny_config, attention_backend=backend)).eval()
        logits = model(input_ids, attention_mask=attention_mask).logits
        assert torch.isfinite(logits).all()


def test_static_cache_matches_dynamic_cache(tiny_config) -> None:
    config = replace(tiny_config, attention_backend="sdpa")
    model = SmolLMForCausalLM(config).eval()
    prompt = torch.tensor([[0, 0, 7, 8, 9], [3, 4, 5, 6, 7]])
    prompt_mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
    next_token = torch.tensor([[10], [11]])
    full_mask = torch.cat((prompt_mask, torch.ones_like(next_token)), dim=-1)

    cache = StaticKVCache(config, batch_size=2, max_length=8)
    static_prompt = model(prompt, attention_mask=prompt_mask, past_key_values=cache)
    static_next = model(next_token, attention_mask=full_mask, past_key_values=cache)
    dynamic_prompt = model(prompt, attention_mask=prompt_mask, use_cache=True)
    dynamic_next = model(
        next_token,
        attention_mask=full_mask,
        past_key_values=dynamic_prompt.past_key_values,
        use_cache=True,
    )

    assert static_next.past_key_values is cache
    assert cache.length == 6
    assert torch.allclose(static_prompt.logits, dynamic_prompt.logits, atol=1e-6)
    assert torch.allclose(static_next.logits, dynamic_next.logits, atol=1e-5)


def test_static_cache_rejects_overflow(tiny_config) -> None:
    model = SmolLMForCausalLM(tiny_config).eval()
    cache = StaticKVCache(tiny_config, batch_size=1, max_length=4)
    with pytest.raises(ValueError, match="full"):
        model(torch.tensor([[1, 2, 3, 4, 5]]), past_key_values=cache)
