from dataclasses import replace

import torch

from smollm_lab.modeling import (
    GroupedQueryAttention,
    RMSNorm,
    SmolLMForCausalLM,
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
