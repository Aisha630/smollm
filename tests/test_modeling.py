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
    assert attention(inputs).shape == inputs.shape


def test_causal_mask_blocks_future_information(tiny_config) -> None:
    torch.manual_seed(1)
    attention = GroupedQueryAttention(tiny_config).eval()
    prefix = torch.randn(1, 3, tiny_config.hidden_size)
    first_suffix = torch.randn(1, 2, tiny_config.hidden_size)
    second_suffix = torch.randn(1, 2, tiny_config.hidden_size)
    first = attention(torch.cat((prefix, first_suffix), dim=1))[:, :3]
    second = attention(torch.cat((prefix, second_suffix), dim=1))[:, :3]
    assert torch.allclose(first, second, atol=1e-6)


def test_model_output_loss_and_weight_tying(tiny_config) -> None:
    model = SmolLMForCausalLM(tiny_config)
    input_ids = torch.randint(0, tiny_config.vocab_size, (2, 8))
    output = model(input_ids, labels=input_ids)
    assert output.logits.shape == (2, 8, tiny_config.vocab_size)
    assert output.loss is not None and output.loss.isfinite()
    assert model.lm_head.weight is model.model.embed_tokens.weight
