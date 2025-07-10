import torch
from torch import nn

from smollm_lab.lora import LoRAConfig, LoRALinear, inject_lora, merge_lora, parameter_summary
from smollm_lab.modeling import SmolLMForCausalLM


def test_zero_initialized_adapter_preserves_base_output() -> None:
    torch.manual_seed(0)
    base = nn.Linear(8, 6, bias=False)
    inputs = torch.randn(2, 4, 8)
    expected = base(inputs).clone()
    adapted = LoRALinear(base, rank=2, alpha=4)
    assert torch.equal(adapted(inputs), expected)


def test_only_adapter_parameters_are_trainable(tiny_config) -> None:
    model = SmolLMForCausalLM(tiny_config)
    names = inject_lora(model, LoRAConfig(target_modules=("q_proj", "v_proj")))
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert len(names) == tiny_config.num_hidden_layers * 2
    assert trainable
    assert all("lora_a" in name or "lora_b" in name for name in trainable)
    assert parameter_summary(model)["trainable_percent"] < 10


def test_merge_preserves_adapter_output() -> None:
    torch.manual_seed(3)
    layer = LoRALinear(nn.Linear(8, 6, bias=False), rank=2, alpha=4)
    nn.init.normal_(layer.lora_b)
    inputs = torch.randn(2, 5, 8)
    expected = layer(inputs)
    merged = layer.merge()
    assert torch.allclose(merged(inputs), expected, atol=1e-6)


def test_merge_unloads_wrappers(tiny_config) -> None:
    model = SmolLMForCausalLM(tiny_config).eval()
    inject_lora(model, LoRAConfig(target_modules=("q_proj",), dropout=0))
    for module in model.modules():
        if isinstance(module, LoRALinear):
            nn.init.normal_(module.lora_b)
    inputs = torch.randint(0, tiny_config.vocab_size, (1, 6))
    expected = model(inputs).logits
    merged_names = merge_lora(model)
    actual = model(inputs).logits
    assert len(merged_names) == tiny_config.num_hidden_layers
    assert not any(isinstance(module, LoRALinear) for module in model.modules())
    assert torch.allclose(actual, expected, atol=1e-5)
