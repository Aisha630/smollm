from dataclasses import replace

import torch

from smollm_lab.generation import greedy_generate
from smollm_lab.modeling import SmolLMForCausalLM


def test_greedy_generate_appends_tokens_without_mutating_input(tiny_config) -> None:
    model = SmolLMForCausalLM(tiny_config).eval()
    input_ids = torch.tensor([[1, 2, 3]])
    original = input_ids.clone()
    output = greedy_generate(model, input_ids, max_new_tokens=4)
    assert output.shape == (1, 7)
    assert torch.equal(input_ids, original)


def test_cached_generation_matches_uncached_generation(tiny_config) -> None:
    model = SmolLMForCausalLM(replace(tiny_config, attention_backend="sdpa")).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    cached = greedy_generate(model, input_ids, max_new_tokens=6, use_cache=True)
    uncached = greedy_generate(model, input_ids, max_new_tokens=6, use_cache=False)
    assert torch.equal(cached, uncached)
