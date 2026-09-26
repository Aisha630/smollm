from dataclasses import replace

import pytest
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


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
@pytest.mark.parametrize(
    "cache_options",
    [
        {"use_cache": False},
        {"use_cache": True, "cache_implementation": "dynamic"},
        {"use_cache": True, "cache_implementation": "static"},
    ],
    ids=["no_cache", "dynamic_cache", "static_cache"],
)
def test_batched_left_padded_generation_matches_individual_prompts(
    tiny_config, backend, cache_options
) -> None:
    torch.manual_seed(0)
    model = SmolLMForCausalLM(replace(tiny_config, attention_backend=backend)).eval()
    prompts = [[5, 6, 7, 8, 9, 10], [11, 12], [13, 14, 15, 16]]
    width = max(len(prompt) for prompt in prompts)
    input_ids = torch.tensor([[0] * (width - len(prompt)) + prompt for prompt in prompts])
    attention_mask = torch.tensor(
        [[0] * (width - len(prompt)) + [1] * len(prompt) for prompt in prompts]
    )

    batched = greedy_generate(
        model, input_ids, max_new_tokens=8, attention_mask=attention_mask, **cache_options
    )

    for row, prompt in enumerate(prompts):
        individual = greedy_generate(
            model, torch.tensor([prompt]), max_new_tokens=8, **cache_options
        )
        assert torch.equal(batched[row, width - len(prompt) :], individual[0])


def test_static_cache_generation_matches_dynamic_cache_generation(tiny_config) -> None:
    model = SmolLMForCausalLM(replace(tiny_config, attention_backend="sdpa")).eval()
    input_ids = torch.tensor([[0, 0, 3, 4], [5, 6, 7, 8]])
    attention_mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])
    dynamic = greedy_generate(model, input_ids, 10, attention_mask=attention_mask)
    static = greedy_generate(
        model, input_ids, 10, attention_mask=attention_mask, cache_implementation="static"
    )
    assert torch.equal(static, dynamic)


def test_finished_rows_are_padded_until_every_row_stops(tiny_config) -> None:
    model = SmolLMForCausalLM(tiny_config).eval()
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    free_running = greedy_generate(model, input_ids, max_new_tokens=6)
    eos_token_id = free_running[0, 3].item()
    pad_token_id = tiny_config.vocab_size - 1

    output = greedy_generate(
        model, input_ids, 6, eos_token_id=eos_token_id, pad_token_id=pad_token_id
    )

    assert output[0, 3].item() == eos_token_id
    assert torch.all(output[0, 4:] == pad_token_id)
    other_row_stop = (free_running[1, 3:] == eos_token_id).nonzero()
    expected_length = 6 if len(other_row_stop) == 0 else other_row_stop[0].item() + 1
    assert output.shape[1] == 3 + expected_length
    assert torch.equal(output[1, 3:], free_running[1, 3 : 3 + expected_length])


def test_generation_rejects_unknown_cache_implementation(tiny_config) -> None:
    model = SmolLMForCausalLM(tiny_config).eval()
    with pytest.raises(ValueError, match="cache_implementation"):
        greedy_generate(model, torch.tensor([[1]]), 1, cache_implementation="paged")
