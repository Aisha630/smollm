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
