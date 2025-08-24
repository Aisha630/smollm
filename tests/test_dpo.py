import math

import torch

from smollm_lab.dpo import dpo_loss, preference_accuracy, sequence_log_probs


def test_sequence_log_probs_respects_response_mask() -> None:
    logits = torch.zeros(1, 4, 3)
    input_ids = torch.tensor([[0, 1, 2, 1]])
    response_mask = torch.tensor([[0, 0, 1, 1]])
    result = sequence_log_probs(logits, input_ids, response_mask=response_mask)
    assert torch.allclose(result, torch.tensor([-2 * math.log(3)]))


def test_sequence_log_probs_ignores_padding() -> None:
    logits = torch.zeros(1, 4, 2)
    input_ids = torch.tensor([[0, 1, 0, 0]])
    attention_mask = torch.tensor([[1, 1, 0, 0]])
    result = sequence_log_probs(logits, input_ids, attention_mask=attention_mask)
    assert torch.allclose(result, torch.tensor([-math.log(2)]))


def test_dpo_loss_rewards_larger_policy_margin() -> None:
    neutral = torch.zeros(2)
    weak = dpo_loss(torch.tensor([0.1, 0.2]), neutral, neutral, neutral)
    strong = dpo_loss(torch.tensor([2.0, 3.0]), neutral, neutral, neutral)
    assert strong.loss < weak.loss
    assert torch.all(strong.chosen_rewards > strong.rejected_rewards)


def test_dpo_loss_has_policy_gradients_only() -> None:
    policy_chosen = torch.tensor([0.5, 0.3], requires_grad=True)
    policy_rejected = torch.tensor([0.1, 0.2], requires_grad=True)
    reference_chosen = torch.tensor([0.2, 0.2])
    reference_rejected = torch.tensor([0.1, 0.1])
    result = dpo_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected)
    result.loss.backward()
    assert policy_chosen.grad is not None
    assert policy_rejected.grad is not None


def test_preference_accuracy() -> None:
    chosen = torch.tensor([2.0, 0.0, 3.0, 1.0])
    rejected = torch.tensor([1.0, 1.0, 2.0, 1.0])
    assert preference_accuracy(chosen, rejected).item() == 0.5
