"""SmolLM architecture, parameter-efficient fine-tuning, and alignment tools."""

from smollm_lab.config import SmolLMConfig
from smollm_lab.dpo import DPOResult, dpo_loss, preference_accuracy, sequence_log_probs
from smollm_lab.lora import LoRAConfig, LoRALinear, inject_lora, merge_lora
from smollm_lab.modeling import SmolLMForCausalLM, SmolLMModel, StaticKVCache

__all__ = [
    "DPOResult",
    "LoRAConfig",
    "LoRALinear",
    "SmolLMConfig",
    "SmolLMForCausalLM",
    "SmolLMModel",
    "StaticKVCache",
    "dpo_loss",
    "inject_lora",
    "merge_lora",
    "preference_accuracy",
    "sequence_log_probs",
]
