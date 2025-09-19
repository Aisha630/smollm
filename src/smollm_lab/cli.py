import argparse

import torch

from smollm_lab.config import SmolLMConfig
from smollm_lab.dpo import dpo_loss
from smollm_lab.lora import LoRAConfig, inject_lora, parameter_summary
from smollm_lab.modeling import SmolLMForCausalLM


def smoke_test(seed: int) -> None:
    torch.manual_seed(seed)
    config = SmolLMConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    model = SmolLMForCausalLM(config)
    inputs = torch.randint(0, config.vocab_size, (2, 16))
    output = model(inputs)
    modules = inject_lora(model, LoRAConfig(target_modules=("q_proj", "v_proj")))
    stats = parameter_summary(model)

    neutral = torch.zeros(2)
    preferred = torch.tensor([0.5, 1.0])
    result = dpo_loss(preferred, neutral, neutral, neutral)
    print(f"logits_shape={tuple(output.logits.shape)}")
    print(f"lora_modules={len(modules)}")
    print(f"trainable={stats['trainable']:,} ({stats['trainable_percent']:.2f}%)")
    print(f"synthetic_dpo_loss={result.loss.item():.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Utilities for the SmolLM Alignment Lab")
    parser.add_argument("command", choices=["smoke"], help="command to run")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.command == "smoke":
        smoke_test(args.seed)


if __name__ == "__main__":
    main()
