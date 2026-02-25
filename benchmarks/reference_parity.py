from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import datasets
import torch
import torch.nn.functional as F
import transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from smollm_lab import SmolLMConfig, SmolLMForCausalLM


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_custom_config(reference_config, attention_backend: str) -> SmolLMConfig:
    return SmolLMConfig(
        vocab_size=reference_config.vocab_size,
        hidden_size=reference_config.hidden_size,
        intermediate_size=reference_config.intermediate_size,
        num_hidden_layers=reference_config.num_hidden_layers,
        num_attention_heads=reference_config.num_attention_heads,
        num_key_value_heads=reference_config.num_key_value_heads,
        max_position_embeddings=reference_config.max_position_embeddings,
        rope_theta=reference_config.rope_theta,
        rms_norm_eps=reference_config.rms_norm_eps,
        tie_word_embeddings=reference_config.tie_word_embeddings,
        attention_backend=attention_backend,
    )


@torch.inference_mode()
def compare_models(
    reference,
    custom,
    token_ids: torch.Tensor,
    context_length: int,
) -> dict[str, float | int]:
    reference_nll = 0.0
    custom_nll = 0.0
    scored_tokens = 0
    max_logit_difference = 0.0
    matching_predictions = 0
    started = time.perf_counter()

    for start in range(0, token_ids.shape[1], context_length):
        chunk = token_ids[:, start : start + context_length]
        if chunk.shape[1] < 2:
            continue
        attention_mask = torch.ones_like(chunk)
        reference_logits = reference(input_ids=chunk, attention_mask=attention_mask).logits.float()
        custom_logits = custom(input_ids=chunk, attention_mask=attention_mask).logits
        labels = chunk[:, 1:]
        reference_shifted = reference_logits[:, :-1, :]
        custom_shifted = custom_logits[:, :-1, :]

        reference_nll += F.cross_entropy(
            reference_shifted.reshape(-1, reference_shifted.shape[-1]),
            labels.reshape(-1),
            reduction="sum",
        ).item()
        custom_nll += F.cross_entropy(
            custom_shifted.reshape(-1, custom_shifted.shape[-1]),
            labels.reshape(-1),
            reduction="sum",
        ).item()
        scored_tokens += labels.numel()
        max_logit_difference = max(
            max_logit_difference,
            (reference_shifted - custom_shifted).abs().max().item(),
        )
        matching_predictions += (
            (reference_shifted.argmax(-1) == custom_shifted.argmax(-1)).sum().item()
        )

    reference_mean_nll = reference_nll / scored_tokens
    custom_mean_nll = custom_nll / scored_tokens
    reference_perplexity = math.exp(reference_mean_nll)
    custom_perplexity = math.exp(custom_mean_nll)
    return {
        "scored_tokens": scored_tokens,
        "reference_mean_nll": reference_mean_nll,
        "custom_mean_nll": custom_mean_nll,
        "reference_perplexity": reference_perplexity,
        "custom_perplexity": custom_perplexity,
        "absolute_perplexity_difference": abs(custom_perplexity - reference_perplexity),
        "relative_perplexity_difference_percent": (
            100 * abs(custom_perplexity - reference_perplexity) / reference_perplexity
        ),
        "max_absolute_logit_difference": max_logit_difference,
        "top1_token_agreement_percent": 100 * matching_predictions / scored_tokens,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare SmolLM reference and custom perplexity")
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM-135M")
    parser.add_argument("--dataset-id", default="Salesforce/wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-tokens", type=int, default=8_192)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--attention-backend", choices=["eager", "sdpa"], default="eager")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    device = select_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    reference = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    custom = SmolLMForCausalLM(build_custom_config(reference.config, args.attention_backend)).eval()
    load_result = custom.load_state_dict(reference.state_dict(), strict=True)

    dataset = load_dataset(
        args.dataset_id,
        args.dataset_config,
        split=args.split,
    )
    text = "\n\n".join(row["text"] for row in dataset if row["text"].strip())
    token_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
    token_ids = token_ids[:, : args.num_tokens].to(device)
    reference.to(device)
    custom.to(device)

    metrics = compare_models(reference, custom, token_ids, args.context_length)
    result = {
        "model_id": args.model_id,
        "model_revision": getattr(reference.config, "_commit_hash", None),
        "dataset_id": args.dataset_id,
        "dataset_config": args.dataset_config,
        "dataset_split": args.split,
        "dataset_fingerprint": dataset._fingerprint,
        "evaluation_tokens": token_ids.shape[1],
        "context_length": args.context_length,
        "attention_backend": args.attention_backend,
        "dtype": "float32",
        "device": str(device),
        "missing_keys": load_result.missing_keys,
        "unexpected_keys": load_result.unexpected_keys,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "datasets_version": datasets.__version__,
        **metrics,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{rendered}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
