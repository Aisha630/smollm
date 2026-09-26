"""Compare batched greedy generation against Hugging Face ``generate``.

Prompts of different lengths are left-padded into one batch. Each custom configuration must
reproduce the reference token sequences exactly, including rows that stop at EOS, and must match
its own output when every prompt is generated on its own.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch
import transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from smollm_lab import SmolLMConfig, SmolLMForCausalLM
from smollm_lab.generation import greedy_generate

CONFIGURATIONS = {
    "eager_no_cache": ("eager", {"use_cache": False}),
    "eager_kv_cache": ("eager", {"use_cache": True}),
    "sdpa_kv_cache": ("sdpa", {"use_cache": True}),
    "sdpa_static_cache": ("sdpa", {"use_cache": True, "cache_implementation": "static"}),
}


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_prompts(tokenizer, count: int, shortest: int, longest: int) -> list[list[int]]:
    """Return WikiText-2 paragraphs truncated to lengths spread evenly across a range."""

    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    paragraphs = [
        row["text"].strip()
        for row in dataset
        if row["text"].strip() and not row["text"].strip().startswith("=")
    ]
    prompts = []
    for text in paragraphs:
        token_ids = tokenizer(text, add_special_tokens=False).input_ids
        target = shortest + (len(prompts) * (longest - shortest)) // max(count - 1, 1)
        if len(token_ids) >= target:
            prompts.append(token_ids[:target])
        if len(prompts) == count:
            return prompts
    raise ValueError("not enough long paragraphs for the requested prompt lengths")


def left_pad(prompts: list[list[int]], pad_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(prompt) for prompt in prompts)
    input_ids = torch.tensor([[pad_token_id] * (width - len(p)) + p for p in prompts])
    attention_mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in prompts])
    return input_ids, attention_mask


def strip_continuation(row: torch.Tensor, prompt_width: int, eos_token_id: int) -> list[int]:
    """Return generated tokens up to and including the first EOS."""

    tokens = row[prompt_width:].tolist()
    return tokens[: tokens.index(eos_token_id) + 1] if eos_token_id in tokens else tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM-135M")
    parser.add_argument("--num-prompts", type=int, default=16)
    parser.add_argument("--shortest-prompt", type=int, default=8)
    parser.add_argument("--longest-prompt", type=int, default=256)
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument(
        "--stop-text",
        help="Treat this text's token as EOS, e.g. '.' so rows stop at different steps",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    device = select_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    eos_token_id = tokenizer.eos_token_id
    if args.stop_text is not None:
        stop_ids = tokenizer(args.stop_text, add_special_tokens=False).input_ids
        if len(stop_ids) != 1:
            raise ValueError("--stop-text must encode to exactly one token")
        eos_token_id = stop_ids[0]
    prompts = load_prompts(tokenizer, args.num_prompts, args.shortest_prompt, args.longest_prompt)
    input_ids, attention_mask = left_pad(prompts, pad_token_id)
    width = input_ids.shape[1]
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    reference = AutoModelForCausalLM.from_pretrained(
        args.model_id, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    state_dict = reference.state_dict()
    reference.to(device)
    with torch.inference_mode():
        reference_output = reference.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.new_tokens,
            do_sample=False,
            num_beams=1,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        ).cpu()
    reference_tokens = [strip_continuation(row, width, eos_token_id) for row in reference_output]
    revision = getattr(reference.config, "_commit_hash", None)
    del reference

    results = {}
    for name, (backend, options) in CONFIGURATIONS.items():
        model = SmolLMForCausalLM(replace(SmolLMConfig(), attention_backend=backend))
        model.load_state_dict(state_dict, strict=True)
        model.to(device).eval()
        batched = greedy_generate(
            model,
            input_ids,
            args.new_tokens,
            eos_token_id=eos_token_id,
            attention_mask=attention_mask,
            pad_token_id=pad_token_id,
            **options,
        ).cpu()
        batched_tokens = [strip_continuation(row, width, eos_token_id) for row in batched]
        individual_tokens = []
        for prompt in prompts:
            single = greedy_generate(
                model,
                torch.tensor([prompt], device=device),
                args.new_tokens,
                eos_token_id=eos_token_id,
                **options,
            ).cpu()
            individual_tokens.append(strip_continuation(single[0], len(prompt), eos_token_id))
        compared = sum(len(tokens) for tokens in reference_tokens)
        matching = sum(
            sum(a == b for a, b in zip(ours, theirs, strict=False))
            for ours, theirs in zip(batched_tokens, reference_tokens, strict=True)
        )
        results[name] = {
            "attention_backend": backend,
            **options,
            "sequences_matching_reference": sum(
                ours == theirs
                for ours, theirs in zip(batched_tokens, reference_tokens, strict=True)
            ),
            "reference_tokens_matched": matching,
            "reference_tokens_compared": compared,
            "batched_matches_individual": sum(
                a == b for a, b in zip(batched_tokens, individual_tokens, strict=True)
            ),
            "full_output_identical": torch.equal(batched, reference_output),
        }
        print(name, json.dumps(results[name]), flush=True)
        del model

    result = {
        "model_id": args.model_id,
        "model_revision": revision,
        "device": str(device),
        "dtype": "float32",
        "num_prompts": len(prompts),
        "prompt_lengths": [len(prompt) for prompt in prompts],
        "new_tokens": args.new_tokens,
        "eos_token_id": eos_token_id,
        "stop_text": args.stop_text,
        "reference_rows_stopping_at_eos": sum(eos_token_id in row for row in reference_tokens),
        "reference_generated_lengths": [len(tokens) for tokens in reference_tokens],
        "configurations": results,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{rendered}\n", encoding="utf-8")
    all_match = all(
        entry["sequences_matching_reference"] == len(prompts)
        and entry["batched_matches_individual"] == len(prompts)
        for entry in results.values()
    )
    print("all configurations match:", all_match)


if __name__ == "__main__":
    main()
