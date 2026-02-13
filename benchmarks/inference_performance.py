from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from dataclasses import replace
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from smollm_lab import SmolLMConfig, SmolLMForCausalLM
from smollm_lab.generation import greedy_generate

MODES = {
    "eager_no_cache": {"attention_backend": "eager", "use_cache": False},
    "sdpa_kv_cache": {"attention_backend": "sdpa", "use_cache": True},
}


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def memory_snapshot(device: torch.device) -> dict[str, float]:
    synchronize(device)
    if device.type == "cuda":
        return {
            "allocated_mib": torch.cuda.memory_allocated(device) / 2**20,
            "reserved_mib": torch.cuda.memory_reserved(device) / 2**20,
        }
    if device.type == "mps":
        return {
            "allocated_mib": torch.mps.current_allocated_memory() / 2**20,
            "driver_mib": torch.mps.driver_allocated_memory() / 2**20,
        }
    return {}


def time_call(function, device: torch.device, repetitions: int) -> tuple[float, list[float]]:
    durations = []
    for _ in range(repetitions):
        synchronize(device)
        started = time.perf_counter()
        function()
        synchronize(device)
        durations.append(time.perf_counter() - started)
    return statistics.median(durations), durations


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SmolLM inference optimizations")
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM-135M")
    parser.add_argument("--prompt-tokens", type=int, default=256)
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.prompt_tokens < 2 or args.new_tokens < 1:
        raise ValueError("prompt_tokens must be at least 2 and new_tokens must be positive")
    device = select_device(args.device)
    dtype = getattr(torch, args.dtype)
    mode = MODES[args.mode]
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(row["text"] for row in dataset if row["text"].strip())
    input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
    input_ids = input_ids[:, : args.prompt_tokens].to(device)
    attention_mask = torch.ones_like(input_ids)

    reference = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.float32,
        attn_implementation="eager",
    )
    model_revision = getattr(reference.config, "_commit_hash", None)
    config = replace(SmolLMConfig(), attention_backend=mode["attention_backend"])
    model = SmolLMForCausalLM(config)
    model.load_state_dict(reference.state_dict(), strict=True)
    del reference
    model.to(device=device, dtype=dtype).eval()
    if args.compile:
        model = torch.compile(model, mode="reduce-overhead")
    model_memory = memory_snapshot(device)

    with torch.inference_mode():
        for _ in range(args.warmup):
            model(input_ids=input_ids, attention_mask=attention_mask)
        prefill_median, prefill_samples = time_call(
            lambda: model(input_ids=input_ids, attention_mask=attention_mask),
            device,
            args.repetitions,
        )

        warmup_tokens = min(args.new_tokens, 8)
        for _ in range(args.warmup):
            greedy_generate(
                model,
                input_ids,
                max_new_tokens=warmup_tokens,
                attention_mask=attention_mask,
                use_cache=mode["use_cache"],
            )

        generated = None

        def run_generation() -> None:
            nonlocal generated
            generated = greedy_generate(
                model,
                input_ids,
                max_new_tokens=args.new_tokens,
                attention_mask=attention_mask,
                use_cache=mode["use_cache"],
            )

        decode_median, decode_samples = time_call(run_generation, device, args.repetitions)

    if generated is None:
        raise RuntimeError("generation benchmark did not produce output")
    generated_cpu = generated.detach().cpu()
    token_bytes = generated_cpu.numpy().tobytes()
    result = {
        "mode": args.mode,
        "attention_backend": mode["attention_backend"],
        "use_kv_cache": mode["use_cache"],
        "model_id": args.model_id,
        "model_revision": model_revision,
        "device": str(device),
        "dtype": args.dtype,
        "compiled": args.compile,
        "prompt_tokens": input_ids.shape[1],
        "new_tokens": args.new_tokens,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "prefill_median_seconds": prefill_median,
        "prefill_tokens_per_second": input_ids.numel() / prefill_median,
        "decode_median_seconds": decode_median,
        "decode_tokens_per_second": args.new_tokens / decode_median,
        "prefill_samples_seconds": prefill_samples,
        "decode_samples_seconds": decode_samples,
        "generated_token_sha256": hashlib.sha256(token_bytes).hexdigest(),
        "model_memory": model_memory,
        "post_benchmark_memory": memory_snapshot(device),
        "torch_version": torch.__version__,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{rendered}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
