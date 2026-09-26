"""Sweep batched generation throughput, latency, and peak memory.

Each cell generates a fixed number of tokens for a batch of WikiText-2 prompts. With ``--ragged``
the prompts in a batch have different lengths and are left-padded to the cell's context length.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import statistics
import time
from dataclasses import replace
from pathlib import Path

import torch
import transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import smollm_lab
from smollm_lab import SmolLMConfig, SmolLMForCausalLM
from smollm_lab.generation import greedy_generate

MODES = {
    "eager_no_cache": {"attention_backend": "eager", "generate": {"use_cache": False}},
    "sdpa_kv_cache": {"attention_backend": "sdpa", "generate": {"use_cache": True}},
    "sdpa_static_cache": {
        "attention_backend": "sdpa",
        "generate": {"use_cache": True, "cache_implementation": "static"},
    },
    "hf_generate": {"attention_backend": None, "generate": {}},
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


def release_memory(device: torch.device) -> None:
    gc.collect()
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def allocated_bytes(device: torch.device) -> int:
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device)
    if device.type == "mps":
        return torch.mps.current_allocated_memory()
    return 0


class PeakMemoryTracker:
    """Track peak allocated device memory during one generation call.

    CUDA exposes an exact allocator high-water mark. MPS does not, so allocated memory is sampled
    inside every attention block (while score and cache tensors are live) and after every
    projection to the vocabulary. The MPS driver high-water mark is reported alongside it.
    """

    def __init__(self, model: torch.nn.Module, device: torch.device) -> None:
        self.device = device
        self.peak = 0
        self.handles = []
        if device.type == "mps":
            for name, module in model.named_modules():
                if name.endswith("o_proj"):
                    self.handles.append(module.register_forward_pre_hook(self._sample_pre))
                if name.endswith("lm_head"):
                    self.handles.append(module.register_forward_hook(self._sample_post))
            self.handles.append(model.register_forward_hook(self._sample_post))

    def _sample_pre(self, *_) -> None:
        self.peak = max(self.peak, allocated_bytes(self.device))

    def _sample_post(self, *_) -> None:
        self.peak = max(self.peak, allocated_bytes(self.device))

    def __enter__(self) -> PeakMemoryTracker:
        release_memory(self.device)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.baseline = allocated_bytes(self.device)
        self.peak = self.baseline
        return self

    def __exit__(self, *_) -> None:
        synchronize(self.device)
        for handle in self.handles:
            handle.remove()
        if self.device.type == "cuda":
            self.peak = torch.cuda.max_memory_allocated(self.device)
        self.peak = max(self.peak, allocated_bytes(self.device))
        self.driver_after = (
            torch.mps.driver_allocated_memory() if self.device.type == "mps" else None
        )


def time_call(function, device: torch.device, repetitions: int) -> list[float]:
    durations = []
    for _ in range(repetitions):
        synchronize(device)
        started = time.perf_counter()
        function()
        synchronize(device)
        durations.append(time.perf_counter() - started)
    return durations


def build_batch(
    token_ids: torch.Tensor,
    batch_size: int,
    context_length: int,
    ragged: bool,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return left-padded prompts drawn from disjoint windows of the corpus."""

    if ragged and batch_size > 1:
        shortest = context_length // 2
        lengths = [
            context_length - (index * (context_length - shortest)) // (batch_size - 1)
            for index in range(batch_size)
        ]
    else:
        lengths = [context_length] * batch_size
    input_ids = torch.full((batch_size, context_length), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, context_length), dtype=torch.long)
    for row, length in enumerate(lengths):
        start = row * context_length
        if start + length > token_ids.numel():
            raise ValueError("corpus is too small for the requested batch and context length")
        input_ids[row, context_length - length :] = token_ids[start : start + length]
        attention_mask[row, context_length - length :] = 1
    return input_ids, attention_mask


def make_generate(mode: str, model, new_tokens: int, pad_token_id: int):
    settings = MODES[mode]["generate"]
    if mode == "hf_generate":

        def run(input_ids, attention_mask, max_new_tokens=new_tokens):
            return model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=None,
                num_beams=1,
                pad_token_id=pad_token_id,
            )

        return run

    def run(input_ids, attention_mask, max_new_tokens=new_tokens):
        return greedy_generate(
            model,
            input_ids,
            max_new_tokens=max_new_tokens,
            attention_mask=attention_mask,
            **settings,
        )

    return run


def load_model(mode: str, model_id: str, device: torch.device, dtype: torch.dtype):
    reference = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float32, attn_implementation="sdpa"
    )
    revision = getattr(reference.config, "_commit_hash", None)
    if mode == "hf_generate":
        return reference.to(device=device, dtype=dtype).eval(), revision
    config = replace(SmolLMConfig(), attention_backend=MODES[mode]["attention_backend"])
    model = SmolLMForCausalLM(config)
    model.load_state_dict(reference.state_dict(), strict=True)
    del reference
    return model.to(device=device, dtype=dtype).eval(), revision


def measure_cell(
    generate,
    model,
    device: torch.device,
    token_ids: torch.Tensor,
    batch_size: int,
    context_length: int,
    args: argparse.Namespace,
    pad_token_id: int,
    weight_bytes: int,
) -> dict:
    input_ids, attention_mask = build_batch(
        token_ids, batch_size, context_length, args.ragged, pad_token_id
    )
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    with torch.inference_mode():
        for _ in range(args.warmup):
            generate(input_ids, attention_mask)
        with PeakMemoryTracker(model, device) as memory:
            generated = generate(input_ids, attention_mask)
        first_token = time_call(
            lambda ids=input_ids, mask=attention_mask: generate(ids, mask, 1),
            device,
            args.repetitions,
        )
        total = time_call(
            lambda ids=input_ids, mask=attention_mask: generate(ids, mask),
            device,
            args.repetitions,
        )
    total_median = statistics.median(total)
    first_median = statistics.median(first_token)
    new_tokens = generated.shape[1] - context_length
    cell = {
        "batch_size": batch_size,
        "context_length": context_length,
        "prompt_tokens": int(attention_mask.sum().item()),
        "new_tokens": new_tokens,
        "total_median_seconds": total_median,
        "first_token_median_seconds": first_median,
        "inter_token_latency_ms": 1e3 * (total_median - first_median) / (new_tokens - 1),
        "generated_tokens_per_second": batch_size * new_tokens / total_median,
        "decode_tokens_per_second": (batch_size * (new_tokens - 1) / (total_median - first_median)),
        "peak_allocated_mib": memory.peak / 2**20,
        "peak_activation_and_cache_mib": (memory.peak - weight_bytes) / 2**20,
        "driver_after_mib": (None if memory.driver_after is None else memory.driver_after / 2**20),
        "generated_token_sha256": hashlib.sha256(
            generated[:, context_length:].cpu().numpy().tobytes()
        ).hexdigest(),
        "total_samples_seconds": total,
        "first_token_samples_seconds": first_token,
    }
    return cell


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--label", help="Name recorded in the output, e.g. the code revision")
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM-135M")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--context-lengths", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--ragged", action="store_true")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.new_tokens < 2:
        raise ValueError("new_tokens must be at least 2 to separate first-token latency")
    device = select_device(args.device)
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(row["text"] for row in dataset if row["text"].strip())
    token_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]

    model, revision = load_model(args.mode, args.model_id, device, dtype)
    generate = make_generate(args.mode, model, args.new_tokens, pad_token_id)
    weight_bytes = allocated_bytes(device)

    cells = []
    for context_length in args.context_lengths:
        for batch_size in args.batch_sizes:
            try:
                cell = measure_cell(
                    generate,
                    model,
                    device,
                    token_ids,
                    batch_size,
                    context_length,
                    args,
                    pad_token_id,
                    weight_bytes,
                )
            except (torch.OutOfMemoryError, RuntimeError) as error:
                if "out of memory" not in str(error).lower():
                    raise
                cell = {"batch_size": batch_size, "context_length": context_length, "oom": True}
                print(f"ctx={context_length:5d} bs={batch_size:3d} out of memory", flush=True)
            else:
                print(
                    f"ctx={context_length:5d} bs={batch_size:3d} "
                    f"{cell['generated_tokens_per_second']:9.1f} tok/s "
                    f"ttft={1e3 * cell['first_token_median_seconds']:8.1f} ms "
                    f"itl={cell['inter_token_latency_ms']:7.2f} ms "
                    f"peak={cell['peak_allocated_mib']:8.1f} MiB",
                    flush=True,
                )
            cells.append(cell)
            release_memory(device)

    result = {
        "label": args.label,
        "mode": args.mode,
        "model_id": args.model_id,
        "model_revision": revision,
        "implementation_path": str(Path(smollm_lab.__file__).parent),
        "device": str(device),
        "platform": platform.platform(),
        "dtype": args.dtype,
        "ragged": args.ragged,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "weight_mib": weight_bytes / 2**20,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cells": cells,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{json.dumps(result, indent=2, sort_keys=True)}\n", "utf-8")


if __name__ == "__main__":
    main()
