# SmolLM Alignment Lab

A from-scratch implementation of a 135M-parameter decoder-only language model, extended with parameter-efficient fine-tuning and direct preference optimization.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

This project demonstrates end-to-end LLM engineering across model architecture, efficient adaptation, preference alignment, and evaluation:

```mermaid
flowchart LR
    A[SmolLM architecture] --> B[Pretrained weights]
    B --> C[LoRA adaptation]
    C --> D[Supervised fine-tuning]
    D --> E[Preference pairs]
    E --> F[DPO alignment]
    F --> G[Evaluation]
```

The core architecture is implemented directly in PyTorch: RMSNorm, rotary positional embeddings, grouped-query attention, SwiGLU feed-forward blocks, causal masking, decoder layers, and tied token embeddings. The same codebase then adds LoRA adapters and a reference-regularized DPO objective.

## Experiment highlights

| Stage            |                               Result | Evaluation setup                                       |
| ---------------- | -----------------------------------: | ------------------------------------------------------ |
| Reference parity |          Perplexity: 33.7002 = 33.7002 | 8,160 WikiText-2 predictions; 100% top-1 agreement     |
| Inference optimization |             8.50× decode throughput | SDPA + native GQA + KV cache + compilation             |
| LoRA fine-tuning | 990,720 trainable parameters (0.73%) | Rank 4 adapters across attention and MLP projections   |
| LoRA validation  |               Best perplexity: 22.15 | 3,000-example Dolly subset; best checkpoint at epoch 2 |
| DPO alignment    |      Preference accuracy: 76% → 87% | 100-example held-out preference split                  |
| GEC alignment    |               BLEU: 0.4722 → 0.4808 | 485-example CoEdIT validation split, SFT → DPO        |

Metrics use fixed evaluation splits. Training runs used GPU acceleration, and reference parity was independently verified on both CPU and Apple MPS. See [Benchmarks](docs/BENCHMARKS.md) for configurations, per-epoch metrics, and evaluation scope.

Reproduce the reference parity comparison with:

```bash
uv sync --extra train
uv run python benchmarks/reference_parity.py
```

Reproduce the inference benchmark with:

```bash
uv run python benchmarks/inference_performance.py --mode eager_no_cache
uv run python benchmarks/inference_performance.py --mode sdpa_kv_cache --compile
```

## Engineering highlights

- Device-safe rotary embeddings and combined causal/padding masks
- Fused SDPA with native grouped-query attention, avoiding materialized KV-head copies
- Per-layer KV caching for incremental autoregressive decoding
- Optional `torch.compile` and float16 inference paths with output-parity checks
- Weight tying between token embeddings and the language-model head
- LoRA injection by module name, frozen-base training, and numerically verified merge/unload
- DPO over response tokens only, excluding prompt and padding tokens from sequence likelihoods
- Unit tests for tensor shapes, causal invariance, gradient flow, adapter equivalence, and preference loss
- Typed configuration, command-line smoke test, packaging metadata, and CI across Python 3.10 and 3.12

## Quick start

```bash
git clone https://github.com/Aisha630/smollm-alignment-lab.git
cd smollm-alignment-lab
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
smollm-lab smoke
```

The smoke test builds a compact two-layer model, runs a forward pass, injects LoRA adapters, reports trainable parameters, and evaluates a synthetic DPO batch. It runs on CPU and does not download model weights.

## Library usage

```python
import torch

from smollm_lab import LoRAConfig, SmolLMConfig, SmolLMForCausalLM, inject_lora

config = SmolLMConfig()
model = SmolLMForCausalLM(config)

adapted_layers = inject_lora(
    model,
    LoRAConfig(
        rank=4,
        alpha=8,
        target_modules=("q_proj", "v_proj", "up_proj", "down_proj", "gate_proj"),
    ),
)

input_ids = torch.randint(0, config.vocab_size, (1, 32))
logits = model(input_ids).logits
print(len(adapted_layers), logits.shape)
```

## Repository layout

```text
src/smollm_lab/
├── modeling.py      # Transformer architecture
├── lora.py          # Adapter injection and merging
├── dpo.py           # Preference objective and metrics
├── generation.py    # Greedy autoregressive decoding
└── cli.py           # CPU-friendly smoke test
tests/               # Behavioral and numerical tests
benchmarks/          # Reproducible reference-parity runner and results
docs/                # Architecture and benchmark details
```

## Production-minded design

The implementation favors explicit, inspectable model mechanics while enforcing numerical and behavioral correctness through tests. Core components are framework-native PyTorch modules, adapters can be merged for zero-overhead inference, preference likelihoods are masked at token level, and the package installs through standard Python tooling. The architecture is ready to extend with fused attention, mixed precision, activation checkpointing, and distributed training for larger runs.

## References

- Allal et al., [SmolLM: blazingly fast and remarkably powerful](https://huggingface.co/blog/smollm)
- Ainslie et al., [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)
- Hu et al., [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- Rafailov et al., [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- Su et al., [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
