# Benchmarks

The system was evaluated across architecture parity, parameter-efficient adaptation, and preference alignment. Metrics below use fixed data splits and single seeded runs.

## 1. Architecture parity

The PyTorch implementation and Hugging Face reference model loaded the same `HuggingFaceTB/SmolLM-135M` checkpoint at revision `1d461723eec654e65efdc40cf49301c89c0c92f4`. All weights loaded strictly with zero missing or unexpected keys.

Both models evaluated the first 8,192 tokens from the WikiText-2 raw test split in 32 non-overlapping 256-token windows. Cross-entropy was calculated externally from each model's logits over the same 8,160 next-token predictions, avoiding framework-specific loss behavior.

| Metric | Hugging Face reference | Custom implementation | Difference |
| --- | ---: | ---: | ---: |
| Mean negative log-likelihood | 3.5175047931 | 3.5175047931 | 0.0 |
| Perplexity | 33.7002344078 | 33.7002344078 | 0.0 |
| Top-1 token agreement | - | 100% | - |
| Maximum absolute logit difference | - | - | 0.0 |

The primary comparison used float32 inference on Apple MPS with PyTorch 2.13.0, Transformers 4.57.6, and Datasets 5.0.1. An independent CPU run also produced zero perplexity difference, zero maximum logit difference, and 100% top-1 agreement. The dataset fingerprint was `a46124b21ac53738`. Run `uv run python benchmarks/reference_parity.py` to reproduce the measurement; the structured MPS result is stored in [`benchmarks/results/reference_parity.json`](../benchmarks/results/reference_parity.json).

The optimized SDPA backend was evaluated over the same 8,160 predictions. It retained 100% top-1 agreement and produced perplexity 33.7002361723, a relative difference of 0.0000052% from the Hugging Face reference. Its structured result is stored in [`benchmarks/results/reference_parity_sdpa.json`](../benchmarks/results/reference_parity_sdpa.json).

### Generation parity

Greedy generation was compared against Hugging Face `generate` on 16 WikiText-2 prompts with lengths spread from 8 to 256 tokens, left-padded into one batch, generating up to 64 tokens each. All four decoding configurations reproduced the reference output tensor exactly, including padding, and each batched sequence also matched the same prompt generated on its own:

| Configuration | Sequences matching reference | Batched = individual |
| --- | ---: | ---: |
| Eager attention, no cache | 16 / 16 | 16 / 16 |
| Eager attention, KV cache | 16 / 16 | 16 / 16 |
| SDPA, KV cache | 16 / 16 | 16 / 16 |
| SDPA, static KV cache | 16 / 16 | 16 / 16 |

With the model's EOS token, all 1,024 generated tokens matched. To exercise per-row stopping, a second run treated `.` as EOS: five rows stopped after 1, 4, 6, 8, and 25 tokens while the others continued, and all 748 generated tokens and the trailing padding still matched. Run `uv run python benchmarks/generation_parity.py` (add `--stop-text .` for the early-stopping run); results are stored in [`generation_parity.json`](../benchmarks/results/generation_parity.json) and [`generation_parity_stop_period.json`](../benchmarks/results/generation_parity_stop_period.json).

## 2. LoRA adaptation

### Configuration

- Base model: SmolLM-135M
- Dataset: shuffled 3,000-example subset of `databricks/databricks-dolly-15k`
- Split: 80% training / 20% validation
- Target projections: query, value, gate, up, and down
- Rank: 4
- Alpha: 8
- Dropout: 0.3
- Optimizer: AdamW, learning rate `1e-4`, weight decay `0.01`
- Training: five epochs, batch size 8, StepLR schedule

### Results

| Epoch | Train loss | Train perplexity | Validation loss | Validation perplexity |
| ----: | ---------: | ---------------: | --------------: | --------------------: |
|     1 |     3.3577 |            28.72 |          3.1430 |                 23.17 |
|     2 |     2.9689 |            19.47 |          3.0980 |             **22.15** |
|     3 |     2.7078 |            15.00 |          3.0998 |                 22.19 |
|     4 |     2.5798 |            13.19 |          3.1282 |                 22.83 |
|     5 |     2.4418 |            11.49 |          3.1717 |                 23.85 |

The run trained 990,720 adapter parameters, or 0.73% of the 135,505,728-parameter adapted model. Validation performance peaked at epoch 2, motivating checkpoint selection by validation loss rather than final-epoch training loss.

## 3. Preference alignment

### Direct preference optimization

A frozen reference model and trainable policy were evaluated on 1,000 scored preference pairs with a 90/10 split. The custom DPO loop reached a final batch loss of 0.6450 and improved held-out preference accuracy from 76% to 87%.

### Grammatical error correction

SmolLM was supervised-fine-tuned on CoEdIT grammatical error correction data. Candidate corrections were converted into preference pairs and ranked by edit distance to the reference correction. On the 485-example validation split:

| Model stage |             BLEU |
| ----------- | ---------------: |
| SFT         |           0.4722 |
| SFT + DPO   |       **0.4808** |

## 4. Inference optimization

The optimized inference path combines fused scaled-dot-product attention, native grouped-query attention, per-layer KV caching, and `torch.compile`. The eager baseline recomputes the full sequence at every decode step and explicitly repeats KV heads.

Both paths loaded the same SmolLM-135M checkpoint and generated 64 tokens from an identical 256-token WikiText-2 prompt. Results are medians from five steady-state repetitions after two warmups on Apple MPS in float32. The one-time compilation cost is excluded.

| Metric | Eager, no cache | SDPA + KV cache + compile | Change |
| --- | ---: | ---: | ---: |
| Prefill throughput | 8,037.60 tokens/s | 11,805.60 tokens/s | **1.47×** |
| Prefill latency | 31.85 ms | 21.68 ms | **31.9% lower** |
| Decode throughput | 27.20 tokens/s | 231.35 tokens/s | **8.50×** |
| 64-token decode latency | 2.3527 s | 0.2766 s | **88.2% lower** |
| Post-benchmark MPS driver memory | 5,493.80 MiB | 1,666.70 MiB | **69.7% lower** |

The generated-token SHA-256 hashes match exactly across both paths. Separately, float16 reduced resident model allocation from 513.15 MiB to 256.59 MiB and post-benchmark driver allocation by 22.1%, while preserving the generated sequence. Float16 did not improve decode throughput on this MPS device, so it is presented as a memory option rather than included in the speed headline.

The benchmark runner is [`benchmarks/inference_performance.py`](../benchmarks/inference_performance.py), and machine-readable measurements are stored under [`benchmarks/results`](../benchmarks/results).

## 5. Batched generation

Profiling batched generation showed that peak memory grew far faster than the KV cache. Two causes accounted for most of it. First, the forward pass projected every prompt position onto the vocabulary, so a batch of 8 × 512-token prompts materialized a 768 MiB float32 logits tensor to pick one next token per row. Second, the `torch.cat` cache allocated many small per-layer tensors that the MPS caching allocator placed in larger freed blocks: a 202 MiB cache occupied 1,330 MiB. Generation now projects only the last position, and `StaticKVCache` allocates every layer's cache once for the full generation length and writes into it in place.

The sweep generated 64 tokens per prompt for batch sizes 1–8 and prompt lengths 128–512 on Apple M4 Pro (MPS, float32). The baseline is the previous commit (SDPA with the `torch.cat` cache and full-sequence logits); Hugging Face `generate` with SDPA attention is included as an external reference. Throughput counts generated tokens across the batch, including prefill time. Results are medians of three repetitions after one warmup, without `torch.compile`.

| Prompt | Batch | Baseline tok/s | Optimized tok/s | HF `generate` tok/s | Speedup | Baseline peak | Optimized peak | Peak reduction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 1 | 158.7 | 173.2 | 100.7 | 1.09× | 589 MiB | 523 MiB | 11.3% |
| 128 | 8 | 714.3 | 810.2 | 518.1 | 1.13× | 798 MiB | 591 MiB | 26.0% |
| 256 | 4 | 451.1 | 509.3 | 293.1 | 1.13× | 806 MiB | 580 MiB | 28.0% |
| 256 | 8 | 605.8 | 704.0 | 435.5 | 1.16× | 1,747 MiB | 646 MiB | 63.1% |
| 512 | 1 | 140.9 | 156.8 | 80.5 | 1.11× | 674 MiB | 544 MiB | 19.4% |
| 512 | 4 | 351.4 | 418.9 | 255.3 | 1.19× | 1,737 MiB | 635 MiB | 63.5% |
| 512 | 8 | 405.7 | 544.7 | 365.2 | **1.34×** | 2,953 MiB | 757 MiB | **74.4%** |

Peak memory includes the 513 MiB of float32 weights; excluding weights, peak activation and cache memory fell by 72.7–90.1% across all 12 cells. At batch 8 × 512 tokens, inter-token latency fell from 13.21 ms to 9.30 ms. Of the 1.34× speedup there, 1.10× came from last-position logits alone and the rest from the static cache. The optimized path was 1.49–1.95× faster than Hugging Face `generate` across the grid. Its peak memory ranged from 1.3% higher to 24.9% lower than `generate`, with the largest savings on the largest batches: 13.8–24.9% lower at 2,048 or more prompt tokens per batch.

Generated tokens were identical across the baseline, the optimized path, and Hugging Face `generate` in every cell. A ragged sweep, in which each batch's prompts span half to all of the prompt length and are left-padded, gave the same memory reductions, speedups of 1.00–1.26×, and identical tokens.

Run `uv run python benchmarks/inference_sweep.py --mode sdpa_static_cache` (add `--ragged` for mixed lengths; `--mode sdpa_kv_cache` or `hf_generate` for the comparisons). Per-cell measurements are stored under [`benchmarks/results/sweep`](../benchmarks/results/sweep) and summarized in [`inference_sweep_summary.json`](../benchmarks/results/inference_sweep_summary.json).

## Evaluation scope

- Results represent single seeded runs; multi-seed confidence intervals remain future work.
- BLEU does not fully capture grammatical correctness or semantic preservation.
- Heuristic preference generation can introduce model and ranking bias; human evaluation would strengthen the alignment analysis.
- MPS does not expose an allocator high-water mark, so peak memory on MPS is sampled inside every attention block and after the vocabulary projection; short-lived kernel workspaces between samples are not captured. On CUDA the sweep uses `torch.cuda.max_memory_allocated`.
- The batched sweep stops at batch 8 × 512 tokens to stay within the memory budget of the 24 GB development machine.
- Explicit attention math prioritizes auditability and test coverage over fused-kernel throughput.
