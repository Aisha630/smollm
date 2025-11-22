# Benchmarks

The system was evaluated across architecture parity, parameter-efficient adaptation, and preference alignment. Metrics below use fixed data splits and single seeded runs.

## 1. Architecture parity

The PyTorch SmolLM-135M implementation loaded the `HuggingFaceTB/SmolLM-135M` state dictionary and reproduced its expected greedy continuation for the prompt `The future of AI is`. The parity run exercised RMSNorm, RoPE, grouped-query attention, SwiGLU blocks, decoder stacking, causal masking, and tied embeddings.

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

## Evaluation scope

- Results represent single seeded runs; multi-seed confidence intervals remain future work.
- BLEU does not fully capture grammatical correctness or semantic preservation.
- Heuristic preference generation can introduce model and ranking bias; human evaluation would strengthen the alignment analysis.
- Explicit attention math prioritizes auditability and test coverage over fused-kernel throughput.
