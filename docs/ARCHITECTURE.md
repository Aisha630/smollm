# Architecture

## Decoder stack

Each decoder layer uses pre-normalization and two residual branches:

```text
hidden states
    ├─ RMSNorm → grouped-query self-attention → residual add
    └─ RMSNorm → SwiGLU feed-forward network → residual add
```

The default configuration matches SmolLM-135M: a 49,152-token vocabulary, hidden size 576, 30 decoder layers, nine query heads, three key/value heads, and an intermediate size of 1,536.

## Grouped-query attention

Nine query heads share three key/value heads. Repeating each KV head three times restores the head dimension required by scaled dot-product attention while reducing KV projections and cache size relative to standard multi-head attention.

Rotary positional embeddings are applied to queries and keys before KV repetition. A lower-triangular causal mask is combined with the caller's padding mask, so no token can attend to future or padded key positions.

The optimized backend delegates attention to PyTorch scaled-dot-product attention with native GQA. It passes the three KV heads directly to the fused kernel instead of materializing nine repeated KV heads. The eager backend remains available as a transparent numerical reference.

## Incremental decoding

Each decoder layer can return its rotated keys and values as a cache. During generation, the prompt is processed once; subsequent steps project only the newest token and attend over the cached context. Cache-aware position IDs preserve RoPE semantics, and explicit causal masks handle the asymmetric single-query/multi-key decode shape.

Cached and uncached generation are required to produce identical token sequences in the test suite. The cache stores three KV heads per layer rather than nine expanded heads, preserving the memory advantage of grouped-query attention.

## Parameter-efficient adaptation

For a frozen projection `W`, LoRA learns two matrices with rank `r`:

```math
h = Wx + \frac{\alpha}{r}BAx
```

`B` starts at zero, so adapter injection preserves the base model's initial function. For deployment, `BA` is added to `W` and the wrapper is removed. The tests compare outputs before and after merging.

## Preference optimization

DPO contrasts the policy's preference margin with the frozen reference model's margin:

```math
\mathcal{L}_{DPO} = -\log \sigma\left(\beta\left[\log\frac{\pi_\theta(y_w|x)}{\pi_\theta(y_l|x)} - \log\frac{\pi_{ref}(y_w|x)}{\pi_{ref}(y_l|x)}\right]\right)
```

Sequence likelihoods are calculated only over response tokens. This prevents the shared prompt from dominating the objective and excludes padding from both sums and averages.
