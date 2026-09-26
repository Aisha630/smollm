from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from smollm_lab.config import SmolLMConfig


class StaticKVCache:
    """Key/value buffers preallocated for a full generation and written in place.

    Growing a cache with ``torch.cat`` copies every layer's keys and values at each decoding step,
    and each of those many small tensors can be placed in a much larger freed block by the caching
    allocator. This cache makes one allocation for every layer's ``max_length`` positions and
    exposes the filled prefix as a view.
    """

    def __init__(
        self,
        config: SmolLMConfig,
        batch_size: int,
        max_length: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        shape = (batch_size, config.num_key_value_heads, max_length, config.head_dim)
        self.buffer = torch.zeros((2, config.num_hidden_layers, *shape), device=device, dtype=dtype)
        self.keys = [self.buffer[0, index] for index in range(config.num_hidden_layers)]
        self.values = [self.buffer[1, index] for index in range(config.num_hidden_layers)]
        self.length = 0

    def __len__(self) -> int:
        return len(self.keys)

    @property
    def max_length(self) -> int:
        return self.keys[0].shape[-2]

    def update(self, layer_index: int, key: Tensor, value: Tensor) -> tuple[Tensor, Tensor]:
        """Write new positions for one layer and return that layer's filled keys and values."""

        end = self.length + key.shape[-2]
        if end > self.max_length:
            raise ValueError("static key/value cache is full")
        if key.shape[:2] != self.keys[layer_index].shape[:2]:
            raise ValueError("static key/value cache has incompatible batch or head dimensions")
        self.keys[layer_index][:, :, self.length : end] = key
        self.values[layer_index][:, :, self.length : end] = value
        return self.keys[layer_index][:, :, :end], self.values[layer_index][:, :, :end]


KeyValueCache = tuple[tuple[Tensor, Tensor], ...] | StaticKVCache


@dataclass
class CausalLMOutput:
    logits: Tensor
    loss: Tensor | None = None
    past_key_values: KeyValueCache | None = None


@dataclass
class BaseModelOutput:
    last_hidden_state: Tensor
    past_key_values: KeyValueCache | None = None


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden_states.float() * torch.rsqrt(variance + self.eps)
        return (self.weight * normalized).to(input_dtype)


def rotate_half(hidden_states: Tensor) -> Tensor:
    if hidden_states.shape[-1] % 2:
        raise ValueError("rotary embedding dimension must be even")
    first, second = hidden_states.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10_000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(
        self, hidden_states: Tensor, position_ids: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        batch_size, _, sequence_length, _ = hidden_states.shape
        if position_ids is None:
            position_ids = torch.arange(sequence_length, device=hidden_states.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        frequencies = torch.einsum(
            "bi,j->bij", position_ids.float(), self.inv_freq.to(hidden_states.device)
        )
        angles = torch.cat((frequencies, frequencies), dim=-1).unsqueeze(1)
        return angles.cos().to(hidden_states.dtype), angles.sin().to(hidden_states.dtype)


def apply_rotary_pos_emb(
    query: Tensor, key: Tensor, cos: Tensor, sin: Tensor
) -> tuple[Tensor, Tensor]:
    return (
        (query * cos) + (rotate_half(query) * sin),
        (key * cos) + (rotate_half(key) * sin),
    )


def repeat_kv(hidden_states: Tensor, repetitions: int) -> Tensor:
    if repetitions == 1:
        return hidden_states
    batch, key_value_heads, sequence_length, head_dim = hidden_states.shape
    expanded = hidden_states[:, :, None, :, :].expand(
        batch, key_value_heads, repetitions, sequence_length, head_dim
    )
    return expanded.reshape(batch, key_value_heads * repetitions, sequence_length, head_dim)


def build_attention_mask(
    attention_mask: Tensor | None,
    batch_size: int,
    query_length: int,
    key_value_length: int,
    past_length: int,
    device: torch.device,
) -> Tensor:
    query_positions = torch.arange(
        past_length, past_length + query_length, device=device
    ).unsqueeze(-1)
    key_positions = torch.arange(key_value_length, device=device).unsqueeze(0)
    causal = (key_positions <= query_positions).view(1, 1, query_length, key_value_length)
    if attention_mask is None:
        return causal.expand(batch_size, -1, -1, -1)
    if attention_mask.ndim == 2:
        if attention_mask.shape != (batch_size, key_value_length):
            raise ValueError("2D attention_mask must have shape (batch, key/value sequence)")
        key_mask = attention_mask.to(device=device, dtype=torch.bool)[:, None, None, :]
        allowed = causal & key_mask
    elif attention_mask.ndim == 4:
        if attention_mask.shape[-2:] != (query_length, key_value_length):
            raise ValueError("4D attention_mask has incompatible query/key dimensions")
        allowed = causal & attention_mask.to(device=device, dtype=torch.bool)
    else:
        raise ValueError("attention_mask must be 2D or 4D")
    # A left-padding query would otherwise attend to nothing, which some attention kernels turn
    # into NaNs. Letting it attend to itself keeps its (ignored) output finite.
    diagonal = (key_positions == query_positions).view(1, 1, query_length, key_value_length)
    return allowed | diagonal


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: SmolLMConfig, layer_index: int = 0) -> None:
        super().__init__()
        self.layer_index = layer_index
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_backend = config.attention_backend

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.rotary_emb = RotaryEmbedding(self.head_dim, config.rope_theta)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        past_key_value: tuple[Tensor, Tensor] | StaticKVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        batch_size, query_length, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(
            batch_size, query_length, self.num_heads, self.head_dim
        )
        key = self.k_proj(hidden_states).view(
            batch_size, query_length, self.num_key_value_heads, self.head_dim
        )
        value = self.v_proj(hidden_states).view(
            batch_size, query_length, self.num_key_value_heads, self.head_dim
        )
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        if past_key_value is None:
            past_length = 0
        elif isinstance(past_key_value, StaticKVCache):
            past_length = past_key_value.length
        else:
            past_length = past_key_value[0].shape[-2]
        if attention_mask is not None and attention_mask.ndim == 2:
            position_ids = attention_mask.long().cumsum(dim=-1) - 1
            position_ids = position_ids.masked_fill(attention_mask == 0, 0)
            position_ids = position_ids[:, -query_length:]
        else:
            position_ids = torch.arange(
                past_length,
                past_length + query_length,
                device=hidden_states.device,
            ).unsqueeze(0)
            position_ids = position_ids.expand(batch_size, -1)
        cos, sin = self.rotary_emb(query, position_ids)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        if isinstance(past_key_value, StaticKVCache):
            key, value = past_key_value.update(self.layer_index, key, value)
            present_key_value = None
        else:
            if past_key_value is not None:
                past_key, past_value = past_key_value
                if past_key.shape[:2] != (batch_size, self.num_key_value_heads):
                    raise ValueError(
                        "past key/value cache has incompatible batch or head dimensions"
                    )
                key = torch.cat((past_key, key), dim=-2)
                value = torch.cat((past_value, value), dim=-2)
            present_key_value = (key, value) if use_cache else None
        key_value_length = key.shape[-2]

        allowed = build_attention_mask(
            attention_mask,
            batch_size,
            query_length,
            key_value_length,
            past_length,
            hidden_states.device,
        )

        if self.attention_backend == "sdpa":
            context = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=allowed,
                dropout_p=0.0,
                is_causal=False,
                enable_gqa=True,
            )
        else:
            repeated_key = repeat_kv(key, self.num_key_value_groups)
            repeated_value = repeat_kv(value, self.num_key_value_groups)
            scores = torch.matmul(query, repeated_key.transpose(-1, -2)) * (self.head_dim**-0.5)
            scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
            probabilities = F.softmax(scores.float(), dim=-1).to(query.dtype)
            context = torch.matmul(probabilities, repeated_value)
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, query_length, self.hidden_size)
        )
        return self.o_proj(context), present_key_value


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class DecoderLayer(nn.Module):
    def __init__(self, config: SmolLMConfig, layer_index: int = 0) -> None:
        super().__init__()
        self.self_attn = GroupedQueryAttention(config, layer_index)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        past_key_value: tuple[Tensor, Tensor] | StaticKVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        attention_output, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            attention_mask,
            past_key_value,
            use_cache,
        )
        hidden_states = hidden_states + attention_output
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value


class SmolLMModel(nn.Module):
    def __init__(self, config: SmolLMConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            DecoderLayer(config, index) for index in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        past_key_values: KeyValueCache | None = None,
        use_cache: bool = False,
    ) -> BaseModelOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, sequence)")
        if past_key_values is not None and len(past_key_values) != len(self.layers):
            raise ValueError("past_key_values must contain one entry per decoder layer")
        hidden_states = self.embed_tokens(input_ids)
        static_cache = past_key_values if isinstance(past_key_values, StaticKVCache) else None
        next_cache = [] if use_cache and static_cache is None else None
        for index, layer in enumerate(self.layers):
            if static_cache is not None:
                past_key_value = static_cache
            else:
                past_key_value = None if past_key_values is None else past_key_values[index]
            hidden_states, present_key_value = layer(
                hidden_states,
                attention_mask,
                past_key_value,
                use_cache,
            )
            if next_cache is not None and present_key_value is not None:
                next_cache.append(present_key_value)
        if static_cache is not None:
            static_cache.length += input_ids.shape[1]
            next_cache = static_cache
        elif next_cache is not None:
            next_cache = tuple(next_cache)
        return BaseModelOutput(
            last_hidden_state=self.norm(hidden_states),
            past_key_values=next_cache,
        )


class SmolLMForCausalLM(nn.Module):
    def __init__(self, config: SmolLMConfig) -> None:
        super().__init__()
        self.config = config
        self.model = SmolLMModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.tie_weights()

    def tie_weights(self) -> None:
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        past_key_values: KeyValueCache | None = None,
        use_cache: bool = False,
        logits_to_keep: int = 0,
    ) -> CausalLMOutput:
        """Run the model; ``logits_to_keep=n`` projects only the last ``n`` positions."""

        if labels is not None and past_key_values is not None:
            raise ValueError("labels cannot be used with a past key/value cache")
        if logits_to_keep < 0:
            raise ValueError("logits_to_keep must be non-negative")
        if labels is not None and logits_to_keep:
            raise ValueError("labels require logits for every position")
        model_output = self.model(
            input_ids,
            attention_mask,
            past_key_values,
            use_cache,
        )
        hidden_states = model_output.last_hidden_state
        if logits_to_keep:
            hidden_states = hidden_states[:, -logits_to_keep:, :]
        logits = self.lm_head(hidden_states).float()
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return CausalLMOutput(
            logits=logits,
            loss=loss,
            past_key_values=model_output.past_key_values,
        )
