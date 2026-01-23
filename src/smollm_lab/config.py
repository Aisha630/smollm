from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SmolLMConfig:
    """Configuration for the 135M SmolLM architecture or a smaller test model."""

    vocab_size: int = 49_152
    hidden_size: int = 576
    intermediate_size: int = 1_536
    num_hidden_layers: int = 30
    num_attention_heads: int = 9
    num_key_value_heads: int = 3
    max_position_embeddings: int = 2_048
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True
    attention_backend: str = "eager"

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if (self.hidden_size // self.num_attention_heads) % 2:
            raise ValueError("attention head dimension must be even for rotary embeddings")
        positive_fields = (
            self.vocab_size,
            self.hidden_size,
            self.intermediate_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.max_position_embeddings,
        )
        if any(value <= 0 for value in positive_fields):
            raise ValueError("model dimensions must be positive")
        if self.attention_backend not in {"eager", "sdpa"}:
            raise ValueError("attention_backend must be 'eager' or 'sdpa'")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads
