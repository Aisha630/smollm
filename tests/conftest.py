import pytest

from smollm_lab.config import SmolLMConfig


@pytest.fixture
def tiny_config() -> SmolLMConfig:
    return SmolLMConfig(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
