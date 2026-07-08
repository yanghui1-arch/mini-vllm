from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class Qwen3Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool
    torch_dtype: str
    
    @classmethod
    def from_json(cls, config_path: str | Path) -> "Qwen3Config":
        config_path = Path(config_path)
        with open(config_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        return cls(
            vocab_size=raw["vocab_size"],
            hidden_size=raw["hidden_size"],
            intermediate_size=raw["intermediate_size"],
            num_hidden_layers=raw["num_hidden_layers"],
            num_attention_heads=raw["num_attention_heads"],
            num_key_value_heads=raw["num_key_value_heads"],
            head_dim=raw["head_dim"],
            rms_norm_eps=raw["rms_norm_eps"],
            rope_theta=raw.get("rope_theta", 1000000.0),
            tie_word_embeddings=raw.get("tie_word_embeddings", False),
            torch_dtype=raw.get("torch_dtype", "bfloat16"),
        )
    