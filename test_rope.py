# test_rope.py

from pathlib import Path
import torch

from config import Qwen3Config
from model.rope import Qwen3RotaryEmbedding, apply_rotary_pos_emb


def main():
    model_dir = Path("/mnt/yanghui/models/Qwen/Qwen3-4B").expanduser()
    config = Qwen3Config.from_json(model_dir / "config.json")

    B = 2
    T = 8
    num_q_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim

    rope = Qwen3RotaryEmbedding(config)

    q = torch.randn(B, num_q_heads, T, head_dim)
    k = torch.randn(B, num_kv_heads, T, head_dim)

    position_ids = torch.arange(T)

    cos, sin = rope(position_ids, dtype=q.dtype)

    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

    print("q shape:     ", q.shape)
    print("k shape:     ", k.shape)
    print("cos shape:   ", cos.shape)
    print("sin shape:   ", sin.shape)
    print("q_rot shape: ", q_rot.shape)
    print("k_rot shape: ", k_rot.shape)

    print("q has nan:   ", torch.isnan(q_rot).any().item())
    print("k has nan:   ", torch.isnan(k_rot).any().item())

    # RoPE 是旋转，理论上应该保持向量范数基本不变
    q_norm_before = torch.norm(q, dim=-1)
    q_norm_after = torch.norm(q_rot, dim=-1)

    max_norm_diff = (q_norm_before - q_norm_after).abs().max().item()

    print("max q norm diff:", max_norm_diff)


if __name__ == "__main__":
    main()