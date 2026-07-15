import torch
import torch.nn as nn

from config import Qwen3Config

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half_dim = x.shape[-1] // 2
    x1 = x[..., : half_dim]
    x2 = x[..., half_dim: ]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed

class Qwen3RotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta
        
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {self.head_dim}")
        
        inv_freq = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float32)
                / self.head_dim
            )
        )
        
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
    @torch.no_grad()
    def forward(
        self,
        position_ids: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)
        
        # position_ids shape: [B, T]
        device = position_ids.device
        
        inv_freq = self.inv_freq.to(device=device)
        
        freqs = position_ids.float().unsqueeze(-1) * inv_freq.view(1, 1, -1)
        emb = torch.cat([freqs, freqs], dim=-1)
        
        cos = emb.cos().to(dtype=dtype)
        sin = emb.sin().to(dtype=dtype)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        return cos, sin
        