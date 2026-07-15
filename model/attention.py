import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Qwen3Config
from model.layers import Qwen3RMSNorm
from model.rope import Qwen3RotaryEmbedding, apply_rotary_pos_emb
from cache.kv_cache import StaticKVCache

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    
    if n_rep == 1:
        return hidden_states
    
    B, num_kv_heads, T, head_dim = hidden_states.shape
    
    hidden_states = hidden_states[:, :, None, :, :]
    hidden_states = hidden_states.expand(B, num_kv_heads, n_rep, T, head_dim)
    return hidden_states.reshape(B, num_kv_heads * n_rep, T, head_dim)


class Qwen3Attention(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        
        self.config = config
        self.layer_idx = layer_idx
        
        self.hidden_size = config.hidden_size
        self.num_q_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        
        self.num_kv_groups = self.num_q_heads // self.num_kv_heads
        
        if self.num_q_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_attention_heads must be divisible by num_kv_heads,",
                f"got {self.num_q_heads} and {self.num_kv_heads}"
            )
        
        self.q_size = self.num_q_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        
        self.q_proj = nn.Linear(self.hidden_size, self.q_size, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_size, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_size, bias=config.attention_bias)
        
        self.o_proj = nn.Linear(self.q_size, self.hidden_size, bias=config.attention_bias)
        
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.attention_dropout = config.attention_dropout
        
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        kv_cache: StaticKVCache | None = None,
        start_pos: int = 0,
    ):
        B, T, hidden_size = hidden_states.shape
        
        if position_ids is None:
            position_ids = torch.arange(start_pos, start_pos + T, device=hidden_states.device, dtype=torch.long)
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        query_states = query_states.view(B, T, self.num_q_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        
        cos, sin = self.rotary_emb(
            position_ids=position_ids,
            dtype=query_states.dtype
        )
        
        query_states, key_states = apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin
        )
        
        if kv_cache is not None:
            key_states, value_states = kv_cache.update(
                self.layer_idx,
                start_pos=start_pos,
                k_states=key_states,
                v_states=value_states,
            )
        
        key_states = repeat_kv(key_states, self.num_kv_groups)
        value_states = repeat_kv(value_states, self.num_kv_groups)
        
        attn_output = self._attention(
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            start_pos=start_pos
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, self.q_size)
        output = self.o_proj(attn_output)
        
        return output
    
    
    def _attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        start_pos: int = 0,
    ):
        B, H, q_len, D = q.shape
        k_len = k.shape[-2]
        
        attn_scores = q @ k.transpose(-2, -1)
        attn_scores = attn_scores / math.sqrt(D)
        
        q_positions = torch.arange(start_pos, start_pos + q_len, device=attn_scores.device)
        k_positions = torch.arange(k_len, device=attn_scores.device)
        
        # mask without kv-cache
        # mask = torch.triu(torch.ones(T, T, device=attn_scores.device, dtype=torch.bool), diagonal=1)
        # [k_len] -> [1, k_len]
        # [q_len] -> [q_len, 1]
        #            [q_len, k_len] -> [1, 1, q_len, k_len]
        causal_mask = (
            k_positions[None, :] <= q_positions[:, None]
        )[None, None, :, :]
        
        # (B, k_len) -> (B, 1, 1, k_len)
        padding_k_mask = attention_mask.to(
            device=attn_scores.device,
            dtype=torch.bool
        )[:, None, None, :]
        
        # (B, 1, q_len, k_len)
        mask = padding_k_mask & causal_mask
        mask_value = torch.finfo(attn_scores.dtype).min
        
        # Deprecate -torch.inf
        attn_scores = attn_scores.masked_fill(mask == 0, mask_value)
        
        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(dtype=v.dtype)
        attn_output = attn_weights @ v
        
        return attn_output