import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Qwen3Config

class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        
        # For stable with fp32
        hidden_states = hidden_states.float()
        
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        
        return self.weight * hidden_states.to(input_dtype)
    
class Qwen3MLP(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.gate_proj = nn.Linear(
            self.hidden_size,
            self.intermediate_size,
            bias=False
        )
        
        self.up_proj = nn.Linear(
            self.hidden_size,
            self.intermediate_size,
            bias=False
        )
        
        self.down_proj = nn.Linear(
            self.intermediate_size,
            self.hidden_size,
            bias=False
        )
        
        if config.hidden_act != "silu":
            raise ValueError(f"Only silu is supported for now got {config.hidden_act}")
            
        self.act_fn = F.silu
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        
        x = self.act_fn(gate) * up
        x = self.down_proj(x)
        return x
        