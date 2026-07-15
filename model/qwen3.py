import torch
import torch.nn as nn

from config import Qwen3Config
from model.layers import Qwen3RMSNorm, Qwen3MLP
from model.attention import Qwen3Attention
from cache.kv_cache import StaticKVCache

class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        
        self.self_attn = Qwen3Attention(
            config=config,
            layer_idx=layer_idx
        )
        
        self.mlp = Qwen3MLP(config)
        
        self.input_layernorm = Qwen3RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps
        )
        
        self.post_attention_layernorm = Qwen3RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        kv_cache: StaticKVCache | None = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            start_pos=start_pos
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
    
    
class Qwen3Model(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()

        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
        )

        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(
                    config=config,
                    layer_idx=layer_idx,
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.norm = Qwen3RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        kv_cache: StaticKVCache | None = None,
        start_pos: int = 0
    ) -> torch.Tensor:
        """
        input_ids:
          [B, T]

        position_ids:
          [T] 或 [B, T]

        return:
          hidden_states [B, T, hidden_size]
        """
        B, T = input_ids.shape

        if position_ids is None:
            if attention_mask is not None:
                # without kv cache and only support prefill
                position_ids = attention_mask.to(torch.long).cumsum(dim=-1) - 1
                position_ids = position_ids.masked_fill(attention_mask == 0, 0)
                position_ids = position_ids[:, -T:]
            
            else:
                position_ids = torch.arange(
                    start_pos,
                    start_pos + T,
                    device=input_ids.device,
                    dtype=torch.long,
                )

        hidden_states = self.embed_tokens(input_ids)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states=hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
                kv_cache=kv_cache,
                start_pos=start_pos
            )

        hidden_states = self.norm(hidden_states)

        return hidden_states

    
    
class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        self.tie_weights()
    
    
    def tie_weights(self):
        if self.config.tie_word_embeddings is True:
            self.lm_head.weight = self.model.embed_tokens.weight
            
            
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        logits_to_keep: int = 0,
        kv_cache: StaticKVCache | None = None,
        start_pos: int = 0
    ):
        hidden_states = self.model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            start_pos=start_pos
        )
        if logits_to_keep > 0:
            hidden_states = hidden_states[:, -logits_to_keep:, :]
        
        logits = self.lm_head(hidden_states)
        return logits
        