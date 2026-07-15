import torch

class StaticKVCache:
    
    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        dtype: torch.dtype,
        device: torch.device | str
    ):
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        
        cache_shape = (
            num_layers,
            batch_size,
            num_kv_heads,
            max_seq_len,
            head_dim
        )
        
        self.k_cache = torch.empty(
            cache_shape,
            dtype=dtype,
            device=device
        )
        self.v_cache = torch.empty(
            cache_shape,
            dtype=dtype,
            device=device
        )

    @torch.no_grad()
    def update(
        self,
        layer_idx: int,
        start_pos: int,
        k_states: torch.Tensor,
        v_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # k_states shape: [B, kv_heads, seq_len, head_dim]
        batch_size, num_kv_heads, seq_len, head_dim = k_states.shape
        
        if batch_size != self.batch_size:
            raise ValueError(
                f"KV cache batch size mismatch: "
                f"cache batch_size={self.batch_size}, "
                f"input batch_size={batch_size}"
            )
        
        num_new_tokens = k_states.shape[2]
        end_pos = start_pos + num_new_tokens
        self.k_cache[layer_idx, :, :, start_pos:end_pos, :].copy_(k_states)
        self.v_cache[layer_idx, :, :, start_pos:end_pos, :].copy_(v_states)
        
        return self.k_cache[layer_idx, :, :, :end_pos, :], self.v_cache[layer_idx, :, :, :end_pos, :]
        