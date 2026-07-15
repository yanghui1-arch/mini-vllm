import torch

from config import Qwen3Config
from cache.kv_cache import StaticKVCache

@torch.inference_mode()
def cached_greedy_generate(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None
):
    """Only support batch size = 1"""
    
    if input_ids.ndim != 2:
        raise ValueError("The shape of input_ids for greedy generation should ba 2, got ", input_ids.ndim)
    
    prompt_len = input_ids.shape[1]
    config: Qwen3Config = model.config
    kv_cache = StaticKVCache(
        num_layers=config.num_hidden_layers,
        batch_size=input_ids.shape[0],
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        max_seq_len=prompt_len + max_new_tokens,
        dtype=next(model.parameters()).dtype,
        device=input_ids.device
    )
    
    generated_ids = input_ids.clone()
    
    # 1. Prefill
    position_ids = torch.arange(0, prompt_len, device=input_ids.device, dtype=torch.long).unsqueeze(0)
    
    # (B, 1, vocab_size)
    logits = model(
        input_ids=input_ids,
        position_ids=position_ids,
        kv_cache=kv_cache,
        start_pos=0,
        logits_to_keep=1
    )
    
    # (B, 1)
    next_token = torch.argmax(
        logits[:, -1, :],
        dim=-1,
        keepdim=True
    )
    
    generated_ids = torch.cat([generated_ids, next_token], dim=1)
    
    if eos_token_id is not None:
        if next_token.item() == eos_token_id:
            return generated_ids
    
    # 2. Decode
    # Prefill has generated one token so decode max_new_tokens - 1
    for decode_step in range(max_new_tokens - 1):
        start_pos = prompt_len + decode_step
        position_ids = torch.tensor(
            [[start_pos]],
            device=input_ids.device,
            dtype=torch.long
        )
        
        logits = model(
            input_ids=next_token,
            position_ids=position_ids,
            kv_cache=kv_cache,
            start_pos=start_pos,
            logits_to_keep=1
        )
        
        # (B, 1)
        next_token = torch.argmax(
            logits[:, -1, :],
            dim=-1,
            keepdim=True
        )
        generated_ids = torch.cat(
            [generated_ids, next_token],
            dim=1
        )
        
        if eos_token_id is not None:
            if next_token.item() == eos_token_id:
                break

    return generated_ids
    

@torch.inference_mode()
def greedy_generate_without_kv_cache(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None
):
    
    generated_ids = input_ids.clone()
    
    for _ in range(max_new_tokens):
        seq_len = generated_ids.shape[1]
        position_ids = torch.arange(
            seq_len,
            device=generated_ids.device,
            dtype=torch.long
        ).unsqueeze(0)
        
        logits = model(
            generated_ids,
            position_ids=position_ids,
            logits_to_keep=1
        )
        
        next_token = torch.argmax(
            logits[:, -1, :],
            dim=-1,
            keepdim=True
        )
        generated_ids = torch.cat(
            [generated_ids, next_token],
            dim=1
        )
        
        if eos_token_id is not None:
            if next_token.item() == eos_token_id:
                break
                
    return generated_ids


@torch.inference_mode()
def batched_cached_greedy_generate(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
) -> torch.Tensor:
    # Only support same length of prompt
    
    batch_size, prompt_len = input_ids.shape
    device = input_ids.device
    dtype = next(model.parameters()).dtype
    config = model.config
    
    kv_cache = StaticKVCache(
        num_layers=config.num_hidden_layers,
        batch_size=batch_size,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        max_seq_len=prompt_len + max_new_tokens,
        dtype=dtype,
        device=device,
    )
    
    generated_ids = input_ids.clone()
    
    # 1. Batched Prefill
    prefill_position_ids = torch.arange(
        prompt_len,
        device=device,
        dtype=torch.long
    )
    
    logits = model(
        input_ids=input_ids,
        position_ids=prefill_position_ids,
        kv_cache=kv_cache,
        start_pos=0,
        logits_to_keep=1,
    )
    
    # (B, 1)
    next_token = torch.argmax(
        logits[:, -1, :],
        dim=-1,
        keepdim=True
    )
    
    generated_ids = torch.cat(
        [generated_ids, next_token],
        dim=1
    )
    
    # 2. Batched Decode
    for decode_step in range(max_new_tokens - 1):
        start_pos = prompt_len + decode_step
        decode_position_ids = torch.tensor(
            [start_pos],
            device=device,
            dtype=torch.long
        ).unsqueeze(0)
        
        logits = model(
            input_ids=next_token,
            position_ids=decode_position_ids,
            kv_cache=kv_cache,
            start_pos=start_pos,
            logits_to_keep=1
        )
        
        next_token = torch.argmax(
            logits[:, -1, :],
            dim=-1,
            keepdim=True
        )
        
        generated_ids = torch.cat(
            [generated_ids, next_token],
            dim=1
        )
        
    return generated_ids


@torch.inference_mode()
def batched_left_padding_cached_greedy_generate(
    model,
    config,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int
) -> torch.Tensor:
    batch_size, padded_seq_len = input_ids.shape
    # (B)
    prompt_lengths = attention_mask.sum(dim=-1).long()
    
    max_seq_len = padded_seq_len + max_new_tokens
    
    kv_cache = StaticKVCache(
        num_layers=config.num_hidden_layers,
        batch_size=batch_size,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        max_seq_len=max_seq_len,
        dtype=next(model.parameters()).dtype,
        device=input_ids.device,
    )    
    
    running_attention_mask = attention_mask
    generated_token_ids = input_ids.clone()
    
    # 1. Prefill
    prefill_position_ids = running_attention_mask.long().cumsum(dim=-1) - 1
    prefill_position_ids = prefill_position_ids.masked_fill(running_attention_mask == 0, 0)
    
    logits = model(
        input_ids,
        attention_mask=running_attention_mask,
        position_ids=prefill_position_ids,
        kv_cache=kv_cache,
        start_pos=0,
        logits_to_keep=1
    )
    
    next_tokens = torch.argmax(
        logits[:, -1, :],
        dim=-1,
        keepdim=True
    )
    
    generated_token_ids = torch.cat(
        [generated_token_ids, next_tokens],
        dim=1
    )
    
    for decode_step in range(max_new_tokens - 1):
        new_valid_token_mask = torch.ones(
            batch_size,
            1,
            dtype=running_attention_mask.dtype,
            device=running_attention_mask.device
        )
        
        running_attention_mask = torch.cat(
            [running_attention_mask, new_valid_token_mask],
            dim=1
        )
        
        # (B, 1)
        decode_position_ids = prompt_lengths[:, None] + decode_step
        decode_start_pos = padded_seq_len + decode_step
        
        logits = model(
            input_ids=next_tokens,
            attention_mask=running_attention_mask,
            position_ids=decode_position_ids,
            kv_cache=kv_cache,
            start_pos=decode_start_pos,
            logits_to_keep=1
        )
        
        next_tokens = torch.argmax(
            logits[:, -1, :],
            dim=-1,
            keepdim=True
        )
        
        generated_token_ids = torch.cat(
            [generated_token_ids, next_tokens],
            dim=1
        )
        
    return generated_token_ids


@torch.inference_mode()
def batched_left_padding_cached_greedy_generate_with_eos(
    model,
    config: Qwen3Config,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int,
    pad_token_id: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns output_ids, output_attention_mask, generated_lengths, finished"""
    
    batch_size, padded_seq_len = input_ids.shape
    prompt_length = attention_mask.sum(dim=1).long()
    
    output_ids = input_ids.clone()
    output_attention_mask = attention_mask.clone()
    
    generated_lengths = torch.zeros(
        batch_size,
        dtype=torch.long,
        device=input_ids.device
    )
    
    finished = torch.zeros(
        batch_size,
        dtype=torch.bool,
        device=input_ids.device
    )
    
    max_seq_len = padded_seq_len + max_new_tokens
    
    kv_cache = StaticKVCache(
        num_layers=config.num_hidden_layers,
        batch_size=batch_size,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        max_seq_len=max_seq_len,
        dtype=next(model.parameters()).dtype,
        device=input_ids.device,
    )
    
    running_attention_mask = attention_mask
    
    # 1. Prefill
    prefill_position_ids = running_attention_mask.long().cumsum(dim=1) - 1
    prefill_position_ids = prefill_position_ids.masked_fill(running_attention_mask == 0, 0)
    logits = model(
        input_ids=input_ids,
        attention_mask=running_attention_mask,
        position_ids=prefill_position_ids,
        kv_cache=kv_cache,
        start_pos=0,
        logits_to_keep=1
    )
    
    next_tokens = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    # (B)
    active = ~finished
    
    first_generated_mask = active.to(dtype=output_attention_mask.dtype)[:, None]
    
    output_ids = torch.cat([output_ids, next_tokens], dim=-1)
    output_attention_mask = torch.cat([output_attention_mask, first_generated_mask], dim=-1)
    generated_lengths = generated_lengths + active.long()
    
    # (B)
    newly_finished = active & (next_tokens.squeeze(-1) == eos_token_id)
    
    finished = finished | newly_finished
    
    if bool(finished.all()) or max_new_tokens == 1:
        return (
            output_ids,
            output_attention_mask,
            generated_lengths,
            finished
        )
    
    # 2. Cache decode
    for decode_step in range(max_new_tokens - 1):
        active = ~finished
        decode_input_ids = torch.where(
            active[:, None],
            next_tokens,
            torch.full_like(
                next_tokens,
                fill_value=pad_token_id
            )
        )
        
        cached_valid_mask = active.to(dtype=running_attention_mask.dtype)[:, None]
        running_attention_mask = torch.cat(
            [running_attention_mask, cached_valid_mask],
            dim=-1
        )
        
        decode_position_ids = (prompt_length + generated_lengths - 1)[:, None]
        decode_position_ids = decode_position_ids.masked_fill(~active[:, None], 0)
        
        decode_start_pos = padded_seq_len + decode_step
        
        logits = model(
            input_ids=decode_input_ids,
            attention_mask=running_attention_mask,
            position_ids=decode_position_ids,
            kv_cache=kv_cache,
            start_pos=decode_start_pos,
            logits_to_keep=1,
        )
        
        predicted_tokens = torch.argmax(
            logits[:, -1, :],
            dim=-1,
            keepdim=True
        )
        
        next_tokens = torch.where(
            active[:, None],
            predicted_tokens,
            torch.full_like(
                predicted_tokens,
                fill_value=pad_token_id
            )
        )
        
        generated_valid_mask = active.to(dtype=output_attention_mask.dtype)[:, None]
        output_ids = torch.cat(
            [output_ids, next_tokens],
            dim=-1
        )
        output_attention_mask = torch.cat(
            [output_attention_mask, generated_valid_mask],
            dim=-1
        )
        generated_lengths = generated_lengths + active.long()
        newly_finished = (
            active
            & (
                next_tokens.squeeze(-1) == eos_token_id
            )
        )
        
        finished = finished | newly_finished
        
        if bool(finished.all()):
            break
            
    return (
        output_ids,
        output_attention_mask,
        generated_lengths,
        finished
    )
        