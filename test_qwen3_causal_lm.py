# test_qwen3_causal_lm.py

import gc
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from config import Qwen3Config
from model.qwen3 import Qwen3ForCausalLM
from cache.kv_cache import StaticKVCache
from weight_loader import load_safetensors


def load_model_weights(module, state_dict):
    missing, unexpected = module.load_state_dict(
        state_dict,
        strict=False,
    )

    # Qwen3-4B 开启了 tie_word_embeddings，
    # checkpoint 可能没有单独保存 lm_head.weight。
    real_missing = [
        key
        for key in missing
        if key != "lm_head.weight"
    ]

    if real_missing:
        print("Missing keys:", real_missing)

    if unexpected:
        print("Unexpected keys:", unexpected)

    assert not real_missing
    assert not unexpected


def main():
    model_dir = Path(
        "/mnt/yanghui/models/Qwen/Qwen3-4B"
    ).expanduser()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    dtype = torch.bfloat16

    config = Qwen3Config.from_json(
        model_dir / "config.json"
    )

    input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(1, 8),
        dtype=torch.long,
        device=device,
    )

    position_ids = torch.arange(
        input_ids.shape[1],
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    print("input_ids shape:", input_ids.shape)
    print("device:         ", device)
    print("dtype:          ", dtype)

    # =========================================================
    # Hugging Face official implementation
    # =========================================================

    print()
    print("=== Run Hugging Face Qwen3ForCausalLM ===")

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        attn_implementation="eager",
        trust_remote_code=True,
    )

    hf_model = hf_model.to(device)
    hf_model.eval()

    with torch.inference_mode():
        hf_outputs = hf_model(
            input_ids=input_ids,
            position_ids=position_ids,
            use_cache=False,
        )

        # 只保留最后一个位置，减少后续 CPU 内存占用。
        hf_logits = hf_outputs.logits[:, -1:, :]

    hf_logits = hf_logits.float().cpu()

    print("HF logits shape:", hf_logits.shape)
    print("HF logits dtype:", hf_logits.dtype)
    print(
        "HF has nan:     ",
        torch.isnan(hf_logits).any().item(),
    )

    del hf_outputs
    del hf_model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # =========================================================
    # Our implementation
    # =========================================================

    print()
    print("=== Run Mini Qwen3ForCausalLM ===")

    state_dict = load_safetensors(model_dir)

    model = Qwen3ForCausalLM(config)

    load_model_weights(
        model,
        state_dict,
    )

    # 如果 checkpoint 没有 lm_head.weight，
    # 让 lm_head 和 embedding 共享权重。
    if config.tie_word_embeddings:
        model.tie_weights()

    del state_dict
    gc.collect()

    model = model.to(
        device=device,
        dtype=dtype,
    )
    model.eval()

    with torch.inference_mode():
        mini_logits = model(
            input_ids=input_ids,
            position_ids=position_ids,
            logits_to_keep=1,
        )

    mini_logits = mini_logits.float().cpu()

    print("Mini logits shape:", mini_logits.shape)
    print("Mini logits dtype:", mini_logits.dtype)
    print(
        "Mini has nan:     ",
        torch.isnan(mini_logits).any().item(),
    )

    # =========================================================
    # Compare logits
    # =========================================================

    print()
    print("=== Compare logits ===")

    assert hf_logits.shape == mini_logits.shape

    diff = torch.abs(
        hf_logits - mini_logits
    )

    print("max diff: ", diff.max().item())
    print("mean diff:", diff.mean().item())

    hf_next_token = torch.argmax(
        hf_logits[:, -1, :],
        dim=-1,
    )

    mini_next_token = torch.argmax(
        mini_logits[:, -1, :],
        dim=-1,
    )

    print("HF next token:  ", hf_next_token.item())
    print("Mini next token:", mini_next_token.item())

    print(
        "same next token:",
        torch.equal(
            hf_next_token,
            mini_next_token,
        ),
    )

    torch.testing.assert_close(
        mini_logits,
        hf_logits,
        rtol=1e-1,
        atol=1e-1,
    )

    print()
    print("Qwen3ForCausalLM logits test passed.")
    
        # =========================================================
    # Test Mini Qwen3ForCausalLM with KV Cache
    # =========================================================

    print()
    print("=== Test Mini Qwen3ForCausalLM KV Cache ===")

    prompt_len = input_ids.shape[1]

    kv_cache = StaticKVCache(
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        max_seq_len=128,
        dtype=dtype,
        device=device,
    )

    with torch.inference_mode():
        # 1. Cached prefill：处理整个 prompt，并写入 KV Cache
        cached_prefill_logits = model(
            input_ids=input_ids,
            position_ids=position_ids,
            kv_cache=kv_cache,
            start_pos=0,
            logits_to_keep=1,
        )

        # cached_prefill_logits: [1, 1, vocab_size]
        next_token_ids = torch.argmax(
            cached_prefill_logits,
            dim=-1,
        )

        # next_token_ids: [1, 1]
        print("next token ids shape:", next_token_ids.shape)
        print("next token id:       ", next_token_ids.item())

        # 2. Cached decode：只输入刚刚预测出来的一个 token
        decode_position_ids = torch.tensor(
            [[prompt_len]],
            dtype=torch.long,
            device=device,
        )

        cached_decode_logits = model(
            input_ids=next_token_ids,
            position_ids=decode_position_ids,
            kv_cache=kv_cache,
            start_pos=prompt_len,
            logits_to_keep=1,
        )

        # 3. Reference：不用 Cache，完整计算 prompt + next_token
        full_input_ids = torch.cat(
            [input_ids, next_token_ids],
            dim=1,
        )

        full_position_ids = torch.arange(
            full_input_ids.shape[1],
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)

        reference_decode_logits = model(
            input_ids=full_input_ids,
            position_ids=full_position_ids,
            kv_cache=None,
            start_pos=0,
            logits_to_keep=1,
        )

    # 转成 FP32 并移动到 CPU 后比较
    cached_prefill_logits_cpu = (
        cached_prefill_logits.float().cpu()
    )

    cached_decode_logits_cpu = (
        cached_decode_logits.float().cpu()
    )

    reference_decode_logits_cpu = (
        reference_decode_logits.float().cpu()
    )

    # ---------------------------------------------------------
    # Compare cached prefill with original no-cache prefill
    # ---------------------------------------------------------

    prefill_diff = torch.abs(
        cached_prefill_logits_cpu - mini_logits
    )

    print()
    print("Cached prefill logits shape:", cached_prefill_logits_cpu.shape)
    print("Cached prefill max diff:    ", prefill_diff.max().item())
    print("Cached prefill mean diff:   ", prefill_diff.mean().item())

    torch.testing.assert_close(
        cached_prefill_logits_cpu,
        mini_logits,
        rtol=1e-4,
        atol=1e-4,
    )

    # ---------------------------------------------------------
    # Compare cached decode with full recomputation
    # ---------------------------------------------------------

    decode_diff = torch.abs(
        cached_decode_logits_cpu
        - reference_decode_logits_cpu
    )

    print()
    print("Cached decode logits shape:   ", cached_decode_logits_cpu.shape)
    print("Reference decode logits shape:", reference_decode_logits_cpu.shape)
    print("Cached decode max diff:       ", decode_diff.max().item())
    print("Cached decode mean diff:      ", decode_diff.mean().item())

    cached_next_token = torch.argmax(
        cached_decode_logits_cpu[:, -1, :],
        dim=-1,
    )

    reference_next_token = torch.argmax(
        reference_decode_logits_cpu[:, -1, :],
        dim=-1,
    )

    print("Cached next token:   ", cached_next_token.item())
    print("Reference next token:", reference_next_token.item())
    print(
        "Same next token:     ",
        torch.equal(
            cached_next_token,
            reference_next_token,
        ),
    )

    assert cached_decode_logits_cpu.shape == (
        1,
        1,
        config.vocab_size,
    )

    assert reference_decode_logits_cpu.shape == (
        1,
        1,
        config.vocab_size,
    )

    assert not torch.isnan(cached_decode_logits_cpu).any()
    assert not torch.isnan(reference_decode_logits_cpu).any()

    assert torch.equal(
        cached_next_token,
        reference_next_token,
    )

    torch.testing.assert_close(
        cached_decode_logits_cpu,
        reference_decode_logits_cpu,
        rtol=1e-1,
        atol=1e-1,
    )

    print()
    print("Qwen3ForCausalLM KV Cache test passed.")


if __name__ == "__main__":
    main()