# test_static_kv_cache_batch.py

import torch

from cache.kv_cache import StaticKVCache


@torch.inference_mode()
def test_static_kv_cache_batch():
    torch.manual_seed(0)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    dtype = torch.bfloat16

    batch_size = 2
    num_layers = 2
    num_kv_heads = 4
    head_dim = 8
    prefill_len = 3
    max_seq_len = 16

    print("\n===== Test batched StaticKVCache =====")
    print("device:", device)
    print("dtype:", dtype)
    print("batch size:", batch_size)
    print("num layers:", num_layers)
    print("num kv heads:", num_kv_heads)
    print("head dim:", head_dim)
    print("max sequence length:", max_seq_len)

    kv_cache = StaticKVCache(
        num_layers=num_layers,
        batch_size=batch_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_seq_len=max_seq_len,
        dtype=dtype,
        device=device,
    )

    # ---------------------------------------------------------
    # 1. Batched prefill
    # ---------------------------------------------------------
    prefill_key_states = torch.randn(
        batch_size,
        num_kv_heads,
        prefill_len,
        head_dim,
        dtype=dtype,
        device=device,
    )

    prefill_value_states = torch.randn(
        batch_size,
        num_kv_heads,
        prefill_len,
        head_dim,
        dtype=dtype,
        device=device,
    )

    cached_keys, cached_values = kv_cache.update(
        layer_idx=0,
        start_pos=0,
        k_states=prefill_key_states,
        v_states=prefill_value_states,
    )

    expected_prefill_shape = (
        batch_size,
        num_kv_heads,
        prefill_len,
        head_dim,
    )

    print("\n----- Batched prefill -----")
    print("input key shape:", prefill_key_states.shape)
    print("cached key shape:", cached_keys.shape)
    print("cached value shape:", cached_values.shape)

    assert cached_keys.shape == expected_prefill_shape, (
        f"Unexpected cached key shape: "
        f"expected={expected_prefill_shape}, "
        f"actual={tuple(cached_keys.shape)}"
    )

    assert cached_values.shape == expected_prefill_shape, (
        f"Unexpected cached value shape: "
        f"expected={expected_prefill_shape}, "
        f"actual={tuple(cached_values.shape)}"
    )

    assert torch.equal(cached_keys, prefill_key_states), (
        "Cached keys do not match prefill key states"
    )

    assert torch.equal(cached_values, prefill_value_states), (
        "Cached values do not match prefill value states"
    )

    # 确保两个 batch slot 保存的是各自的数据，
    # 而不是错误地写到了同一个位置。
    assert torch.equal(
        cached_keys[0],
        prefill_key_states[0],
    )

    assert torch.equal(
        cached_keys[1],
        prefill_key_states[1],
    )

    # ---------------------------------------------------------
    # 2. Batched single-token append
    # ---------------------------------------------------------
    decode_key_states = torch.randn(
        batch_size,
        num_kv_heads,
        1,
        head_dim,
        dtype=dtype,
        device=device,
    )

    decode_value_states = torch.randn(
        batch_size,
        num_kv_heads,
        1,
        head_dim,
        dtype=dtype,
        device=device,
    )

    cached_keys, cached_values = kv_cache.update(
        layer_idx=0,
        start_pos=prefill_len,
        k_states=decode_key_states,
        v_states=decode_value_states,
    )

    expected_keys = torch.cat(
        [prefill_key_states, decode_key_states],
        dim=2,
    )

    expected_values = torch.cat(
        [prefill_value_states, decode_value_states],
        dim=2,
    )

    expected_decode_shape = (
        batch_size,
        num_kv_heads,
        prefill_len + 1,
        head_dim,
    )

    print("\n----- Batched decode append -----")
    print("decode key shape:", decode_key_states.shape)
    print("cached key shape:", cached_keys.shape)
    print("cached value shape:", cached_values.shape)

    assert cached_keys.shape == expected_decode_shape, (
        f"Unexpected cached key shape after decode: "
        f"expected={expected_decode_shape}, "
        f"actual={tuple(cached_keys.shape)}"
    )

    assert cached_values.shape == expected_decode_shape, (
        f"Unexpected cached value shape after decode: "
        f"expected={expected_decode_shape}, "
        f"actual={tuple(cached_values.shape)}"
    )

    assert torch.equal(cached_keys, expected_keys), (
        "Cached keys are incorrect after batched decode append"
    )

    assert torch.equal(cached_values, expected_values), (
        "Cached values are incorrect after batched decode append"
    )

    print("\nBatched StaticKVCache test passed.")


def main():
    test_static_kv_cache_batch()


if __name__ == "__main__":
    main()