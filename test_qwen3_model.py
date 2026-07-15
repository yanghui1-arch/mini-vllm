# test_qwen3_model.py

from pathlib import Path
import torch

from config import Qwen3Config
from weight_loader import load_safetensors
from model.qwen3 import Qwen3Model
from cache.kv_cache import StaticKVCache


def load_model_weights(module, state_dict, prefix: str):
    sub_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            sub_state_dict[new_key] = value

    missing, unexpected = module.load_state_dict(sub_state_dict, strict=True)

    if missing:
        print("Missing keys:")
        for k in missing:
            print("  ", k)

    if unexpected:
        print("Unexpected keys:")
        for k in unexpected:
            print("  ", k)

    assert not missing
    assert not unexpected


@torch.inference_mode()
def test_model_single_token_decode(model, config):
    torch.manual_seed(42)

    device = model.embed_tokens.weight.device
    dtype = model.embed_tokens.weight.dtype

    batch_size = 1
    prompt_len = 8

    prompt_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(batch_size, prompt_len),
        device=device,
    )

    new_token_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(batch_size, 1),
        device=device,
    )

    kv_cache = StaticKVCache(
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        max_seq_len=128,
        dtype=dtype,
        device=device,
    )

    # 1. Prefill：将 prompt 的 K/V 写入所有层的 Cache
    model(
        input_ids=prompt_ids,
        kv_cache=kv_cache,
        start_pos=0,
    )

    # 2. Cached decode：只输入一个新 token
    cached_decode_output = model(
        input_ids=new_token_ids,
        kv_cache=kv_cache,
        start_pos=prompt_len,
    )

    # 3. Reference：完整计算 prompt + 新 token
    full_input_ids = torch.cat(
        [prompt_ids, new_token_ids],
        dim=1,
    )

    full_output = model(
        input_ids=full_input_ids,
    )

    reference_decode_output = full_output[:, -1:, :]

    diff = (
        cached_decode_output.float()
        - reference_decode_output.float()
    ).abs()

    print("cached output shape:   ", cached_decode_output.shape)
    print("reference output shape:", reference_decode_output.shape)
    print("max diff:              ", diff.max().item())
    print("mean diff:             ", diff.mean().item())
    print(
        "has nan:               ",
        torch.isnan(cached_decode_output).any().item(),
    )

    assert cached_decode_output.shape == (
        batch_size,
        1,
        config.hidden_size,
    )

    assert not torch.isnan(cached_decode_output).any()

    torch.testing.assert_close(
        cached_decode_output,
        reference_decode_output,
        rtol=1e-4,
        atol=1e-4,
    )

    # 检查每一层的 prompt + 新 token 都已写入
    expected_cache_len = prompt_len + 1

    for layer_idx in range(config.num_hidden_layers):
        cached_k = kv_cache.k_cache[
            layer_idx,
            :,
            :,
            :expected_cache_len,
            :,
        ]

        cached_v = kv_cache.v_cache[
            layer_idx,
            :,
            :,
            :expected_cache_len,
            :,
        ]

        assert not torch.isnan(cached_k).any()
        assert not torch.isnan(cached_v).any()

    print("Qwen3Model single-token decode test passed!")
    
def main():
    model_dir = Path("/mnt/yanghui/models/Qwen/Qwen3-4B").expanduser()

    config = Qwen3Config.from_json(model_dir / "config.json")
    state_dict = load_safetensors(model_dir)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    model = Qwen3Model(config)
    model = model.to(device=device, dtype=dtype)

    load_model_weights(
        model,
        state_dict,
        prefix="model.",
    )

    model.eval()

    B = 1
    T = 4

    input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(B, T),
        device=device,
    )

    position_ids = torch.arange(
        T,
        device=device,
        dtype=torch.long,
    )

    with torch.no_grad():
        hidden_states = model(
            input_ids=input_ids,
            position_ids=position_ids,
        )

    print("input_ids shape:     ", input_ids.shape)
    print("hidden_states shape: ", hidden_states.shape)
    print("hidden_states dtype: ", hidden_states.dtype)
    print("has nan:             ", torch.isnan(hidden_states).any().item())

    assert hidden_states.shape == (B, T, config.hidden_size)
    assert not torch.isnan(hidden_states).any()


if __name__ == "__main__":
    main()