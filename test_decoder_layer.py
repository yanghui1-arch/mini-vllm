# test_decoder_layer.py

from pathlib import Path
import torch

from config import Qwen3Config
from weight_loader import load_safetensors
from model.qwen3 import Qwen3DecoderLayer
from cache.kv_cache import StaticKVCache


def load_submodule_weights(module, state_dict, prefix: str):
    sub_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            sub_state_dict[new_key] = value

    missing, unexpected = module.load_state_dict(sub_state_dict, strict=True)

    if missing:
        print("Missing keys:", missing)

    if unexpected:
        print("Unexpected keys:", unexpected)

    assert not missing
    assert not unexpected

@torch.inference_mode()
def test_decoder_layer_single_token_decode(layer, config):
    torch.manual_seed(42)

    B = 1
    prompt_len = 8

    dtype = layer.self_attn.q_proj.weight.dtype
    device = layer.self_attn.q_proj.weight.device

    prompt_hidden_states = torch.randn(
        B,
        prompt_len,
        config.hidden_size,
        dtype=dtype,
        device=device,
    )

    new_token_hidden_states = torch.randn(
        B,
        1,
        config.hidden_size,
        dtype=dtype,
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

    # 1. Prefill：写入 prompt 在第 0 层产生的 K/V
    layer(
        prompt_hidden_states,
        position_ids=torch.arange(
            prompt_len,
            device=device,
            dtype=torch.long,
        ),
        kv_cache=kv_cache,
        start_pos=0,
    )

    # 2. Cached decode：只输入新 token
    cached_decode_output = layer(
        new_token_hidden_states,
        position_ids=torch.tensor(
            [prompt_len],
            device=device,
            dtype=torch.long,
        ),
        kv_cache=kv_cache,
        start_pos=prompt_len,
    )

    # 3. Reference：一次性计算 prompt + 新 token
    full_hidden_states = torch.cat(
        [
            prompt_hidden_states,
            new_token_hidden_states,
        ],
        dim=1,
    )

    full_output = layer(
        full_hidden_states,
        position_ids=torch.arange(
            prompt_len + 1,
            device=device,
            dtype=torch.long,
        ),
    )

    reference_decode_output = full_output[:, -1:, :]

    diff = (
        cached_decode_output.float()
        - reference_decode_output.float()
    ).abs()

    print("cached decode shape:   ", cached_decode_output.shape)
    print("reference decode shape:", reference_decode_output.shape)
    print("max diff:              ", diff.max().item())
    print("mean diff:             ", diff.mean().item())
    print(
        "has nan:               ",
        torch.isnan(cached_decode_output).any().item(),
    )

    assert cached_decode_output.shape == (
        B,
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

    print("DecoderLayer single-token decode test passed!")

def main():
    model_dir = Path("/mnt/yanghui/models/Qwen/Qwen3-4B").expanduser()

    config = Qwen3Config.from_json(model_dir / "config.json")
    state_dict = load_safetensors(model_dir)

    layer = Qwen3DecoderLayer(
        config=config,
        layer_idx=0,
    )

    load_submodule_weights(
        layer,
        state_dict,
        prefix="model.layers.0.",
    )

    layer.eval()

    B = 2
    T = 8

    x = torch.randn(B, T, config.hidden_size)
    position_ids = torch.arange(T)

    with torch.no_grad():
        y = layer(
            hidden_states=x,
            position_ids=position_ids,
        )

    print("input shape: ", x.shape)
    print("output shape:", y.shape)
    print("output dtype: ", y.dtype)
    print("has nan:     ", torch.isnan(y).any().item())

    assert y.shape == x.shape
    assert not torch.isnan(y).any()
    
    layer.eval()
    test_decoder_layer_single_token_decode(layer, config)

if __name__ == "__main__":
    main()