# test_attention.py

from pathlib import Path
import torch

from config import Qwen3Config
from weight_loader import load_safetensors
from model.attention import Qwen3Attention


def load_submodule_weights(module, state_dict, prefix: str):
    sub_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            sub_state_dict[new_key] = value

    missing, unexpected = module.load_state_dict(sub_state_dict, strict=False)
    
    # it's not nn.Parameter
    allowed_missing = {"rotary_emb.inv_freq"}

    real_missing = [k for k in missing if k not in allowed_missing]

    if real_missing:
        print("Missing keys:", real_missing)
    if unexpected:
        print("Unexpected keys:", unexpected)

    assert not real_missing
    assert not unexpected


def main():
    model_dir = Path("/mnt/yanghui/models/Qwen/Qwen3-4B").expanduser()

    config = Qwen3Config.from_json(model_dir / "config.json")
    state_dict = load_safetensors(model_dir)

    attn = Qwen3Attention(config, layer_idx=0)

    load_submodule_weights(
        attn,
        state_dict,
        prefix="model.layers.0.self_attn.",
    )

    attn.eval()

    B = 2
    T = 8

    x = torch.randn(B, T, config.hidden_size)
    position_ids = torch.arange(T)

    with torch.no_grad():
        y = attn(x, position_ids=position_ids)

    print("input shape: ", x.shape)
    print("output shape:", y.shape)
    print("output dtype: ", y.dtype)
    print("has nan:     ", torch.isnan(y).any().item())

    assert y.shape == x.shape
    assert not torch.isnan(y).any()


if __name__ == "__main__":
    main()