# test_layers.py

from pathlib import Path
import torch

from config import Qwen3Config
from model.layers import Qwen3RMSNorm, Qwen3MLP
from weight_loader import load_safetensors


def load_submodule_weights(module, state_dict, prefix: str):
    """
    从完整 state_dict 里加载某个子模块的权重。

    例如 prefix = "model.layers.0.mlp."
    会把：
      model.layers.0.mlp.gate_proj.weight
    变成：
      gate_proj.weight
    """
    sub_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix): ]
            sub_state_dict[new_key] = value
            
    missing, unexpected = module.load_state_dict(sub_state_dict, strict=True)
    
    if missing:
        print("Missing keys:", missing)
    if unexpected:
        print("Unexpected keys:", unexpected)


def main():
    model_dir = Path("/mnt/yanghui/models/Qwen/Qwen3-4B").expanduser()

    config = Qwen3Config.from_json(model_dir / "config.json")
    state_dict = load_safetensors(model_dir)

    print("=== Test RMSNorm ===")
    norm = Qwen3RMSNorm(
        hidden_size=config.hidden_size,
        eps=config.rms_norm_eps,
    )

    load_submodule_weights(
        norm,
        state_dict,
        prefix="model.layers.0.input_layernorm.",
    )

    x = torch.randn(2, 8, config.hidden_size)
    y = norm(x)

    print("input shape: ", x.shape)
    print("output shape:", y.shape)
    print("output dtype: ", y.dtype)
    print("has nan:     ", torch.isnan(y).any().item())

    print()
    print("=== Test MLP ===")
    mlp = Qwen3MLP(config)

    load_submodule_weights(
        mlp,
        state_dict,
        prefix="model.layers.0.mlp.",
    )

    x = torch.randn(2, 8, config.hidden_size)
    y = mlp(x)

    print("input shape: ", x.shape)
    print("output shape:", y.shape)
    print("output dtype: ", y.dtype)
    print("has nan:     ", torch.isnan(y).any().item())


if __name__ == "__main__":
    main()