from pathlib import Path

from config import Qwen3Config
from weight_loader import load_safetensors, print_weight_shapes

def main():
    model_dir = Path("/mnt/yanghui/models/Qwen/Qwen3-4B").expanduser()
    
    config = Qwen3Config.from_json(model_dir / "config.json")
    print("=== Config ===")
    print(config)

    print()
    print("=== Loading weights ===")
    state_dict = load_safetensors(model_dir)

    print()
    print("=== Weight shapes ===")
    print_weight_shapes(state_dict)

    print()
    print(f"Total tensors: {len(state_dict)}")

if __name__ == "__main__":
    main()    
