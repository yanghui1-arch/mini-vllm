from pathlib import Path
from safetensors.torch import load_file
import torch

def find_safetensor_files(model_dir: str | Path) -> list[Path]:
    model_dir = Path(model_dir)
    files = sorted(model_dir.glob("*.safetensors"))
    
    if not files:
        raise FileNotFoundError(f"No .safetensors files found in {model_dir}")
    
    return files

def load_safetensors(model_dir: str | Path, device: str = "cpu") -> dict[str, torch.Tensor]:
    files = find_safetensor_files(model_dir)
    
    state_dict = {}
    
    for file in files:
        part = load_file(file, device=device)
        for key, value in part.items():
            if key in state_dict:
                raise ValueError(f"Duplicated weight key found: {key}")
            state_dict[key] = value
            
    return state_dict

def print_weight_shapes(state_dict: dict[str, torch.Tensor]) -> None:
    for name, tensor in state_dict.items():
        print(f"{name:80s} {tuple(tensor.shape)} {tensor.dtype}")
