import torch
import time

from generation.greedy import (
    cached_greedy_generate,
    greedy_generate_without_kv_cache,
)
from pathlib import Path
from config import Qwen3Config
from model.qwen3 import Qwen3ForCausalLM
from weight_loader import load_safetensors


@torch.inference_mode()
def test_cached_greedy_generate(
    model,
    input_ids: torch.Tensor,
):
    model.eval()

    max_new_tokens = 16
    prompt_len = input_ids.shape[1]

    print("\n===== Test cached greedy generation =====")
    print("input_ids shape:", input_ids.shape)
    print("prompt length:", prompt_len)
    print("max new tokens:", max_new_tokens)

    reference_ids = greedy_generate_without_kv_cache(
        model=model,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=None,
    )

    cached_ids = cached_greedy_generate(
        model=model,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=None,
    )

    print("reference ids shape:", reference_ids.shape)
    print("cached ids shape:", cached_ids.shape)

    print("reference ids:")
    print(reference_ids)

    print("cached ids:")
    print(cached_ids)

    assert reference_ids.shape == cached_ids.shape, (
        f"shape mismatch: "
        f"reference={reference_ids.shape}, "
        f"cached={cached_ids.shape}"
    )

    reference_new_tokens = reference_ids[:, prompt_len:]
    cached_new_tokens = cached_ids[:, prompt_len:]

    print("\nGenerated token comparison:")

    first_mismatch_step = None

    for step in range(max_new_tokens):
        reference_token = reference_new_tokens[0, step].item()
        cached_token = cached_new_tokens[0, step].item()

        matched = reference_token == cached_token

        print(
            f"step={step:2d}, "
            f"reference_token={reference_token:6d}, "
            f"cached_token={cached_token:6d}, "
            f"matched={matched}"
        )

        if not matched and first_mismatch_step is None:
            first_mismatch_step = step

    if first_mismatch_step is not None:
        raise AssertionError(
            "cached greedy generation mismatch: "
            f"first mismatch at generation step "
            f"{first_mismatch_step}"
        )

    assert torch.equal(reference_ids, cached_ids)

    print("\ncached greedy generation test passed")
    

def synchronize(device: torch.device):
    """CUDA 默认异步执行，计时前后必须同步。"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def benchmark_greedy_generation(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 32,
    warmup_runs: int = 1,
    benchmark_runs: int = 3,
):
    model.eval()
    device = input_ids.device

    generators = {
        "without kv cache": greedy_generate_without_kv_cache,
        "with kv cache": cached_greedy_generate,
    }

    results = {}

    print("\n===== Benchmark greedy generation =====")
    print("prompt length:", input_ids.shape[1])
    print("max new tokens:", max_new_tokens)
    print("warmup runs:", warmup_runs)
    print("benchmark runs:", benchmark_runs)

    for name, generate_fn in generators.items():
        print(f"\n--- {name} ---")

        # 预热：排除首次 CUDA kernel 初始化等额外开销
        for _ in range(warmup_runs):
            generate_fn(
                model=model,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                eos_token_id=None,
            )

        synchronize(device)

        elapsed_times = []
        generated_token_counts = []

        for run in range(benchmark_runs):
            synchronize(device)
            start_time = time.perf_counter()

            generated_ids = generate_fn(
                model=model,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                eos_token_id=None,
            )

            synchronize(device)
            elapsed = time.perf_counter() - start_time

            generated_tokens = (
                generated_ids.shape[1] - input_ids.shape[1]
            )
            tps = generated_tokens / elapsed

            elapsed_times.append(elapsed)
            generated_token_counts.append(generated_tokens)

            print(
                f"run={run + 1}, "
                f"time={elapsed:.4f}s, "
                f"tokens={generated_tokens}, "
                f"TPS={tps:.2f}"
            )

        total_time = sum(elapsed_times)
        total_generated_tokens = sum(generated_token_counts)
        average_tps = total_generated_tokens / total_time
        average_time = total_time / benchmark_runs

        results[name] = {
            "average_time": average_time,
            "average_tps": average_tps,
        }

        print(
            f"average time: {average_time:.4f}s\n"
            f"average TPS:  {average_tps:.2f}"
        )

    no_cache_tps = results["without kv cache"]["average_tps"]
    cached_tps = results["with kv cache"]["average_tps"]
    speedup = cached_tps / no_cache_tps

    print("\n===== Benchmark result =====")
    print(f"without kv cache TPS: {no_cache_tps:.2f}")
    print(f"with kv cache TPS:    {cached_tps:.2f}")
    print(f"speedup:              {speedup:.2f}x")
    
    
    
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

    model = Qwen3ForCausalLM(config).to(
        device=device,
        dtype=dtype,
    )

    state_dict = load_safetensors(model_dir)
    model.load_state_dict(state_dict, strict=False)

    if config.tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight

    model.eval()

    input_ids = torch.tensor(
        [[9707, 11, 358, 1079, 311, 1492, 264, 3891]],
        device=device,
        dtype=torch.long,
    ).repeat(1, 16)

    test_cached_greedy_generate(
        model=model,
        input_ids=input_ids,
    )

    benchmark_greedy_generation(
        model=model,
        input_ids=input_ids,
        max_new_tokens=32,
        warmup_runs=1,
        benchmark_runs=3,
    )


if __name__ == "__main__":
    main()