from pathlib import Path

import torch

from config import Qwen3Config
from cache.kv_cache import StaticKVCache
from model.qwen3 import Qwen3ForCausalLM
from weight_loader import load_safetensors


def create_kv_cache(
    config: Qwen3Config,
    batch_size: int,
    max_seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> StaticKVCache:
    return StaticKVCache(
        num_layers=config.num_hidden_layers,
        batch_size=batch_size,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        max_seq_len=max_seq_len,
        dtype=dtype,
        device=device,
    )

@torch.inference_mode()
def test_batched_prefill(
    model: Qwen3ForCausalLM,
    config: Qwen3Config,
    input_ids: torch.Tensor,
):
    model.eval()

    batch_size, prompt_len = input_ids.shape
    device = input_ids.device
    dtype = next(model.parameters()).dtype

    print("\n===== Test batched prefill =====")
    print("input_ids shape:", input_ids.shape)
    print("batch size:", batch_size)
    print("prompt length:", prompt_len)
    print("device:", device)
    print("model dtype:", dtype)

    assert batch_size == 2, (
        f"This test expects batch_size=2, "
        f"but got batch_size={batch_size}"
    )

    # 两个 prompt 等长，所以它们使用相同的位置：
    # 0, 1, 2, ..., prompt_len - 1
    position_ids = torch.arange(
        prompt_len,
        device=device,
        dtype=torch.long,
    )

    # ---------------------------------------------------------
    # 1. 两个请求一起进行 batched prefill
    # ---------------------------------------------------------
    batched_kv_cache = create_kv_cache(
        config=config,
        batch_size=batch_size,
        max_seq_len=prompt_len,
        dtype=dtype,
        device=device,
    )

    batched_logits = model(
        input_ids=input_ids,
        position_ids=position_ids,
        kv_cache=batched_kv_cache,
        start_pos=0,
        logits_to_keep=1,
    )

    print("\n----- Batched prefill -----")
    print("batched logits shape:", batched_logits.shape)

    expected_logits_shape = (
        batch_size,
        1,
        config.vocab_size,
    )

    assert batched_logits.shape == expected_logits_shape, (
        f"Unexpected batched logits shape: "
        f"expected={expected_logits_shape}, "
        f"actual={tuple(batched_logits.shape)}"
    )

    # ---------------------------------------------------------
    # 2. 两个请求分别进行单请求 prefill
    # ---------------------------------------------------------
    single_logits_list = []

    print("\n----- Separate single-request prefill -----")

    for request_idx in range(batch_size):
        single_input_ids = input_ids[
            request_idx : request_idx + 1
        ]

        single_kv_cache = create_kv_cache(
            config=config,
            batch_size=1,
            max_seq_len=prompt_len,
            dtype=dtype,
            device=device,
        )

        single_logits = model(
            input_ids=single_input_ids,
            position_ids=position_ids,
            kv_cache=single_kv_cache,
            start_pos=0,
            logits_to_keep=1,
        )

        print(
            f"request={request_idx}, "
            f"input shape={single_input_ids.shape}, "
            f"logits shape={single_logits.shape}"
        )

        single_logits_list.append(single_logits)

    reference_logits = torch.cat(
        single_logits_list,
        dim=0,
    )

    print("\nreference logits shape:", reference_logits.shape)

    assert reference_logits.shape == batched_logits.shape, (
        f"Logits shape mismatch: "
        f"reference={tuple(reference_logits.shape)}, "
        f"batched={tuple(batched_logits.shape)}"
    )

    # ---------------------------------------------------------
    # 3. 比较 logits 和 greedy next token
    # ---------------------------------------------------------
    batched_logits_fp32 = batched_logits.float()
    reference_logits_fp32 = reference_logits.float()

    abs_diff = torch.abs(
        batched_logits_fp32 - reference_logits_fp32
    )

    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()

    batched_next_tokens = torch.argmax(
        batched_logits[:, -1, :],
        dim=-1,
    )

    reference_next_tokens = torch.argmax(
        reference_logits[:, -1, :],
        dim=-1,
    )

    print("\n----- Comparison -----")
    print(f"max absolute difference:  {max_abs_diff:.6f}")
    print(f"mean absolute difference: {mean_abs_diff:.6f}")
    print("batched next tokens:", batched_next_tokens)
    print("reference next tokens:", reference_next_tokens)

    for request_idx in range(batch_size):
        request_diff = abs_diff[request_idx].max().item()

        batched_token = batched_next_tokens[
            request_idx
        ].item()

        reference_token = reference_next_tokens[
            request_idx
        ].item()

        matched = batched_token == reference_token

        print(
            f"request={request_idx}, "
            f"max_diff={request_diff:.6f}, "
            f"batched_token={batched_token}, "
            f"reference_token={reference_token}, "
            f"matched={matched}"
        )

        assert matched, (
            f"Next token mismatch for request {request_idx}: "
            f"batched={batched_token}, "
            f"reference={reference_token}"
        )

    # BF16 中，batch size 不同可能让底层矩阵乘法选择不同的
    # 计算路径，因此 logits 不一定逐元素完全相同。
    assert max_abs_diff <= 0.25, (
        f"Batched prefill logits differ too much from "
        f"single-request logits: max_abs_diff={max_abs_diff}"
    )

    print("\nBatched prefill test passed.")


def main():
    torch.manual_seed(0)

    model_dir = Path(
        "/mnt/yanghui/models/Qwen/Qwen3-4B"
    ).expanduser()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    dtype = torch.bfloat16

    print("===== Load model =====")
    print("model directory:", model_dir)
    print("device:", device)
    print("dtype:", dtype)

    config = Qwen3Config.from_json(
        model_dir / "config.json"
    )

    model = Qwen3ForCausalLM(config).to(
        device=device,
        dtype=dtype,
    )

    state_dict = load_safetensors(model_dir)

    model.load_state_dict(
        state_dict,
        strict=False,
    )

    if config.tie_word_embeddings:
        model.lm_head.weight = (
            model.model.embed_tokens.weight
        )

    model.eval()

    # 两个不同但等长的 prompt。
    input_ids = torch.tensor(
        [
            [
                9707,
                11,
                358,
                1079,
                311,
                1492,
                264,
                3891,
            ],
            [
                9707,
                11,
                358,
                1079,
                311,
                1492,
                264,
                220,
            ],
        ],
        device=device,
        dtype=torch.long,
    )

    test_batched_prefill(
        model=model,
        config=config,
        input_ids=input_ids,
    )


if __name__ == "__main__":
    main()