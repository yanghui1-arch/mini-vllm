# test_batched_decode.py

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
def test_batched_decode(
    model: Qwen3ForCausalLM,
    config: Qwen3Config,
    input_ids: torch.Tensor,
):
    model.eval()

    batch_size, prompt_len = input_ids.shape
    device = input_ids.device
    dtype = next(model.parameters()).dtype

    assert batch_size == 2, (
        f"This test expects batch_size=2, "
        f"but got batch_size={batch_size}"
    )

    print("\n===== Test batched single-token decode =====")
    print("input_ids shape:", input_ids.shape)
    print("batch size:", batch_size)
    print("prompt length:", prompt_len)
    print("device:", device)
    print("model dtype:", dtype)

    prefill_position_ids = torch.arange(
        prompt_len,
        device=device,
        dtype=torch.long,
    )

    decode_position_ids = torch.tensor(
        [prompt_len],
        device=device,
        dtype=torch.long,
    )

    # =========================================================
    # 1. Batched path
    #
    # 两个请求一起 prefill，再一起 decode 一次。
    # =========================================================
    batched_kv_cache = create_kv_cache(
        config=config,
        batch_size=batch_size,
        max_seq_len=prompt_len + 1,
        dtype=dtype,
        device=device,
    )

    batched_prefill_logits = model(
        input_ids=input_ids,
        position_ids=prefill_position_ids,
        kv_cache=batched_kv_cache,
        start_pos=0,
        logits_to_keep=1,
    )

    batched_decode_input_ids = torch.argmax(
        batched_prefill_logits[:, -1, :],
        dim=-1,
        keepdim=True,
    )

    print("\n----- Batched prefill -----")
    print(
        "batched prefill logits shape:",
        batched_prefill_logits.shape,
    )
    print(
        "batched decode input shape:",
        batched_decode_input_ids.shape,
    )
    print(
        "batched decode input tokens:",
        batched_decode_input_ids.squeeze(-1),
    )

    assert batched_decode_input_ids.shape == (
        batch_size,
        1,
    ), (
        f"Unexpected batched decode input shape: "
        f"{tuple(batched_decode_input_ids.shape)}"
    )

    batched_decode_logits = model(
        input_ids=batched_decode_input_ids,
        position_ids=decode_position_ids,
        kv_cache=batched_kv_cache,
        start_pos=prompt_len,
        logits_to_keep=1,
    )

    batched_next_tokens = torch.argmax(
        batched_decode_logits[:, -1, :],
        dim=-1,
    )

    print("\n----- Batched decode -----")
    print(
        "batched decode logits shape:",
        batched_decode_logits.shape,
    )
    print(
        "tokens predicted after decode:",
        batched_next_tokens,
    )

    expected_logits_shape = (
        batch_size,
        1,
        config.vocab_size,
    )

    assert batched_decode_logits.shape == expected_logits_shape, (
        f"Unexpected batched decode logits shape: "
        f"expected={expected_logits_shape}, "
        f"actual={tuple(batched_decode_logits.shape)}"
    )

    # =========================================================
    # 2. Reference path
    #
    # 每个请求创建自己的 KV Cache，分别执行：
    # prefill -> single-token decode
    # =========================================================
    reference_decode_logits_list = []
    reference_decode_input_tokens = []

    print("\n----- Separate single-request decode -----")

    for request_idx in range(batch_size):
        single_input_ids = input_ids[
            request_idx : request_idx + 1
        ]

        single_kv_cache = create_kv_cache(
            config=config,
            batch_size=1,
            max_seq_len=prompt_len + 1,
            dtype=dtype,
            device=device,
        )

        single_prefill_logits = model(
            input_ids=single_input_ids,
            position_ids=prefill_position_ids,
            kv_cache=single_kv_cache,
            start_pos=0,
            logits_to_keep=1,
        )

        single_decode_input_id = torch.argmax(
            single_prefill_logits[:, -1, :],
            dim=-1,
            keepdim=True,
        )

        # 前一步已经验证过 batched prefill。
        # 这里再次确保 batched 和 single path
        # 为 decode 产生了相同的输入 token。
        batched_decode_token = batched_decode_input_ids[
            request_idx,
            0,
        ].item()

        single_decode_token = single_decode_input_id[
            0,
            0,
        ].item()

        assert batched_decode_token == single_decode_token, (
            f"Prefill next-token mismatch for request "
            f"{request_idx}: "
            f"batched={batched_decode_token}, "
            f"single={single_decode_token}"
        )

        single_decode_logits = model(
            input_ids=single_decode_input_id,
            position_ids=decode_position_ids,
            kv_cache=single_kv_cache,
            start_pos=prompt_len,
            logits_to_keep=1,
        )

        print(
            f"request={request_idx}, "
            f"decode_input_token={single_decode_token}, "
            f"decode_logits_shape="
            f"{single_decode_logits.shape}"
        )

        reference_decode_input_tokens.append(
            single_decode_token
        )

        reference_decode_logits_list.append(
            single_decode_logits
        )

    reference_decode_logits = torch.cat(
        reference_decode_logits_list,
        dim=0,
    )

    reference_next_tokens = torch.argmax(
        reference_decode_logits[:, -1, :],
        dim=-1,
    )

    print("\nreference decode logits shape:",
          reference_decode_logits.shape)
    print("reference decode input tokens:",
          reference_decode_input_tokens)
    print("reference predicted tokens:",
          reference_next_tokens)

    assert (
        reference_decode_logits.shape
        == batched_decode_logits.shape
    ), (
        f"Decode logits shape mismatch: "
        f"reference="
        f"{tuple(reference_decode_logits.shape)}, "
        f"batched="
        f"{tuple(batched_decode_logits.shape)}"
    )

    # =========================================================
    # 3. Compare logits and predicted tokens
    # =========================================================
    batched_logits_fp32 = batched_decode_logits.float()
    reference_logits_fp32 = (
        reference_decode_logits.float()
    )

    abs_diff = torch.abs(
        batched_logits_fp32
        - reference_logits_fp32
    )

    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()

    print("\n----- Decode comparison -----")
    print(f"max absolute difference:  {max_abs_diff:.6f}")
    print(f"mean absolute difference: {mean_abs_diff:.6f}")
    print("batched next tokens:", batched_next_tokens)
    print("reference next tokens:", reference_next_tokens)

    for request_idx in range(batch_size):
        request_max_diff = abs_diff[
            request_idx
        ].max().item()

        batched_token = batched_next_tokens[
            request_idx
        ].item()

        reference_token = reference_next_tokens[
            request_idx
        ].item()

        matched = batched_token == reference_token

        print(
            f"request={request_idx}, "
            f"max_diff={request_max_diff:.6f}, "
            f"batched_token={batched_token}, "
            f"reference_token={reference_token}, "
            f"matched={matched}"
        )

        assert matched, (
            f"Decode next-token mismatch for request "
            f"{request_idx}: "
            f"batched={batched_token}, "
            f"reference={reference_token}"
        )

    # BF16 在 batch_size=1 和 batch_size=2 时可能使用
    # 不同的 GEMM 计算路径，因此不要求逐元素完全相等。
    #
    # 这个阈值主要用于发现明显的 cache 位置、batch 维度
    # 或 attention mask 错误。
    assert max_abs_diff <= 0.5, (
        f"Batched decode logits differ too much from "
        f"single-request decode logits: "
        f"max_abs_diff={max_abs_diff}"
    )

    print("\nBatched single-token decode test passed.")


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

    test_batched_decode(
        model=model,
        config=config,
        input_ids=input_ids,
    )


if __name__ == "__main__":
    main()