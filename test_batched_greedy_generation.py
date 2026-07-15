# test_batched_generation.py
# Because cuda keneral is not the same when applying different batch size
# we allow different tokens when logits of top1 and top2 are limit.

from pathlib import Path

import torch

from cache.kv_cache import StaticKVCache
from config import Qwen3Config
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


def get_top2(logits: torch.Tensor):
    """
    Args:
        logits: [vocab_size]

    Returns:
        top1_token
        top1_logit
        top2_token
        top2_logit
        gap
    """
    values, indices = torch.topk(
        logits.float(),
        k=2,
        dim=-1,
    )

    top1_token = indices[0].item()
    top2_token = indices[1].item()

    top1_logit = values[0].item()
    top2_logit = values[1].item()

    gap = top1_logit - top2_logit

    return (
        top1_token,
        top1_logit,
        top2_token,
        top2_logit,
        gap,
    )


@torch.inference_mode()
def test_batched_generation_stepwise(
    model: Qwen3ForCausalLM,
    config: Qwen3Config,
    input_ids: torch.Tensor,
):
    model.eval()

    batch_size, prompt_len = input_ids.shape
    max_new_tokens = 16

    device = input_ids.device
    dtype = next(model.parameters()).dtype

    print("\n===== Test batched generation step by step =====")
    print("input_ids shape:", input_ids.shape)
    print("batch size:", batch_size)
    print("prompt length:", prompt_len)
    print("max new tokens:", max_new_tokens)
    print("device:", device)
    print("model dtype:", dtype)

    assert batch_size == 2

    # ---------------------------------------------------------
    # 创建 batched cache
    # ---------------------------------------------------------
    batched_kv_cache = create_kv_cache(
        config=config,
        batch_size=batch_size,
        max_seq_len=prompt_len + max_new_tokens,
        dtype=dtype,
        device=device,
    )

    # ---------------------------------------------------------
    # 每个请求分别创建 single-request cache
    # ---------------------------------------------------------
    single_kv_caches = [
        create_kv_cache(
            config=config,
            batch_size=1,
            max_seq_len=prompt_len + max_new_tokens,
            dtype=dtype,
            device=device,
        )
        for _ in range(batch_size)
    ]

    prefill_position_ids = torch.arange(
        prompt_len,
        device=device,
        dtype=torch.long,
    )

    # =========================================================
    # 1. Batched prefill
    # =========================================================
    batched_logits = model(
        input_ids=input_ids,
        position_ids=prefill_position_ids,
        kv_cache=batched_kv_cache,
        start_pos=0,
        logits_to_keep=1,
    )

    # =========================================================
    # 2. Separate single-request prefill
    # =========================================================
    single_logits_list = []

    for request_idx in range(batch_size):
        single_logits = model(
            input_ids=input_ids[
                request_idx : request_idx + 1
            ],
            position_ids=prefill_position_ids,
            kv_cache=single_kv_caches[request_idx],
            start_pos=0,
            logits_to_keep=1,
        )

        single_logits_list.append(single_logits)

    reference_logits = torch.cat(
        single_logits_list,
        dim=0,
    )

    generated_tokens = []
    ambiguous_steps = []

    # =========================================================
    # 3. 每一步使用 batched path 生成的 token 作为共同输入
    #
    # 这样 batched 和 single path 的 token 历史始终相同。
    # =========================================================
    for generation_step in range(max_new_tokens):
        print(
            f"\n----- Generation step "
            f"{generation_step} -----"
        )

        abs_diff = torch.abs(
            batched_logits.float()
            - reference_logits.float()
        )

        global_max_diff = abs_diff.max().item()
        global_mean_diff = abs_diff.mean().item()

        print(
            f"global max diff:  {global_max_diff:.6f}"
        )
        print(
            f"global mean diff: {global_mean_diff:.6f}"
        )

        # 之前的测试中观察到 BF16 最大误差约为
        # 0.17～0.25。这里留出一定余量。
        assert global_max_diff <= 0.5, (
            f"Logits differ too much at generation step "
            f"{generation_step}: "
            f"max_diff={global_max_diff}"
        )

        batched_next_tokens = torch.argmax(
            batched_logits[:, -1, :],
            dim=-1,
            keepdim=True,
        )

        reference_next_tokens = torch.argmax(
            reference_logits[:, -1, :],
            dim=-1,
            keepdim=True,
        )

        for request_idx in range(batch_size):
            batched_request_logits = batched_logits[
                request_idx,
                -1,
                :,
            ]

            reference_request_logits = reference_logits[
                request_idx,
                -1,
                :,
            ]

            request_max_diff = abs_diff[
                request_idx
            ].max().item()

            (
                batched_top1,
                batched_top1_logit,
                batched_top2,
                batched_top2_logit,
                batched_gap,
            ) = get_top2(batched_request_logits)

            (
                reference_top1,
                reference_top1_logit,
                reference_top2,
                reference_top2_logit,
                reference_gap,
            ) = get_top2(reference_request_logits)

            matched = batched_top1 == reference_top1

            print(
                f"\nrequest={request_idx}"
            )
            print(
                f"max diff: {request_max_diff:.6f}"
            )
            print(
                f"batched:   "
                f"top1={batched_top1}, "
                f"logit={batched_top1_logit:.6f}; "
                f"top2={batched_top2}, "
                f"logit={batched_top2_logit:.6f}; "
                f"gap={batched_gap:.6f}"
            )
            print(
                f"reference: "
                f"top1={reference_top1}, "
                f"logit={reference_top1_logit:.6f}; "
                f"top2={reference_top2}, "
                f"logit={reference_top2_logit:.6f}; "
                f"gap={reference_gap:.6f}"
            )
            print("top1 matched:", matched)

            if not matched:
                # 每个 logit 最多可能移动 request_max_diff。
                # 因此两个 logits 的相对差距最多可能变化
                # 约 2 * request_max_diff。
                numerically_ambiguous = (
                    reference_gap
                    <= 2.0 * request_max_diff
                )

                print(
                    "numerically ambiguous:",
                    numerically_ambiguous,
                )

                if not numerically_ambiguous:
                    raise AssertionError(
                        f"Unexpected argmax mismatch at "
                        f"request={request_idx}, "
                        f"generation_step={generation_step}. "
                        f"reference_gap={reference_gap}, "
                        f"max_diff={request_max_diff}"
                    )

                ambiguous_steps.append(
                    {
                        "request": request_idx,
                        "step": generation_step,
                        "batched_token": batched_top1,
                        "reference_token": reference_top1,
                        "reference_gap": reference_gap,
                        "max_diff": request_max_diff,
                    }
                )

        # 关键点：
        # 使用 batched path 的 token 同时推进 batched cache
        # 和所有 single-request cache。
        #
        # 因此下一步两条路径看到的 token 历史完全一致。
        generated_tokens.append(
            batched_next_tokens.clone()
        )

        if generation_step == max_new_tokens - 1:
            break

        start_pos = prompt_len + generation_step

        decode_position_ids = torch.tensor(
            [start_pos],
            device=device,
            dtype=torch.long,
        )

        # -----------------------------------------------------
        # Batched decode
        # -----------------------------------------------------
        batched_logits = model(
            input_ids=batched_next_tokens,
            position_ids=decode_position_ids,
            kv_cache=batched_kv_cache,
            start_pos=start_pos,
            logits_to_keep=1,
        )

        # -----------------------------------------------------
        # Separate single-request decode
        #
        # 注意：这里输入的是 batched path 生成的 token，
        # 而不是 single path 自己的 argmax。
        # -----------------------------------------------------
        single_logits_list = []

        for request_idx in range(batch_size):
            single_token = batched_next_tokens[
                request_idx : request_idx + 1
            ]

            single_logits = model(
                input_ids=single_token,
                position_ids=decode_position_ids,
                kv_cache=single_kv_caches[request_idx],
                start_pos=start_pos,
                logits_to_keep=1,
            )

            single_logits_list.append(single_logits)

        reference_logits = torch.cat(
            single_logits_list,
            dim=0,
        )

    generated_tokens = torch.cat(
        generated_tokens,
        dim=1,
    )

    print("\n===== Generated tokens from batched path =====")
    print(generated_tokens)

    print("\n===== Ambiguous argmax steps =====")

    if not ambiguous_steps:
        print("No ambiguous steps.")
    else:
        for item in ambiguous_steps:
            print(item)

    print(
        "\nBatched generation stepwise test passed."
    )


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

    test_batched_generation_stepwise(
        model=model,
        config=config,
        input_ids=input_ids,
    )


if __name__ == "__main__":
    main()