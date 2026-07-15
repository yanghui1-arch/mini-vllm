import gc
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cache.kv_cache import StaticKVCache
from config import Qwen3Config
from model.qwen3 import Qwen3ForCausalLM
from weight_loader import load_safetensors


MODEL_DIR = Path("/mnt/yanghui/models/Qwen/Qwen3-4B")

# BF16 下 batch size 不同可能导致微小数值分叉。
MAX_ABS_TOLERANCE = 1.0
MEAN_ABS_TOLERANCE = 0.15
TIE_TOLERANCE = 0.5


def calculate_position_ids(
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    根据 attention_mask 计算逻辑 position_ids。

    例如：

    attention_mask:
    [0, 0, 1, 1, 1]

    position_ids:
    [0, 0, 0, 1, 2]
    """
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids = position_ids.masked_fill(
        attention_mask == 0,
        0,
    )
    return position_ids


def build_left_padded_batch(
    prompt_token_ids: list[list[int]],
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    将不同长度 prompt 构造成左 padding batch。

    Returns:
        input_ids:
            [B, padded_seq_len]

        attention_mask:
            [B, padded_seq_len]

        prompt_lengths:
            [B]
    """
    batch_size = len(prompt_token_ids)
    prompt_lengths = torch.tensor(
        [len(ids) for ids in prompt_token_ids],
        dtype=torch.long,
        device=device,
    )

    padded_seq_len = int(prompt_lengths.max().item())

    input_ids = torch.full(
        (batch_size, padded_seq_len),
        fill_value=pad_token_id,
        dtype=torch.long,
        device=device,
    )

    attention_mask = torch.zeros(
        (batch_size, padded_seq_len),
        dtype=torch.long,
        device=device,
    )

    for request_idx, token_ids in enumerate(prompt_token_ids):
        prompt_len = len(token_ids)

        input_ids[
            request_idx,
            padded_seq_len - prompt_len:
        ] = torch.tensor(
            token_ids,
            dtype=torch.long,
            device=device,
        )

        attention_mask[
            request_idx,
            padded_seq_len - prompt_len:
        ] = 1

    return input_ids, attention_mask, prompt_lengths


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


def load_mini_model(
    model_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Qwen3Config, Qwen3ForCausalLM]:
    config = Qwen3Config.from_json(
        model_dir / "config.json"
    )

    model = Qwen3ForCausalLM(config)
    model = model.to(
        device=device,
        dtype=dtype,
    )

    state_dict = load_safetensors(model_dir)

    missing_keys, unexpected_keys = model.load_state_dict(
        state_dict,
        strict=False,
    )

    real_missing_keys = [
        key
        for key in missing_keys
        if key != "lm_head.weight"
    ]

    print("missing keys:")
    print(missing_keys)

    print("\nunexpected keys:")
    print(unexpected_keys)

    assert not real_missing_keys, (
        f"unexpected missing keys: {real_missing_keys}"
    )
    assert not unexpected_keys, (
        f"unexpected state-dict keys: {unexpected_keys}"
    )

    if config.tie_word_embeddings:
        if hasattr(model, "tie_weights"):
            model.tie_weights()
        else:
            model.lm_head.weight = (
                model.model.embed_tokens.weight
            )

    model.eval()

    del state_dict
    gc.collect()

    return config, model


def assert_finite(
    tensor: torch.Tensor,
    name: str,
) -> None:
    assert not torch.isnan(tensor).any(), (
        f"{name} contains NaN"
    )
    assert not torch.isinf(tensor).any(), (
        f"{name} contains Inf"
    )


def assert_next_tokens_compatible(
    lhs_logits: torch.Tensor,
    rhs_logits: torch.Tensor,
    description: str,
    tie_tolerance: float = TIE_TOLERANCE,
) -> None:
    """
    检查 greedy token 是否一致。

    如果 token 不一致，则检查两个候选 token 在两边是否属于 near tie。
    """
    lhs_logits = lhs_logits.float().cpu()
    rhs_logits = rhs_logits.float().cpu()

    assert lhs_logits.ndim == 3
    assert rhs_logits.ndim == 3
    assert lhs_logits.shape == rhs_logits.shape

    batch_size = lhs_logits.shape[0]

    for request_idx in range(batch_size):
        lhs_row = lhs_logits[request_idx, -1]
        rhs_row = rhs_logits[request_idx, -1]

        lhs_token = int(torch.argmax(lhs_row).item())
        rhs_token = int(torch.argmax(rhs_row).item())

        if lhs_token == rhs_token:
            print(
                f"{description}, item {request_idx}: "
                f"same next token {lhs_token}"
            )
            continue

        print(
            f"{description}, item {request_idx}: "
            "different next tokens but checking near tie"
        )

        lhs_preference_gap = float(
            lhs_row[lhs_token] - lhs_row[rhs_token]
        )
        rhs_preference_gap = float(
            rhs_row[rhs_token] - rhs_row[lhs_token]
        )

        print(f"lhs token: {lhs_token}")
        print(f"rhs token: {rhs_token}")
        print(
            "lhs preference gap:",
            lhs_preference_gap,
        )
        print(
            "rhs preference gap:",
            rhs_preference_gap,
        )

        assert lhs_preference_gap <= tie_tolerance, (
            f"{description}, item {request_idx}: "
            f"lhs strongly prefers token {lhs_token}; "
            f"gap={lhs_preference_gap}, "
            f"tolerance={tie_tolerance}"
        )

        assert rhs_preference_gap <= tie_tolerance, (
            f"{description}, item {request_idx}: "
            f"rhs strongly prefers token {rhs_token}; "
            f"gap={rhs_preference_gap}, "
            f"tolerance={tie_tolerance}"
        )

        print(
            f"{description}, item {request_idx}: "
            "accepted as BF16 near-tie argmax flip"
        )


def compare_logits(
    lhs_logits: torch.Tensor,
    rhs_logits: torch.Tensor,
    description: str,
    max_abs_tolerance: float = MAX_ABS_TOLERANCE,
    mean_abs_tolerance: float = MEAN_ABS_TOLERANCE,
) -> None:
    lhs_logits = lhs_logits.float().cpu()
    rhs_logits = rhs_logits.float().cpu()

    assert lhs_logits.shape == rhs_logits.shape, (
        f"{description}: shape mismatch: "
        f"{lhs_logits.shape} vs {rhs_logits.shape}"
    )

    difference = torch.abs(lhs_logits - rhs_logits)

    max_difference = float(difference.max().item())
    mean_difference = float(difference.mean().item())

    lhs_next_tokens = torch.argmax(
        lhs_logits[:, -1, :],
        dim=-1,
    )
    rhs_next_tokens = torch.argmax(
        rhs_logits[:, -1, :],
        dim=-1,
    )

    print(f"\n{description}")
    print(
        "max absolute difference: ",
        max_difference,
    )
    print(
        "mean absolute difference:",
        mean_difference,
    )
    print(
        "lhs next token:",
        lhs_next_tokens.tolist(),
    )
    print(
        "rhs next token:",
        rhs_next_tokens.tolist(),
    )

    assert max_difference <= max_abs_tolerance, (
        f"{description}: max difference too large: "
        f"{max_difference} > {max_abs_tolerance}"
    )

    assert mean_difference <= mean_abs_tolerance, (
        f"{description}: mean difference too large: "
        f"{mean_difference} > {mean_abs_tolerance}"
    )

    assert_next_tokens_compatible(
        lhs_logits,
        rhs_logits,
        description,
    )


def main() -> None:
    torch.manual_seed(0)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.bfloat16
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    print(
        "===== Left-padding batched cached decode test ====="
    )
    print("model directory:", MODEL_DIR)
    print("device:", device)
    print("dtype:", dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
    )

    pad_token_id = tokenizer.pad_token_id

    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    assert pad_token_id is not None

    # 先 tokenize 一段足够长的文本，然后截取为长度 3 和 20。
    # 这样可以稳定构造不同长度的合法 token 序列。
    seed_text = (
        "Hello, I am learning how large language model inference "
        "engines use key value caches, attention masks, rotary "
        "position embeddings, static batching, and decoding."
    )

    seed_token_ids = tokenizer.encode(
        seed_text,
        add_special_tokens=False,
    )

    assert len(seed_token_ids) >= 20, (
        "seed text did not produce at least 20 tokens"
    )

    prompt_token_ids = [
        seed_token_ids[:3],
        seed_token_ids[:20],
    ]

    input_ids, prefill_attention_mask, prompt_lengths = (
        build_left_padded_batch(
            prompt_token_ids=prompt_token_ids,
            pad_token_id=pad_token_id,
            device=device,
        )
    )

    prefill_position_ids = calculate_position_ids(
        prefill_attention_mask
    )

    batch_size, padded_seq_len = input_ids.shape

    print("\n===== Batched prefill inputs =====")

    print("\ninput_ids shape:")
    print(input_ids.shape)

    print("\ninput_ids:")
    print(input_ids)

    print("\nprefill attention_mask:")
    print(prefill_attention_mask)

    print("\nprefill position_ids:")
    print(prefill_position_ids)

    print("\nprompt lengths:")
    print(prompt_lengths.tolist())

    assert batch_size == 2
    assert padded_seq_len == 20
    assert prompt_lengths.tolist() == [3, 20]

    assert prefill_attention_mask.shape == (
        batch_size,
        padded_seq_len,
    )

    assert prefill_position_ids.shape == (
        batch_size,
        padded_seq_len,
    )

    assert prefill_position_ids[0, -3:].tolist() == [
        0,
        1,
        2,
    ]

    assert prefill_position_ids[1].tolist() == list(
        range(20)
    )

    # ============================================================
    # Hugging Face reference
    #
    # HF 不使用我们自定义的 StaticKVCache。
    # 它通过“完整序列重新 forward”产生 decode 参考 logits。
    # ============================================================

    print("\n===== Load Hugging Face reference =====")

    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype=dtype,
        attn_implementation="eager",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device)

    hf_model.eval()

    with torch.inference_mode():
        # --------------------------------------------------------
        # HF batched prefill
        # --------------------------------------------------------
        hf_prefill_outputs = hf_model(
            input_ids=input_ids,
            attention_mask=prefill_attention_mask,
            position_ids=prefill_position_ids,
            use_cache=False,
        )

        hf_prefill_logits = (
            hf_prefill_outputs.logits[:, -1:, :]
            .float()
            .cpu()
        )

        assert_finite(
            hf_prefill_logits,
            "HF prefill logits",
        )

        # 使用 HF batched prefill 产生统一的 decode 输入 token。
        #
        # 后续 mini batch 和 mini single 都输入完全相同的 token，
        # 从而只测试 cache、mask、position 的正确性。
        decode_input_ids_cpu = torch.argmax(
            hf_prefill_logits[:, -1, :],
            dim=-1,
            keepdim=True,
        )

        decode_input_ids = decode_input_ids_cpu.to(
            device=device,
            dtype=torch.long,
        )

        print("\n===== Decode input tokens =====")

        for request_idx in range(batch_size):
            token_id = int(
                decode_input_ids_cpu[request_idx, 0].item()
            )

            print(
                f"request {request_idx}: "
                f"token_id={token_id}, "
                f"text={tokenizer.decode([token_id])!r}"
            )

        # --------------------------------------------------------
        # 更新 attention mask
        #
        # 新加入的 decode token 是有效 token，因此追加一列 1。
        # --------------------------------------------------------
        decode_attention_mask = torch.cat(
            [
                prefill_attention_mask,
                torch.ones(
                    batch_size,
                    1,
                    dtype=prefill_attention_mask.dtype,
                    device=device,
                ),
            ],
            dim=-1,
        )

        # 当前 decode token 的逻辑位置。
        #
        # request 0:
        # prompt length = 3
        # decode position = 3
        #
        # request 1:
        # prompt length = 20
        # decode position = 20
        decode_position_ids = prompt_lengths[:, None]

        # 用完整 attention mask 重新计算 position_ids，
        # 它最后一列应当与 decode_position_ids 完全一致。
        full_position_ids = calculate_position_ids(
            decode_attention_mask
        )

        assert torch.equal(
            full_position_ids[:, -1:],
            decode_position_ids,
        )

        assert decode_attention_mask.shape == (
            batch_size,
            padded_seq_len + 1,
        )

        assert decode_position_ids.shape == (
            batch_size,
            1,
        )

        assert decode_position_ids.tolist() == [
            [3],
            [20],
        ]

        assert decode_attention_mask.sum(dim=-1).tolist() == [
            4,
            21,
        ]

        print("\n===== Batched decode metadata =====")

        print("\ndecode input_ids shape:")
        print(decode_input_ids.shape)

        print("\ndecode attention_mask:")
        print(decode_attention_mask)

        print("\ndecode position_ids:")
        print(decode_position_ids)

        print("\nphysical decode start_pos:")
        print(padded_seq_len)

        # --------------------------------------------------------
        # HF batched full recomputation reference
        # --------------------------------------------------------
        hf_full_batch_input_ids = torch.cat(
            [
                input_ids,
                decode_input_ids,
            ],
            dim=-1,
        )

        hf_batch_decode_outputs = hf_model(
            input_ids=hf_full_batch_input_ids,
            attention_mask=decode_attention_mask,
            position_ids=full_position_ids,
            use_cache=False,
        )

        hf_batch_decode_logits = (
            hf_batch_decode_outputs.logits[:, -1:, :]
            .float()
            .cpu()
        )

        assert_finite(
            hf_batch_decode_logits,
            "HF batched decode logits",
        )

        # --------------------------------------------------------
        # HF single-request full recomputation references
        # --------------------------------------------------------
        hf_single_decode_logits: list[torch.Tensor] = []

        for request_idx, prompt_ids in enumerate(
            prompt_token_ids
        ):
            prompt_len = len(prompt_ids)

            single_prompt_ids = torch.tensor(
                prompt_ids,
                dtype=torch.long,
                device=device,
            ).unsqueeze(0)

            single_decode_token = decode_input_ids[
                request_idx:request_idx + 1
            ]

            single_full_input_ids = torch.cat(
                [
                    single_prompt_ids,
                    single_decode_token,
                ],
                dim=-1,
            )

            single_attention_mask = torch.ones(
                1,
                prompt_len + 1,
                dtype=torch.long,
                device=device,
            )

            single_position_ids = torch.arange(
                prompt_len + 1,
                dtype=torch.long,
                device=device,
            ).unsqueeze(0)

            single_outputs = hf_model(
                input_ids=single_full_input_ids,
                attention_mask=single_attention_mask,
                position_ids=single_position_ids,
                use_cache=False,
            )

            single_logits = (
                single_outputs.logits[:, -1:, :]
                .float()
                .cpu()
            )

            assert_finite(
                single_logits,
                f"HF single request {request_idx} logits",
            )

            hf_single_decode_logits.append(single_logits)

    print("\n===== Hugging Face: batch vs single decode =====")

    for request_idx in range(batch_size):
        compare_logits(
            lhs_logits=hf_batch_decode_logits[
                request_idx:request_idx + 1
            ],
            rhs_logits=hf_single_decode_logits[
                request_idx
            ],
            description=(
                f"HF request {request_idx} "
                "batch vs single decode"
            ),
        )

    del hf_prefill_outputs
    del hf_batch_decode_outputs
    del hf_model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ============================================================
    # mini-vLLM cached execution
    # ============================================================

    print("\n===== Load mini-vLLM model =====")

    config, mini_model = load_mini_model(
        model_dir=MODEL_DIR,
        device=device,
        dtype=dtype,
    )

    max_seq_len = padded_seq_len + 8

    # ------------------------------------------------------------
    # Batched cached prefill
    # ------------------------------------------------------------
    batch_kv_cache = create_kv_cache(
        config=config,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        dtype=dtype,
        device=device,
    )

    with torch.inference_mode():
        mini_prefill_logits_gpu = mini_model(
            input_ids=input_ids,
            attention_mask=prefill_attention_mask,
            position_ids=prefill_position_ids,
            kv_cache=batch_kv_cache,
            start_pos=0,
            logits_to_keep=1,
        )

        assert mini_prefill_logits_gpu.shape == (
            batch_size,
            1,
            config.vocab_size,
        )

        assert_finite(
            mini_prefill_logits_gpu,
            "mini batched prefill logits",
        )

        mini_prefill_logits = (
            mini_prefill_logits_gpu.float().cpu()
        )

        # --------------------------------------------------------
        # 一次 batched cached decode
        #
        # 物理写入位置：
        #     两个请求都是 padded_seq_len = 20
        #
        # 逻辑 RoPE 位置：
        #     request 0 = 3
        #     request 1 = 20
        # --------------------------------------------------------
        mini_batch_decode_logits_gpu = mini_model(
            input_ids=decode_input_ids,
            attention_mask=decode_attention_mask,
            position_ids=decode_position_ids,
            kv_cache=batch_kv_cache,
            start_pos=padded_seq_len,
            logits_to_keep=1,
        )

        assert mini_batch_decode_logits_gpu.shape == (
            batch_size,
            1,
            config.vocab_size,
        )

        assert_finite(
            mini_batch_decode_logits_gpu,
            "mini batched cached decode logits",
        )

        mini_batch_decode_logits = (
            mini_batch_decode_logits_gpu.float().cpu()
        )

    print("\n===== Prefill sanity check =====")

    compare_logits(
        lhs_logits=mini_prefill_logits,
        rhs_logits=hf_prefill_logits,
        description="mini batched prefill vs Hugging Face",
    )

    # batch cache 已经完成测试，后面给每个请求创建独立 cache。
    del batch_kv_cache
    del mini_prefill_logits_gpu
    del mini_batch_decode_logits_gpu

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------------------------------------
    # Single-request cached decode
    # ------------------------------------------------------------
    mini_single_decode_logits: list[torch.Tensor] = []

    print("\n===== mini-vLLM single-request cached decode =====")

    for request_idx, prompt_ids in enumerate(
        prompt_token_ids
    ):
        prompt_len = len(prompt_ids)

        single_input_ids = torch.tensor(
            prompt_ids,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)

        single_prefill_attention_mask = torch.ones(
            1,
            prompt_len,
            dtype=torch.long,
            device=device,
        )

        single_prefill_position_ids = torch.arange(
            prompt_len,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)

        single_kv_cache = create_kv_cache(
            config=config,
            batch_size=1,
            max_seq_len=max_seq_len,
            dtype=dtype,
            device=device,
        )

        with torch.inference_mode():
            # Single prefill:
            #
            # request 0 写入 [0, 3)
            # request 1 写入 [0, 20)
            _ = mini_model(
                input_ids=single_input_ids,
                attention_mask=single_prefill_attention_mask,
                position_ids=single_prefill_position_ids,
                kv_cache=single_kv_cache,
                start_pos=0,
                logits_to_keep=1,
            )

            single_decode_attention_mask = torch.cat(
                [
                    single_prefill_attention_mask,
                    torch.ones(
                        1,
                        1,
                        dtype=torch.long,
                        device=device,
                    ),
                ],
                dim=-1,
            )

            single_decode_position_ids = torch.tensor(
                [[prompt_len]],
                dtype=torch.long,
                device=device,
            )

            single_decode_input_ids = decode_input_ids[
                request_idx:request_idx + 1
            ]

            # Single decode:
            #
            # request 0 物理 start_pos = 3
            # request 1 物理 start_pos = 20
            single_decode_logits_gpu = mini_model(
                input_ids=single_decode_input_ids,
                attention_mask=single_decode_attention_mask,
                position_ids=single_decode_position_ids,
                kv_cache=single_kv_cache,
                start_pos=prompt_len,
                logits_to_keep=1,
            )

            assert single_decode_logits_gpu.shape == (
                1,
                1,
                config.vocab_size,
            )

            assert_finite(
                single_decode_logits_gpu,
                (
                    "mini single request "
                    f"{request_idx} decode logits"
                ),
            )

            mini_single_decode_logits.append(
                single_decode_logits_gpu.float().cpu()
            )

        del single_kv_cache
        del single_decode_logits_gpu

    # ============================================================
    # Comparisons
    # ============================================================

    print("\n===== mini-vLLM: batch vs single cached decode =====")

    for request_idx in range(batch_size):
        compare_logits(
            lhs_logits=mini_batch_decode_logits[
                request_idx:request_idx + 1
            ],
            rhs_logits=mini_single_decode_logits[
                request_idx
            ],
            description=(
                f"mini request {request_idx} "
                "batch vs single cached decode"
            ),
        )

    print(
        "\n===== mini-vLLM cached decode "
        "vs Hugging Face full recomputation ====="
    )

    compare_logits(
        lhs_logits=mini_batch_decode_logits,
        rhs_logits=hf_batch_decode_logits,
        description=(
            "mini batched cached decode "
            "vs HF batched full recomputation"
        ),
    )

    for request_idx in range(batch_size):
        compare_logits(
            lhs_logits=mini_single_decode_logits[
                request_idx
            ],
            rhs_logits=hf_single_decode_logits[
                request_idx
            ],
            description=(
                f"request {request_idx} "
                "mini single cached decode "
                "vs HF single full recomputation"
            ),
        )

    mini_second_tokens = torch.argmax(
        mini_batch_decode_logits[:, -1, :],
        dim=-1,
    )

    hf_second_tokens = torch.argmax(
        hf_batch_decode_logits[:, -1, :],
        dim=-1,
    )

    print("\n===== Generated tokens =====")

    print(
        "first generated tokens / decode inputs:",
        decode_input_ids_cpu.squeeze(-1).tolist(),
    )

    print(
        "mini second generated tokens:",
        mini_second_tokens.tolist(),
    )

    print(
        "HF second generated tokens:",
        hf_second_tokens.tolist(),
    )

    print("\n===== All tests passed =====")
    print("Verified:")
    print(
        "1. different-length prompts can perform "
        "left-padded batched cached decode"
    )
    print(
        "2. attention_mask is extended by one valid-token column"
    )
    print(
        "3. batch physical decode start_pos is padded_seq_len"
    )
    print(
        "4. per-request logical decode position_ids "
        "equal prompt lengths"
    )
    print(
        "5. batched cached decode matches "
        "single-request cached decode"
    )
    print(
        "6. mini cached decode matches "
        "Hugging Face full recomputation"
    )
    print(
        "7. decode logits contain no NaN or Inf"
    )


if __name__ == "__main__":
    main()