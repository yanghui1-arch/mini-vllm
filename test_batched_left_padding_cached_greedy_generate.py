import gc
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cache.kv_cache import StaticKVCache
from config import Qwen3Config

from generation.greedy import batched_left_padding_cached_greedy_generate

from model.qwen3 import Qwen3ForCausalLM
from weight_loader import load_safetensors


MODEL_DIR = Path(
    "/mnt/yanghui/models/Qwen/Qwen3-4B"
)

MAX_NEW_TOKENS = 6

# 延续之前 BF16 测试使用的误差范围。
MAX_ABS_TOLERANCE = 1.0
MEAN_ABS_TOLERANCE = 0.15
TIE_TOLERANCE = 0.5


def calculate_position_ids(
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    根据左 padding attention_mask 计算逻辑 position_ids。

    例如：

    attention_mask:
        [0, 0, 1, 1, 1]

    position_ids:
        [0, 0, 0, 1, 2]
    """
    position_ids = (
        attention_mask.long().cumsum(dim=-1) - 1
    )

    position_ids = position_ids.masked_fill(
        attention_mask == 0,
        0,
    )

    return position_ids


def build_left_padded_batch(
    prompt_token_ids: list[list[int]],
    pad_token_id: int,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
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
        [len(token_ids) for token_ids in prompt_token_ids],
        dtype=torch.long,
        device=device,
    )

    padded_seq_len = int(
        prompt_lengths.max().item()
    )

    input_ids = torch.full(
        size=(batch_size, padded_seq_len),
        fill_value=pad_token_id,
        dtype=torch.long,
        device=device,
    )

    attention_mask = torch.zeros(
        batch_size,
        padded_seq_len,
        dtype=torch.long,
        device=device,
    )

    for request_idx, token_ids in enumerate(
        prompt_token_ids
    ):
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

    return (
        input_ids,
        attention_mask,
        prompt_lengths,
    )


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
) -> tuple[
    Qwen3Config,
    Qwen3ForCausalLM,
]:
    config = Qwen3Config.from_json(
        model_dir / "config.json"
    )

    model = Qwen3ForCausalLM(config)

    model = model.to(
        device=device,
        dtype=dtype,
    )

    state_dict = load_safetensors(model_dir)

    missing_keys, unexpected_keys = (
        model.load_state_dict(
            state_dict,
            strict=False,
        )
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
        "unexpected missing keys: "
        f"{real_missing_keys}"
    )

    assert not unexpected_keys, (
        "unexpected state-dict keys: "
        f"{unexpected_keys}"
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


def check_single_row_near_tie(
    lhs_logits: torch.Tensor,
    rhs_logits: torch.Tensor,
    request_idx: int,
    description: str,
    tie_tolerance: float = TIE_TOLERANCE,
) -> None:
    """
    检查指定 request 的两个 greedy token。

    如果 token 不同，则要求两个候选 token 在两边都属于
    near tie。
    """
    lhs_row = (
        lhs_logits[request_idx, -1]
        .float()
        .cpu()
    )

    rhs_row = (
        rhs_logits[request_idx, -1]
        .float()
        .cpu()
    )

    lhs_token = int(
        torch.argmax(lhs_row).item()
    )

    rhs_token = int(
        torch.argmax(rhs_row).item()
    )

    if lhs_token == rhs_token:
        print(
            f"{description}: "
            f"same next token {lhs_token}"
        )
        return

    lhs_preference_gap = float(
        lhs_row[lhs_token]
        - lhs_row[rhs_token]
    )

    rhs_preference_gap = float(
        rhs_row[rhs_token]
        - rhs_row[lhs_token]
    )

    print(
        f"{description}: "
        "different next tokens but checking near tie"
    )

    print("lhs token:", lhs_token)
    print("rhs token:", rhs_token)

    print(
        "lhs preference gap:",
        lhs_preference_gap,
    )

    print(
        "rhs preference gap:",
        rhs_preference_gap,
    )

    assert lhs_preference_gap <= tie_tolerance, (
        f"{description}: lhs strongly prefers "
        f"token {lhs_token}; "
        f"gap={lhs_preference_gap}, "
        f"tolerance={tie_tolerance}"
    )

    assert rhs_preference_gap <= tie_tolerance, (
        f"{description}: rhs strongly prefers "
        f"token {rhs_token}; "
        f"gap={rhs_preference_gap}, "
        f"tolerance={tie_tolerance}"
    )

    print(
        f"{description}: accepted as "
        "BF16 near-tie argmax flip"
    )


def compare_logits(
    lhs_logits: torch.Tensor,
    rhs_logits: torch.Tensor,
    description: str,
    max_abs_tolerance: float = MAX_ABS_TOLERANCE,
    mean_abs_tolerance: float = MEAN_ABS_TOLERANCE,
) -> None:
    """
    比较两个 [B, 1, vocab_size] logits。

    除了误差，还会检查每个 request 的 greedy token。
    """
    lhs_logits = lhs_logits.float().cpu()
    rhs_logits = rhs_logits.float().cpu()

    assert lhs_logits.shape == rhs_logits.shape, (
        f"{description}: shape mismatch: "
        f"{tuple(lhs_logits.shape)} vs "
        f"{tuple(rhs_logits.shape)}"
    )

    difference = torch.abs(
        lhs_logits - rhs_logits
    )

    max_difference = float(
        difference.max().item()
    )

    mean_difference = float(
        difference.mean().item()
    )

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
        "max absolute difference:",
        max_difference,
    )

    print(
        "mean absolute difference:",
        mean_difference,
    )

    print(
        "lhs next tokens:",
        lhs_next_tokens.tolist(),
    )

    print(
        "rhs next tokens:",
        rhs_next_tokens.tolist(),
    )

    assert max_difference <= max_abs_tolerance, (
        f"{description}: max difference too large: "
        f"{max_difference} > "
        f"{max_abs_tolerance}"
    )

    assert mean_difference <= mean_abs_tolerance, (
        f"{description}: mean difference too large: "
        f"{mean_difference} > "
        f"{mean_abs_tolerance}"
    )

    batch_size = lhs_logits.shape[0]

    for request_idx in range(batch_size):
        check_single_row_near_tie(
            lhs_logits=lhs_logits,
            rhs_logits=rhs_logits,
            request_idx=request_idx,
            description=(
                f"{description}, "
                f"request {request_idx}"
            ),
        )


@torch.inference_mode()
def manually_replay_mini_generation(
    model,
    config: Qwen3Config,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    expected_new_token_ids: torch.Tensor,
) -> list[torch.Tensor]:
    """
    手动重放：

        prefill
        +
        多轮 cached decode

    每一步都要求 greedy token 与正式 generate 函数的输出
    完全一致。

    Returns:
        mini_logits_per_step:

            长度为 max_new_tokens 的 list。

            每个元素形状：
                [B, 1, vocab_size]
    """
    batch_size, padded_seq_len = input_ids.shape

    max_new_tokens = (
        expected_new_token_ids.shape[1]
    )

    prompt_lengths = attention_mask.sum(
        dim=-1,
    ).long()

    kv_cache = create_kv_cache(
        config=config,
        batch_size=batch_size,
        max_seq_len=(
            padded_seq_len + max_new_tokens
        ),
        dtype=next(model.parameters()).dtype,
        device=input_ids.device,
    )

    running_attention_mask = attention_mask

    mini_logits_per_step: list[
        torch.Tensor
    ] = []

    # ============================================================
    # Step 0: Prefill 预测第一个新 token
    # ============================================================

    prefill_position_ids = (
        calculate_position_ids(
            running_attention_mask
        )
    )

    logits = model(
        input_ids=input_ids,
        attention_mask=running_attention_mask,
        position_ids=prefill_position_ids,
        kv_cache=kv_cache,
        start_pos=0,
        logits_to_keep=1,
    )

    assert_finite(
        logits,
        "mini manual prefill logits",
    )

    mini_logits_per_step.append(
        logits.float().cpu()
    )

    actual_first_tokens = torch.argmax(
        logits[:, -1, :],
        dim=-1,
        keepdim=True,
    )

    expected_first_tokens = (
        expected_new_token_ids[:, 0:1]
    )

    assert torch.equal(
        actual_first_tokens,
        expected_first_tokens,
    ), (
        "generate function output does not match "
        "manual mini prefill greedy token"
    )

    # Prefill 生成的第一个 token，作为第一次 decode 输入。
    current_tokens = expected_first_tokens

    # ============================================================
    # Step 1 ... N-1: Cached decode
    # ============================================================

    for decode_step in range(
        max_new_tokens - 1
    ):
        new_valid_token_mask = torch.ones(
            batch_size,
            1,
            dtype=running_attention_mask.dtype,
            device=running_attention_mask.device,
        )

        running_attention_mask = torch.cat(
            [
                running_attention_mask,
                new_valid_token_mask,
            ],
            dim=-1,
        )

        decode_position_ids = (
            prompt_lengths[:, None]
            + decode_step
        )

        decode_start_pos = (
            padded_seq_len + decode_step
        )

        logits = model(
            input_ids=current_tokens,
            attention_mask=running_attention_mask,
            position_ids=decode_position_ids,
            kv_cache=kv_cache,
            start_pos=decode_start_pos,
            logits_to_keep=1,
        )

        assert_finite(
            logits,
            (
                "mini manual decode logits "
                f"at generation step "
                f"{decode_step + 1}"
            ),
        )

        mini_logits_per_step.append(
            logits.float().cpu()
        )

        actual_next_tokens = torch.argmax(
            logits[:, -1, :],
            dim=-1,
            keepdim=True,
        )

        expected_next_tokens = (
            expected_new_token_ids[
                :,
                decode_step + 1:
                decode_step + 2,
            ]
        )

        assert torch.equal(
            actual_next_tokens,
            expected_next_tokens,
        ), (
            "generate function output does not match "
            "manual mini cached decode at generation "
            f"step {decode_step + 1}"
        )

        current_tokens = expected_next_tokens

    del kv_cache

    return mini_logits_per_step


@torch.inference_mode()
def collect_hf_logits_on_forced_path(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    forced_new_token_ids: torch.Tensor,
) -> list[torch.Tensor]:
    """
    使用 Hugging Face 完整重算，并强制沿着 mini-vLLM
    生成出的 token 路径运行。

    这样即使某一步发生 BF16 argmax 翻转，后续仍然可以在
    完全相同的历史 token 上比较 logits。
    """
    max_new_tokens = (
        forced_new_token_ids.shape[1]
    )

    running_input_ids = input_ids
    running_attention_mask = attention_mask

    hf_logits_per_step: list[
        torch.Tensor
    ] = []

    for generation_step in range(
        max_new_tokens
    ):
        position_ids = calculate_position_ids(
            running_attention_mask
        )

        outputs = model(
            input_ids=running_input_ids,
            attention_mask=running_attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )

        logits = outputs.logits[:, -1:, :]

        assert_finite(
            logits,
            (
                "HF forced-path logits at "
                f"generation step {generation_step}"
            ),
        )

        hf_logits_per_step.append(
            logits.float().cpu()
        )

        # 最后一步之后不再需要追加 token。
        if generation_step == (
            max_new_tokens - 1
        ):
            continue

        forced_token = forced_new_token_ids[
            :,
            generation_step:
            generation_step + 1,
        ]

        running_input_ids = torch.cat(
            [
                running_input_ids,
                forced_token,
            ],
            dim=-1,
        )

        running_attention_mask = torch.cat(
            [
                running_attention_mask,
                torch.ones(
                    running_attention_mask.shape[0],
                    1,
                    dtype=(
                        running_attention_mask.dtype
                    ),
                    device=(
                        running_attention_mask.device
                    ),
                ),
            ],
            dim=-1,
        )

    return hf_logits_per_step


@torch.inference_mode()
def hf_full_recompute_greedy_generate(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
) -> torch.Tensor:
    """
    Hugging Face 参考实现。

    每一轮都对完整序列重新 forward，不使用 HF KV Cache。

    当前同样：
        1. greedy decoding
        2. 固定生成 max_new_tokens
        3. 不处理 EOS
    """
    running_input_ids = input_ids
    running_attention_mask = attention_mask

    for _ in range(max_new_tokens):
        position_ids = calculate_position_ids(
            running_attention_mask
        )

        outputs = model(
            input_ids=running_input_ids,
            attention_mask=running_attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )

        logits = outputs.logits[:, -1:, :]

        next_tokens = torch.argmax(
            logits[:, -1, :],
            dim=-1,
            keepdim=True,
        )

        running_input_ids = torch.cat(
            [
                running_input_ids,
                next_tokens,
            ],
            dim=-1,
        )

        running_attention_mask = torch.cat(
            [
                running_attention_mask,
                torch.ones(
                    running_attention_mask.shape[0],
                    1,
                    dtype=(
                        running_attention_mask.dtype
                    ),
                    device=(
                        running_attention_mask.device
                    ),
                ),
            ],
            dim=-1,
        )

    return running_input_ids


def compare_generated_sequences(
    mini_new_token_ids: torch.Tensor,
    hf_new_token_ids: torch.Tensor,
    mini_logits_per_step: list[torch.Tensor],
    hf_forced_logits_per_step: list[torch.Tensor],
) -> None:
    """
    比较 mini 与 HF 各自的 greedy generation。

    如果某个请求发生分叉：
        只检查第一次分叉。

    因为第一次分叉之后，两边的历史 token 已经不同，
    后续 token 不再适合直接比较。
    """
    mini_new_token_ids = (
        mini_new_token_ids.cpu()
    )

    hf_new_token_ids = (
        hf_new_token_ids.cpu()
    )

    batch_size, max_new_tokens = (
        mini_new_token_ids.shape
    )

    assert hf_new_token_ids.shape == (
        batch_size,
        max_new_tokens,
    )

    print(
        "\n===== mini-vLLM vs HF generated tokens ====="
    )

    print(
        "mini new token ids:"
    )
    print(mini_new_token_ids)

    print(
        "\nHF new token ids:"
    )
    print(hf_new_token_ids)

    for request_idx in range(batch_size):
        mini_row = mini_new_token_ids[
            request_idx
        ]

        hf_row = hf_new_token_ids[
            request_idx
        ]

        mismatch_indices = torch.nonzero(
            mini_row != hf_row,
            as_tuple=False,
        ).flatten()

        if mismatch_indices.numel() == 0:
            print(
                f"\nrequest {request_idx}: "
                "all generated tokens match exactly"
            )
            continue

        first_mismatch_step = int(
            mismatch_indices[0].item()
        )

        mini_token = int(
            mini_row[first_mismatch_step].item()
        )

        hf_token = int(
            hf_row[first_mismatch_step].item()
        )

        print(
            f"\nrequest {request_idx}: "
            "generation diverged"
        )

        print(
            "first mismatch generation step:",
            first_mismatch_step,
        )

        print(
            "mini token:",
            mini_token,
        )

        print(
            "HF token:",
            hf_token,
        )

        # 第一次分叉之前，两边的生成历史完全相同。
        # 因此可以用 forced-path logits 判断这次分叉是否来自
        # BF16 near tie。
        check_single_row_near_tie(
            lhs_logits=(
                mini_logits_per_step[
                    first_mismatch_step
                ]
            ),
            rhs_logits=(
                hf_forced_logits_per_step[
                    first_mismatch_step
                ]
            ),
            request_idx=request_idx,
            description=(
                f"request {request_idx}, first "
                f"generation divergence at step "
                f"{first_mismatch_step}"
            ),
        )

        print(
            "Later generated tokens are not compared "
            "strictly because the histories have diverged."
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
        "===== Batched left-padding cached greedy "
        "generation test ====="
    )

    print("model directory:", MODEL_DIR)
    print("device:", device)
    print("dtype:", dtype)
    print("max new tokens:", MAX_NEW_TOKENS)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
    )

    pad_token_id = tokenizer.pad_token_id

    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    assert pad_token_id is not None

    seed_text = (
        "Hello, I am learning how large language model "
        "inference engines use key value caches, "
        "attention masks, rotary position embeddings, "
        "static batching, and cached decoding."
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

    (
        input_ids,
        attention_mask,
        prompt_lengths,
    ) = build_left_padded_batch(
        prompt_token_ids=prompt_token_ids,
        pad_token_id=pad_token_id,
        device=device,
    )

    batch_size, padded_seq_len = (
        input_ids.shape
    )

    print("\n===== Test inputs =====")

    print("\ninput_ids shape:")
    print(input_ids.shape)

    print("\ninput_ids:")
    print(input_ids)

    print("\nattention_mask:")
    print(attention_mask)

    print("\nprompt lengths:")
    print(prompt_lengths.tolist())

    assert batch_size == 2
    assert padded_seq_len == 20
    assert prompt_lengths.tolist() == [3, 20]

    # 保存原始输入，检查 generate 函数有没有意外修改它们。
    original_input_ids = input_ids.clone()

    original_attention_mask = (
        attention_mask.clone()
    )

    # ============================================================
    # mini-vLLM generation
    # ============================================================

    print("\n===== Load mini-vLLM model =====")

    config, mini_model = load_mini_model(
        model_dir=MODEL_DIR,
        device=device,
        dtype=dtype,
    )

    print(
        "\n===== Run batched left-padding cached "
        "greedy generation ====="
    )

    mini_output_ids = (
        batched_left_padding_cached_greedy_generate(
            model=mini_model,
            config=config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
        )
    )

    print("\nmini output shape:")
    print(mini_output_ids.shape)

    print("\nmini output ids:")
    print(mini_output_ids)

    assert mini_output_ids.shape == (
        batch_size,
        padded_seq_len + MAX_NEW_TOKENS,
    )

    # Generate 返回：
    #
    # 左 padding 原始输入 + 新生成 token。
    assert torch.equal(
        mini_output_ids[:, :padded_seq_len],
        input_ids,
    ), (
        "the prefix of mini output does not match "
        "the original left-padded input_ids"
    )

    # 函数不能原地修改调用者传入的 Tensor。
    assert torch.equal(
        input_ids,
        original_input_ids,
    ), "generate modified input_ids in place"

    assert torch.equal(
        attention_mask,
        original_attention_mask,
    ), "generate modified attention_mask in place"

    mini_new_token_ids = mini_output_ids[
        :,
        padded_seq_len:,
    ]

    assert mini_new_token_ids.shape == (
        batch_size,
        MAX_NEW_TOKENS,
    )

    print("\nmini newly generated token ids:")
    print(mini_new_token_ids)

    # 手动重放同样的 mini prefill + cached decode。
    #
    # 这里要求与正式 generate 函数完全一致，
    # 不是近似一致。
    print(
        "\n===== Replay mini generation manually ====="
    )

    mini_logits_per_step = (
        manually_replay_mini_generation(
            model=mini_model,
            config=config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            expected_new_token_ids=(
                mini_new_token_ids
            ),
        )
    )

    assert len(mini_logits_per_step) == (
        MAX_NEW_TOKENS
    )

    print(
        "The generate function exactly matches "
        "manual mini prefill + cached decode."
    )

    # 将后续需要的数据放到 CPU，再释放 mini 模型。
    mini_output_ids_cpu = (
        mini_output_ids.cpu()
    )

    mini_new_token_ids_cpu = (
        mini_new_token_ids.cpu()
    )

    del mini_output_ids
    del mini_new_token_ids
    del mini_model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ============================================================
    # Hugging Face reference
    # ============================================================

    print(
        "\n===== Load Hugging Face reference ====="
    )

    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype=dtype,
        attn_implementation="eager",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device)

    hf_model.eval()

    # ------------------------------------------------------------
    # HF 在 mini token 路径上的逐步 logits
    # ------------------------------------------------------------

    print(
        "\n===== Compare logits on the same token path ====="
    )

    mini_new_token_ids_device = (
        mini_new_token_ids_cpu.to(device)
    )

    hf_forced_logits_per_step = (
        collect_hf_logits_on_forced_path(
            model=hf_model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            forced_new_token_ids=(
                mini_new_token_ids_device
            ),
        )
    )

    assert len(hf_forced_logits_per_step) == (
        MAX_NEW_TOKENS
    )

    for generation_step in range(
        MAX_NEW_TOKENS
    ):
        compare_logits(
            lhs_logits=(
                mini_logits_per_step[
                    generation_step
                ]
            ),
            rhs_logits=(
                hf_forced_logits_per_step[
                    generation_step
                ]
            ),
            description=(
                "mini cached logits vs HF full "
                "recomputation logits, "
                f"generation step {generation_step}"
            ),
        )

    # ------------------------------------------------------------
    # HF 自己执行 greedy generation
    # ------------------------------------------------------------

    print(
        "\n===== Run Hugging Face greedy generation ====="
    )

    hf_output_ids = (
        hf_full_recompute_greedy_generate(
            model=hf_model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
        )
    )

    assert hf_output_ids.shape == (
        batch_size,
        padded_seq_len + MAX_NEW_TOKENS,
    )

    assert torch.equal(
        hf_output_ids[:, :padded_seq_len],
        input_ids,
    )

    hf_new_token_ids = hf_output_ids[
        :,
        padded_seq_len:,
    ]

    # 比较两边各自真实生成出的 token。
    #
    # 如果发生分叉，只严格检查第一次分叉是否为 near tie。
    compare_generated_sequences(
        mini_new_token_ids=(
            mini_new_token_ids_cpu
        ),
        hf_new_token_ids=(
            hf_new_token_ids
        ),
        mini_logits_per_step=(
            mini_logits_per_step
        ),
        hf_forced_logits_per_step=(
            hf_forced_logits_per_step
        ),
    )

    # ============================================================
    # Decode generated text
    # ============================================================

    print("\n===== Generated text =====")

    hf_new_token_ids_cpu = (
        hf_new_token_ids.cpu()
    )

    for request_idx in range(batch_size):
        mini_text = tokenizer.decode(
            mini_new_token_ids_cpu[
                request_idx
            ],
            skip_special_tokens=True,
        )

        hf_text = tokenizer.decode(
            hf_new_token_ids_cpu[
                request_idx
            ],
            skip_special_tokens=True,
        )

        print(f"\nrequest {request_idx}")

        print("mini generated text:")
        print(repr(mini_text))

        print("HF generated text:")
        print(repr(hf_text))

    del hf_model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n===== All tests passed =====")

    print("Verified:")

    print(
        "1. generate returns left-padded input_ids "
        "followed by max_new_tokens"
    )

    print(
        "2. generate does not modify input_ids or "
        "attention_mask in place"
    )

    print(
        "3. generate exactly matches a manual mini "
        "prefill + multi-step cached decode"
    )

    print(
        "4. attention_mask, logical position_ids, and "
        "physical cache positions stay correct "
        "through multiple decode steps"
    )

    print(
        "5. mini cached logits match Hugging Face "
        "full recomputation on the same token history"
    )

    print(
        "6. mini and HF greedy generations either "
        "match or first diverge only at a BF16 near tie"
    )


if __name__ == "__main__":
    main()