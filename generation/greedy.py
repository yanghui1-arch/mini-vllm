import gc
import statistics
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from config import Qwen3Config
from generation.greedy import (
    batched_left_padding_cached_greedy_generate,
    batched_left_padding_cached_greedy_generate_with_eos,
)
from model.qwen3 import Qwen3ForCausalLM
from weight_loader import load_safetensors


MODEL_DIR = Path(
    "/mnt/yanghui/models/Qwen/Qwen3-4B"
)

BENCHMARK_BATCH_SIZE = 2
BENCHMARK_MAX_NEW_TOKENS = 16
BENCHMARK_REPEATS = 3


class ScriptedGreedyModel(nn.Module):
    """
    每次 forward 返回预先指定的 greedy token。

    scripted_tokens[call_index][request_index]
    表示第 call_index 次模型调用时，每个请求的 argmax token。
    """

    def __init__(
        self,
        scripted_tokens: list[list[int]],
        vocab_size: int = 16,
    ) -> None:
        super().__init__()

        # generate 内部会通过 next(model.parameters()).dtype
        # 获取 cache dtype，因此需要至少有一个参数。
        self.dummy_parameter = nn.Parameter(
            torch.zeros(1)
        )

        self.scripted_tokens = scripted_tokens
        self.vocab_size = vocab_size

        self.calls: list[dict[str, torch.Tensor | int]] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache,
        start_pos: int,
        logits_to_keep: int,
    ) -> torch.Tensor:
        call_index = len(self.calls)

        if call_index >= len(self.scripted_tokens):
            raise RuntimeError(
                "ScriptedGreedyModel received more calls "
                "than scripted"
            )

        self.calls.append(
            {
                "input_ids": input_ids.detach().cpu(),
                "attention_mask": (
                    attention_mask.detach().cpu()
                ),
                "position_ids": (
                    position_ids.detach().cpu()
                ),
                "start_pos": start_pos,
            }
        )

        token_ids = torch.tensor(
            self.scripted_tokens[call_index],
            dtype=torch.long,
            device=input_ids.device,
        )

        batch_size = input_ids.shape[0]

        logits = torch.full(
            (
                batch_size,
                1,
                self.vocab_size,
            ),
            fill_value=-1000.0,
            dtype=torch.float32,
            device=input_ids.device,
        )

        logits.scatter_(
            dim=-1,
            index=token_ids[:, None, None],
            value=1000.0,
        )

        return logits


def test_finished_state_semantics() -> None:
    """
    严格验证 newly_finished 与累计 finished 的语义。

    Prefill:
        request 0 -> EOS
        request 1 -> token 5

    Decode 1:
        request 0 已经 finished，模型结果应被忽略
        request 1 -> EOS

    最终：
        request 0 generated length = 1
        request 1 generated length = 2
    """
    print(
        "===== Deterministic finished-state test ====="
    )

    pad_token_id = 0
    eos_token_id = 2

    model = ScriptedGreedyModel(
        scripted_tokens=[
            # Prefill 输出
            [eos_token_id, 5],

            # 第一次 decode 输出
            #
            # request 0 的 9 会被忽略，因为它已 finished。
            # request 1 输出 EOS。
            [9, eos_token_id],
        ]
    )

    config = SimpleNamespace(
        num_hidden_layers=1,
        num_key_value_heads=1,
        head_dim=4,
    )

    input_ids = torch.tensor(
        [
            [0, 7],
            [8, 9],
        ],
        dtype=torch.long,
    )

    attention_mask = torch.tensor(
        [
            [0, 1],
            [1, 1],
        ],
        dtype=torch.long,
    )

    (
        output_ids,
        output_attention_mask,
        generated_lengths,
        finished,
    ) = (
        batched_left_padding_cached_greedy_generate_with_eos(
            model=model,
            config=config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=8,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
    )

    expected_output_ids = torch.tensor(
        [
            # request 0:
            # prompt -> EOS -> pad
            [0, 7, eos_token_id, pad_token_id],

            # request 1:
            # prompt -> token 5 -> EOS
            [8, 9, 5, eos_token_id],
        ],
        dtype=torch.long,
    )

    expected_output_attention_mask = torch.tensor(
        [
            # EOS 有效，EOS 后 pad 无效
            [0, 1, 1, 0],

            # token 5 和 EOS 都有效
            [1, 1, 1, 1],
        ],
        dtype=torch.long,
    )

    print("\noutput_ids:")
    print(output_ids)

    print("\noutput_attention_mask:")
    print(output_attention_mask)

    print("\ngenerated_lengths:")
    print(generated_lengths)

    print("\nfinished:")
    print(finished)

    assert torch.equal(
        output_ids,
        expected_output_ids,
    )

    assert torch.equal(
        output_attention_mask,
        expected_output_attention_mask,
    )

    assert generated_lengths.tolist() == [
        1,
        2,
    ]

    assert finished.tolist() == [
        True,
        True,
    ]

    # Prefill + 一次 decode 后已经全部结束。
    assert len(model.calls) == 2

    # ============================================================
    # 检查第一次 decode 输入
    # ============================================================

    decode_call = model.calls[1]

    # request 0 已经 finished，所以输入 pad。
    # request 1 仍 active，所以输入上轮生成的 token 5。
    expected_decode_input_ids = torch.tensor(
        [
            [pad_token_id],
            [5],
        ],
        dtype=torch.long,
    )

    assert torch.equal(
        decode_call["input_ids"],
        expected_decode_input_ids,
    )

    # request 0 的新 cache 位置是占位 pad，所以 mask 追加 0。
    # request 1 的 token 5 是有效 token，所以追加 1。
    expected_decode_attention_mask = torch.tensor(
        [
            [0, 1, 0],
            [1, 1, 1],
        ],
        dtype=torch.long,
    )

    assert torch.equal(
        decode_call["attention_mask"],
        expected_decode_attention_mask,
    )

    # request 0 已 finished，position_id 无意义，设置为 0。
    # request 1 prompt length=2，第一个生成 token 的位置是 2。
    expected_decode_position_ids = torch.tensor(
        [
            [0],
            [2],
        ],
        dtype=torch.long,
    )

    assert torch.equal(
        decode_call["position_ids"],
        expected_decode_position_ids,
    )

    print(
        "\nDeterministic finished-state test passed."
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

    state_dict = load_safetensors(
        model_dir
    )

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

    assert not real_missing_keys, (
        f"unexpected missing keys: "
        f"{real_missing_keys}"
    )

    assert not unexpected_keys, (
        f"unexpected keys: "
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


def prepare_benchmark_inputs(
    tokenizer,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    使用完全相同的 prompt 构造 batch。

    这样所有请求的第一次 greedy token 应当相同，
    我们可以把该 token 临时当作 eos_token_id，
    强制整个 batch 在 prefill 后提前结束。
    """
    prompt = (
        "Explain why a key value cache improves "
        "autoregressive language model inference."
    )

    encoded = tokenizer(
        prompt,
        add_special_tokens=False,
        return_tensors="pt",
    )

    single_input_ids = encoded.input_ids.to(device)

    input_ids = single_input_ids.repeat(
        BENCHMARK_BATCH_SIZE,
        1,
    )

    attention_mask = torch.ones_like(
        input_ids,
        dtype=torch.long,
        device=device,
    )

    return input_ids, attention_mask


def clear_cuda_cache() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def benchmark_cuda_call(
    name: str,
    callable_fn,
    repeats: int,
) -> dict[str, float]:
    elapsed_times_ms: list[float] = []
    incremental_peak_allocated_mb: list[float] = []
    incremental_peak_reserved_mb: list[float] = []

    for repeat_idx in range(repeats):
        clear_cuda_cache()

        baseline_allocated = (
            torch.cuda.memory_allocated()
        )

        baseline_reserved = (
            torch.cuda.memory_reserved()
        )

        torch.cuda.reset_peak_memory_stats()

        start_event = torch.cuda.Event(
            enable_timing=True
        )

        end_event = torch.cuda.Event(
            enable_timing=True
        )

        start_event.record()

        result = callable_fn()

        end_event.record()
        torch.cuda.synchronize()

        elapsed_ms = start_event.elapsed_time(
            end_event
        )

        peak_allocated = (
            torch.cuda.max_memory_allocated()
        )

        peak_reserved = (
            torch.cuda.max_memory_reserved()
        )

        incremental_allocated = max(
            0,
            peak_allocated - baseline_allocated,
        )

        incremental_reserved = max(
            0,
            peak_reserved - baseline_reserved,
        )

        elapsed_times_ms.append(elapsed_ms)

        incremental_peak_allocated_mb.append(
            incremental_allocated
            / 1024
            / 1024
        )

        incremental_peak_reserved_mb.append(
            incremental_reserved
            / 1024
            / 1024
        )

        print(
            f"{name}, repeat {repeat_idx}: "
            f"{elapsed_ms:.3f} ms, "
            "incremental peak allocated "
            f"{incremental_peak_allocated_mb[-1]:.2f} MB, "
            "incremental peak reserved "
            f"{incremental_peak_reserved_mb[-1]:.2f} MB"
        )

        del result

    return {
        "median_time_ms": statistics.median(
            elapsed_times_ms
        ),
        "median_peak_allocated_mb": (
            statistics.median(
                incremental_peak_allocated_mb
            )
        ),
        "median_peak_reserved_mb": (
            statistics.median(
                incremental_peak_reserved_mb
            )
        ),
    }


@torch.inference_mode()
def benchmark_real_qwen() -> None:
    print(
        "\n===== Real Qwen3-4B early-stop benchmark ====="
    )

    if not torch.cuda.is_available():
        print(
            "CUDA is unavailable; skipping CUDA benchmark."
        )
        return

    device = torch.device("cuda")
    dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
    )

    pad_token_id = tokenizer.pad_token_id

    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    assert pad_token_id is not None

    config, model = load_mini_model(
        model_dir=MODEL_DIR,
        device=device,
        dtype=dtype,
    )

    input_ids, attention_mask = (
        prepare_benchmark_inputs(
            tokenizer=tokenizer,
            device=device,
        )
    )

    batch_size, prompt_seq_len = (
        input_ids.shape
    )

    print("batch size:", batch_size)
    print("prompt length:", prompt_seq_len)
    print(
        "requested max new tokens:",
        BENCHMARK_MAX_NEW_TOKENS,
    )

    # ============================================================
    # 找到这个 batch 第一个 greedy token
    # ============================================================

    probe_output_ids = (
        batched_left_padding_cached_greedy_generate(
            model=model,
            config=config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1,
        )
    )

    first_tokens = probe_output_ids[
        :,
        -1,
    ]

    print("\nfirst greedy tokens:")
    print(first_tokens)

    # 因为 prompt 完全相同，所以每行应生成相同 token。
    assert torch.all(
        first_tokens == first_tokens[0]
    ), (
        "identical prompts did not produce identical "
        "first greedy tokens"
    )

    forced_eos_token_id = int(
        first_tokens[0].item()
    )

    print(
        "forced benchmark eos_token_id:",
        forced_eos_token_id,
    )

    del probe_output_ids

    # ============================================================
    # 先验证两条路径的输出语义
    # ============================================================

    no_early_stop_output = (
        batched_left_padding_cached_greedy_generate(
            model=model,
            config=config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=(
                BENCHMARK_MAX_NEW_TOKENS
            ),
        )
    )

    (
        early_stop_output,
        early_stop_mask,
        early_stop_lengths,
        early_stop_finished,
    ) = (
        batched_left_padding_cached_greedy_generate_with_eos(
            model=model,
            config=config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=(
                BENCHMARK_MAX_NEW_TOKENS
            ),
            # 人为把第一个预测 token 当作 EOS，
            # 用于稳定触发全部请求提前停止。
            eos_token_id=forced_eos_token_id,
            pad_token_id=pad_token_id,
        )
    )

    assert no_early_stop_output.shape == (
        batch_size,
        prompt_seq_len
        + BENCHMARK_MAX_NEW_TOKENS,
    )

    # 所有请求在第一个生成 token 后结束。
    assert early_stop_output.shape == (
        batch_size,
        prompt_seq_len + 1,
    )

    assert early_stop_lengths.tolist() == (
        [1] * batch_size
    )

    assert early_stop_finished.tolist() == (
        [True] * batch_size
    )

    assert torch.all(
        early_stop_output[:, -1]
        == forced_eos_token_id
    )

    assert torch.all(
        early_stop_mask[:, -1] == 1
    )

    del no_early_stop_output
    del early_stop_output
    del early_stop_mask
    del early_stop_lengths
    del early_stop_finished

    # ============================================================
    # Warmup
    # ============================================================

    print("\n===== Warmup =====")

    _ = batched_left_padding_cached_greedy_generate(
        model=model,
        config=config,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=(
            BENCHMARK_MAX_NEW_TOKENS
        ),
    )

    _ = (
        batched_left_padding_cached_greedy_generate_with_eos(
            model=model,
            config=config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=(
                BENCHMARK_MAX_NEW_TOKENS
            ),
            eos_token_id=forced_eos_token_id,
            pad_token_id=pad_token_id,
        )
    )

    clear_cuda_cache()

    # ============================================================
    # Benchmark
    # ============================================================

    print("\n===== No early stop =====")

    no_early_stop_stats = benchmark_cuda_call(
        name="no early stop",
        callable_fn=lambda: (
            batched_left_padding_cached_greedy_generate(
                model=model,
                config=config,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=(
                    BENCHMARK_MAX_NEW_TOKENS
                ),
            )
        ),
        repeats=BENCHMARK_REPEATS,
    )

    print("\n===== Early stop after first token =====")

    early_stop_stats = benchmark_cuda_call(
        name="early stop",
        callable_fn=lambda: (
            batched_left_padding_cached_greedy_generate_with_eos(
                model=model,
                config=config,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=(
                    BENCHMARK_MAX_NEW_TOKENS
                ),
                eos_token_id=forced_eos_token_id,
                pad_token_id=pad_token_id,
            )
        ),
        repeats=BENCHMARK_REPEATS,
    )

    speedup = (
        no_early_stop_stats["median_time_ms"]
        / early_stop_stats["median_time_ms"]
    )

    allocated_saving = (
        no_early_stop_stats[
            "median_peak_allocated_mb"
        ]
        - early_stop_stats[
            "median_peak_allocated_mb"
        ]
    )

    print("\n===== Benchmark summary =====")

    print(
        "no early stop median time:",
        f"{no_early_stop_stats['median_time_ms']:.3f} ms",
    )

    print(
        "early stop median time:",
        f"{early_stop_stats['median_time_ms']:.3f} ms",
    )

    print(
        "time speedup:",
        f"{speedup:.2f}x",
    )

    print(
        "no early stop incremental peak allocated:",
        f"{no_early_stop_stats['median_peak_allocated_mb']:.2f} MB",
    )

    print(
        "early stop incremental peak allocated:",
        f"{early_stop_stats['median_peak_allocated_mb']:.2f} MB",
    )

    print(
        "allocated-memory difference:",
        f"{allocated_saving:.2f} MB",
    )

    print(
        "no early stop incremental peak reserved:",
        f"{no_early_stop_stats['median_peak_reserved_mb']:.2f} MB",
    )

    print(
        "early stop incremental peak reserved:",
        f"{early_stop_stats['median_peak_reserved_mb']:.2f} MB",
    )

    print(
        "\nImportant interpretation:"
    )

    print(
        "Early stopping should reduce execution time "
        "because decode iterations are skipped."
    )

    print(
        "Peak CUDA memory may remain nearly unchanged "
        "because StaticKVCache reserves "
        "padded_seq_len + max_new_tokens positions "
        "before generation begins."
    )

    print(
        "Meaningful KV-cache memory reclamation will "
        "come later with request removal and paged "
        "KV-cache block management."
    )

    del model

    clear_cuda_cache()


def main() -> None:
    torch.manual_seed(0)

    test_finished_state_semantics()
    benchmark_real_qwen()

    print("\n===== All tests passed =====")
    print("Verified:")
    print(
        "1. newly_finished only marks active requests "
        "that emit EOS in the current step"
    )
    print(
        "2. cumulative finished preserves requests "
        "that ended in previous steps"
    )
    print(
        "3. EOS is a valid output token"
    )
    print(
        "4. post-EOS placeholders are pad tokens "
        "with attention mask 0"
    )
    print(
        "5. all-finished batches stop early"
    )
    print(
        "6. early stopping reduces decode work"
    )
    print(
        "7. StaticKVCache peak memory is expected to "
        "remain similar despite early stopping"
    )


if __name__ == "__main__":
    main()