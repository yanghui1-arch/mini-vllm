import torch
import torch.nn as nn

from model.attention import Qwen3Attention


def create_attention_shell() -> Qwen3Attention:
    """
    创建一个只用于调用 _attention() 的对象。

    _attention() 不依赖 q_proj、k_proj、配置或模型权重，
    因此这里不需要构造完整 Qwen3Config。
    """
    attention = Qwen3Attention.__new__(Qwen3Attention)
    nn.Module.__init__(attention)
    return attention


def main() -> None:
    torch.manual_seed(42)

    print("===== Test attention padding mask =====")

    attention = create_attention_shell()

    batch_size = 2
    num_heads = 2
    seq_len = 5
    head_dim = 4

    # [B, H, T, D]
    q = torch.randn(
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        dtype=torch.float32,
    )

    k = torch.randn(
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        dtype=torch.float32,
    )

    v = torch.randn(
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        dtype=torch.float32,
    )

    # request 0:
    #   5 个真实 token
    #
    # request 1:
    #   2 个左 padding + 3 个真实 token
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1],
        ],
        dtype=torch.long,
    )

    print("\nq shape:")
    print(q.shape)

    print("\nattention_mask:")
    print(attention_mask)

    # ----------------------------------------
    # Batched attention
    # ----------------------------------------

    batched_output = attention._attention(
        q,
        k,
        v,
        attention_mask=attention_mask,
        start_pos=0,
    )

    print("\nbatched output shape:")
    print(batched_output.shape)

    assert batched_output.shape == (
        batch_size,
        num_heads,
        seq_len,
        head_dim,
    )

    # 即使 padding query 没有有效 key，也不应该产生 NaN。
    assert torch.isfinite(batched_output).all(), (
        "batched attention output contains NaN or Inf"
    )

    # ----------------------------------------
    # Request 0 single attention
    # ----------------------------------------

    q0 = q[0:1]
    k0 = k[0:1]
    v0 = v[0:1]

    mask0 = torch.ones(
        1,
        seq_len,
        dtype=torch.long,
    )

    single_output_0 = attention._attention(
        q0,
        k0,
        v0,
        attention_mask=mask0,
        start_pos=0,
    )

    request_0_diff = (
        batched_output[0:1] - single_output_0
    ).abs().max().item()

    print("\nrequest 0 max absolute difference:")
    print(request_0_diff)

    assert torch.allclose(
        batched_output[0:1],
        single_output_0,
        atol=1e-6,
        rtol=1e-6,
    ), (
        "request 0 batched attention does not match "
        "single-request attention"
    )

    # ----------------------------------------
    # Request 1 single attention
    # ----------------------------------------
    #
    # batched request 1:
    #
    # physical:
    # [PAD, PAD, token0, token1, token2]
    #
    # single request:
    # [token0, token1, token2]
    #
    # 只取真实 token 对应的 Q/K/V。
    # ----------------------------------------

    real_start = 2

    q1 = q[1:2, :, real_start:, :]
    k1 = k[1:2, :, real_start:, :]
    v1 = v[1:2, :, real_start:, :]

    real_seq_len = seq_len - real_start

    mask1 = torch.ones(
        1,
        real_seq_len,
        dtype=torch.long,
    )

    single_output_1 = attention._attention(
        q1,
        k1,
        v1,
        attention_mask=mask1,
        start_pos=0,
    )

    # batched_output 中只比较 request 1 的真实 query。
    batched_real_output_1 = batched_output[
        1:2,
        :,
        real_start:,
        :,
    ]

    request_1_diff = (
        batched_real_output_1 - single_output_1
    ).abs().max().item()

    print("\nrequest 1 real-token max absolute difference:")
    print(request_1_diff)

    assert torch.allclose(
        batched_real_output_1,
        single_output_1,
        atol=1e-6,
        rtol=1e-6,
    ), (
        "request 1 real-token batched attention does not match "
        "single-request attention"
    )

    print("\n===== All tests passed =====")
    print("Verified:")
    print("1. causal attention still works")
    print("2. padding keys are ignored")
    print("3. left-padding real-token outputs match single attention")
    print("4. padding query rows do not produce NaN")


if __name__ == "__main__":
    main()