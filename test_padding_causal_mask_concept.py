import torch

def build_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids = position_ids.masked_fill(attention_mask == 0, 0)
    return position_ids

def build_allowed_attention_mask(
    attention_mask: torch.Tensor
):
    if attention_mask.ndim != 2:
        raise ValueError()
        
    batch_size, seq_len = attention_mask.shape
    
    
    causal_mask = torch.tril(
        torch.ones(
            seq_len,
            seq_len,
            dtype=torch.bool,
            device=attention_mask.device
        )
    )
    
    causal_mask = causal_mask.view(1, 1, seq_len, seq_len)
    padding_key_mask = attention_mask.to(torch.bool).view(batch_size, 1, 1, seq_len)
    allowed_mask = causal_mask & padding_key_mask
    return allowed_mask

def bool_row_to_int_list(row: torch.Tensor) -> list[int]:
    return row.to(torch.int32).tolist()
    
def main() -> None:
    print("===== Padding + causal mask concept test =====")

    # request 0:
    #   8 个真实 token
    #
    # request 1:
    #   左侧 3 个 PAD，后面 5 个真实 token
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1, 1],
        ],
        dtype=torch.long,
    )
    print("\nattention_mask:")
    print(attention_mask)

    position_ids = build_position_ids(attention_mask)

    print("\nposition_ids:")
    print(position_ids)

    expected_position_ids = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [0, 0, 0, 0, 1, 2, 3, 4],
        ],
        dtype=torch.long,
    )
    
    assert torch.equal(position_ids, expected_position_ids), (
        "position_ids are incorrect\n"
        f"actual:\n{position_ids}\n"
        f"expected:\n{expected_position_ids}"
    )
    
    allowed_mask = build_allowed_attention_mask(attention_mask)
    print("\nallowed_mask shape:")
    print(allowed_mask.shape)

    assert allowed_mask.shape == (2, 1, 8, 8)

    request_0_last_query = allowed_mask[0, 0, 7]
    print("\nrequest 0, query position 7:")
    print(bool_row_to_int_list(request_0_last_query))

    expected_request_0_last = torch.tensor(
        [True, True, True, True, True, True, True, True]
    )

    assert torch.equal(
        request_0_last_query.cpu(),
        expected_request_0_last,
    )

    # request 1 最后一个真实 token：
    # 必须屏蔽前三个 padding key。
    request_1_last_query = allowed_mask[1, 0, 7]

    print("\nrequest 1, query position 7:")
    print(bool_row_to_int_list(request_1_last_query))

    expected_request_1_last = torch.tensor(
        [False, False, False, True, True, True, True, True]
    )

    assert torch.equal(
        request_1_last_query.cpu(),
        expected_request_1_last,
    )

    # request 1 的第一个真实 token 位于物理位置 3。
    #
    # 它只能读取自己：
    # - 位置 0、1、2 是 padding，必须屏蔽
    # - 位置 4、5、6、7 是未来 token，causal mask 会屏蔽
    request_1_first_real_query = allowed_mask[1, 0, 3]

    print("\nrequest 1, first real query at physical position 3:")
    print(bool_row_to_int_list(request_1_first_real_query))

    expected_request_1_first_real = torch.tensor(
        [False, False, False, True, False, False, False, False]
    )

    assert torch.equal(
        request_1_first_real_query.cpu(),
        expected_request_1_first_real,
    )

    print("\n===== All tests passed =====")
    print("Verified:")
    print("1. left-padding position_ids are correct")
    print("2. causal mask blocks future tokens")
    print("3. padding key mask blocks PAD K/V")
    print("4. real tokens cannot attend to left-padding tokens")


if __name__ == "__main__":
    main()