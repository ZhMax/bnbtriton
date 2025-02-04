import math

import torch

from bnbtriton.triton_utils import is_triton_available

if not is_triton_available():

    def quantize_rowwise_bf16(x: torch.Tensor):
        return None
else:
    import triton
    import triton.language as tl

    # rowwise quantize

    # TODO: autotune this better.
    @triton.autotune(
        configs=[
            triton.Config({}, num_stages=1, num_warps=8),
            triton.Config({}, num_stages=2, num_warps=8),
            triton.Config({}, num_stages=4, num_warps=8),
            triton.Config({}, num_stages=8, num_warps=8),
            triton.Config({}, num_stages=1),
            triton.Config({}, num_stages=2),
            triton.Config({}, num_stages=4),
            triton.Config({}, num_stages=8),
            triton.Config({}, num_warps=1),
            triton.Config({}, num_warps=2),
            triton.Config({}, num_warps=4),
            triton.Config({}, num_warps=8),
        ],
        key=['n_elements'],
    )
    @triton.jit
    def _prune_rowwise(
        X,
        output_mask,
        M, K,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
        # BLOCK_K: tl.constexpr,
        P2: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)
        rk = tl.arange(0, BLOCK_SIZE)

        # amax_val = 0.0
        # for k in range(0, tl.cdiv(K, BLOCK_SIZE)):
        #     k_remaining = K - k * BLOCK_SIZE
        #     offsets = pid * K + k * BLOCK_SIZE + rk
        #     val = tl.load(X + offsets, mask=rk < k_remaining, other=0.0)
        #     abs_val = tl.abs(val)
        #     max_val = tl.max(val)
        #     if max_val > amax_val:
        #         amax_val = max_val

        mask_val = False
        for k in range(0, tl.cdiv(K, BLOCK_SIZE)):
            if mask_val == False:
                k_remaining = K - k * BLOCK_SIZE
                offsets = pid * K + k * BLOCK_SIZE + rk
                val = tl.load(X + offsets, mask=rk < k_remaining, other=0.0)
                abs_val = tl.abs(val)
                max_val = tl.max(abs_val)
                if max_val > 0.0:
                    mask_val = True

        tl.store(output_mask + pid, mask_val)

        # block_start = pid * BLOCK_SIZE
        # arange = tl.arange(0, P2)
        # offsets = block_start + arange
        # row_mask = arange < BLOCK_SIZE
        # x = tl.load(x_ptr + offsets, mask=row_mask)

        # y = tl.zeros_like(x)

        # if x > y:
        #     tl.store(output_mask + pid, True)
        # else:
        #     tl.store(output_mask + pid, False)

        # abs_x = tl.abs(x)
        # sum_val = tl.sum(abs_x, axis=0)
        # if sum_val == 0.0:
        #     tl.store(output_mask + pid, True)
        # else:
        #     tl.store(output_mask + pid, False)

    def prune_rowwise(x: torch.Tensor):
        output_mask = torch.zeros(x.shape[0], device=x.device, dtype=torch.bool)

        

        assert x.is_cuda and output_mask.is_cuda
        # n_elements = x.numel()
        M, K = x.shape
        n_elements = M * K
        grid = lambda META: (M, )
        P2 = int(2 ** (math.ceil(math.log2(K))))
        block_size = min(P2, 1024)
        _prune_rowwise[grid](x, output_mask, M, K, n_elements, BLOCK_SIZE=block_size, P2=P2)
        return output_mask
