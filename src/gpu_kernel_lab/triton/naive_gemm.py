from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _naive_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    m: tl.constexpr,
    n: tl.constexpr,
    k: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, k, BLOCK_K):
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + (k_start + offs_k[None, :]) * stride_ak,
            mask=(offs_m[:, None] < m) & (k_start + offs_k[None, :] < k),
            other=0.0,
        )
        b = tl.load(
            b_ptr + (k_start + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn,
            mask=(k_start + offs_k[:, None] < k) & (offs_n[None, :] < n),
            other=0.0,
        )
        accumulator += tl.dot(a, b)

    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        accumulator,
        mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
    )


def triton_naive_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("the Triton provider requires CUDA tensors")
    if a.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("the first Triton exercise supports float16 and bfloat16")
    m, k = a.shape
    _, n = b.shape
    output = torch.empty((m, n), device=a.device, dtype=a.dtype)
    block_m, block_n, block_k = 32, 32, 32
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _naive_gemm_kernel[grid](
        a, b, output, m, n, k,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
    )
    return output

