"""
Matrix Multiply — A Tiled, Compute-Bound Kernel.

This module implements a tiled matrix multiplication C = A @ B using Triton.
Unlike vector_add (element-wise) and softmax (memory-bound reduction), matmul
demonstrates:

1. Tiling — loading BLOCK-sized tiles into SRAM to maximize data reuse
2. 2D grid — each program computes a BLOCK_M x BLOCK_N tile of the output
3. Compute-bound optimization — arithmetic intensity increases with tile size
4. tl.dot — Triton's tile-level matrix multiply, mapped to GPU tensor cores

This is the most important kernel in deep learning: every linear layer,
attention projection, and embedding lookup is a matrix multiplication.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton Kernel
# ---------------------------------------------------------------------------

@triton.jit
def matmul_kernel(
    a_ptr,      # pointer to matrix A [M x K]
    b_ptr,      # pointer to matrix B [K x N]
    c_ptr,      # pointer to output C [M x N]
    M,          # number of rows in A / C
    N,          # number of columns in B / C
    K,          # shared dimension (cols of A, rows of B)
    stride_am,  # stride for A rows
    stride_ak,  # stride for A columns
    stride_bk,  # stride for B rows
    stride_bn,  # stride for B columns
    stride_cm,  # stride for C rows
    stride_cn,  # stride for C columns
    BLOCK_M: tl.constexpr,  # tile height (rows of C per program)
    BLOCK_N: tl.constexpr,  # tile width (cols of C per program)
    BLOCK_K: tl.constexpr,  # inner dimension tile size (>= 16 for tl.dot)
):
    """Each program computes a BLOCK_M x BLOCK_N tile of C."""
    # 2D grid: which tile of C am I computing?
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Row and column offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Pointers to the first A tile [BLOCK_M x BLOCK_K] and B tile [BLOCK_K x BLOCK_N]
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + tl.arange(0, BLOCK_K)[None, :] * stride_ak
    b_ptrs = b_ptr + tl.arange(0, BLOCK_K)[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Accumulator — must be float32 for precision even if inputs are float16
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in tiles
    for k in range(0, K, BLOCK_K):
        # Boundary masks — tiles at edges may extend past M or K
        offs_k = k + tl.arange(0, BLOCK_K)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        # Load tiles from HBM → SRAM (masked loads return 0.0)
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Tile-level matrix multiply — maps to tensor cores on modern GPUs
        acc += tl.dot(a_tile, b_tile)

        # Advance pointers to next K tile
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Store the output tile
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Matrix multiplication C = A @ B using our Triton kernel."""
    assert a.is_cuda and b.is_cuda, "Inputs must be on GPU"
    assert a.ndim == 2 and b.ndim == 2, "Inputs must be 2D"
    assert a.shape[1] == b.shape[0], f"Shape mismatch: {a.shape} @ {b.shape}"

    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Block sizes tuned for T4 (64KB SRAM per SM):
    # A tile: 64x32 = 8KB, B tile: 32x64 = 8KB, acc: 64x64 = 16KB → 32KB total
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32

    # 2D grid: one program per output tile
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c


# ---------------------------------------------------------------------------
# Correctness check (run this file directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    M, N, K = 512, 512, 512
    a = torch.randn(M, K, device="cuda")
    b = torch.randn(K, N, device="cuda")

    output_triton = matmul(a, b)
    output_torch = torch.matmul(a, b)

    max_diff = (output_triton - output_torch).abs().max().item()
    print(f"Max difference: {max_diff:.2e}")
    # Looser tolerance than softmax — matmul accumulates K=512 products per
    # output element, and different summation orders cause rounding differences
    assert torch.allclose(output_triton, output_torch, atol=1e-2), "Results don't match!"
    print("✓ Triton matmul matches torch.matmul")
