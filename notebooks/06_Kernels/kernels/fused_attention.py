"""
Fused Attention — The Crown Jewel Kernel.

This module implements a FlashAttention-style fused attention kernel using Triton:
    O = softmax(Q @ K^T / sqrt(d)) @ V
without ever materializing the N x N attention matrix.

This kernel combines every concept from the previous four notebooks:
1. Pointer arithmetic + masking (01_Vector_Add)
2. Row-wise reductions + fusion (02_Softmax)
3. Tiling with K-loop (03_Matrix_Multiply)
4. Multiple external inputs (04_LayerNorm)

Plus the key new idea: online softmax (running max + running sum) across K/V
tiles, enabling incremental computation without the full score matrix.

Each program processes one query row, looping over all K/V rows in blocks of
BLOCK_K. The three online softmax state variables (running max m_i, running
sum l_i, running output o_i) are updated at each tile and normalized at the end.
"""

import math
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Fused Attention Kernel
# ---------------------------------------------------------------------------

@triton.jit
def fused_attention_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    N,              # sequence length
    d,              # head dimension
    stride_qn, stride_qd,   # Q strides [N, d]
    stride_kn, stride_kd,   # K strides [N, d]
    stride_vn, stride_vd,   # V strides [N, d]
    stride_on, stride_od,   # O strides [N, d]
    BLOCK_K: tl.constexpr,  # K/V tile size (how many rows per iteration)
    D: tl.constexpr,        # head dimension rounded to power of 2
):
    """Each program computes one row of the output O."""
    # Which query row am I processing?
    q_idx = tl.program_id(0)

    # Load this query row: q[D] (loaded once, reused across all K/V blocks)
    d_offsets = tl.arange(0, D)
    d_mask = d_offsets < d
    q = tl.load(Q_ptr + q_idx * stride_qn + d_offsets * stride_qd,
                mask=d_mask, other=0.0).to(tl.float32)

    # Scaling factor: 1/sqrt(d)
    scale = 1.0 / tl.sqrt(tl.cast(d, tl.float32))

    # === Online softmax state ===
    m_i = float('-inf')                           # running max (scalar)
    l_i = 0.0                                     # running sum / denominator (scalar)
    o_i = tl.zeros((D,), dtype=tl.float32)        # running output accumulator [D]

    # Loop over K/V in blocks of BLOCK_K
    for k_start in range(0, N, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < N

        # Load K block: [BLOCK_K, D]
        k_ptrs = K_ptr + k_offsets[:, None] * stride_kn + d_offsets[None, :] * stride_kd
        k_block = tl.load(k_ptrs, mask=k_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

        # Compute scores: q[D] dot K_block[BLOCK_K, D]^T -> [BLOCK_K]
        scores = tl.sum(q[None, :] * k_block, axis=1) * scale
        scores = tl.where(k_mask, scores, float('-inf'))

        # --- Online softmax update ---
        # Block max
        m_ij = tl.max(scores, axis=0)
        m_new = tl.maximum(m_i, m_ij)

        # Rescale old accumulator for the new max
        alpha = tl.exp(m_i - m_new)

        # New block's softmax weights (unnormalized)
        p_ij = tl.exp(scores - m_new)

        # Update running sum
        l_new = alpha * l_i + tl.sum(p_ij, axis=0)

        # Load V block: [BLOCK_K, D]
        v_ptrs = V_ptr + k_offsets[:, None] * stride_vn + d_offsets[None, :] * stride_vd
        v_block = tl.load(v_ptrs, mask=k_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

        # Update running output: rescale old + add weighted V
        o_i = alpha * o_i + tl.sum(p_ij[:, None] * v_block, axis=0)

        # Advance state
        m_i = m_new
        l_i = l_new

    # Final normalization
    o_i = o_i / l_i

    # Store output row
    o_ptrs = O_ptr + q_idx * stride_on + d_offsets * stride_od
    tl.store(o_ptrs, o_i, mask=d_mask)


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

def fused_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Fused attention: O = softmax(Q @ K^T / sqrt(d)) @ V, no N x N matrix."""
    assert Q.is_cuda and K.is_cuda and V.is_cuda, "Inputs must be on GPU"
    assert Q.ndim == 2 and K.ndim == 2 and V.ndim == 2, "Inputs must be 2D [N, d]"
    assert Q.shape == K.shape == V.shape, "Q, K, V must have the same shape"

    N, d = Q.shape
    O = torch.empty_like(Q)

    BLOCK_K = 64
    D = triton.next_power_of_2(d)

    # One program per query row
    grid = (N,)

    fused_attention_kernel[grid](
        Q, K, V, O,
        N, d,
        Q.stride(0), Q.stride(1),
        K.stride(0), K.stride(1),
        V.stride(0), V.stride(1),
        O.stride(0), O.stride(1),
        BLOCK_K=BLOCK_K, D=D,
    )
    return O


# ---------------------------------------------------------------------------
# Correctness check (run this file directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    N, d = 256, 64
    Q = torch.randn(N, d, device="cuda")
    K = torch.randn(N, d, device="cuda")
    V = torch.randn(N, d, device="cuda")

    output_triton = fused_attention(Q, K, V)

    # Reference: naive attention (materializes N x N)
    scores = Q @ K.T / math.sqrt(d)
    attn_weights = torch.softmax(scores, dim=-1)
    output_ref = attn_weights @ V

    max_diff = (output_triton - output_ref).abs().max().item()
    print(f"Max difference: {max_diff:.2e}")
    assert torch.allclose(output_triton, output_ref, atol=1e-2), "Results don't match!"
    print("✓ Triton fused_attention matches naive attention")
