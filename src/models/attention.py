"""
Attention implementations for educational purposes.

Provides:
- naive_attention: Standard scaled dot-product attention (O(N^2) memory)
- flash_attention: Tiled attention with online softmax (O(N) memory)
- pytorch_sdpa: Wrapper around torch.nn.functional.scaled_dot_product_attention
"""

import math
import torch
import torch.nn.functional as F


def naive_attention(Q, K, V):
    """
    Standard scaled dot-product attention.

    Computes: softmax(QK^T / sqrt(d)) V

    Args:
        Q: Query tensor [B, N, d]
        K: Key tensor [B, N, d]
        V: Value tensor [B, N, d]

    Returns:
        Output tensor [B, N, d]

    Memory: O(N^2) — materializes the full N x N attention matrix.
    """
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)  # [B, N, N]
    attn_weights = torch.softmax(scores, dim=-1)          # [B, N, N]
    output = attn_weights @ V                              # [B, N, d]
    return output


def flash_attention(Q, K, V, block_size=32):
    """
    Flash Attention forward pass with tiled online softmax.

    Implements Algorithm 1 from Dao et al. (2022). Processes Q, K, V in blocks
    to avoid materializing the full N x N attention matrix. Uses the online
    softmax trick to maintain running statistics (max, sum) across blocks.

    NOTE: This is a pure-PyTorch educational implementation. It demonstrates
    the algorithm's correctness and O(N) memory, but won't match the speed of
    a fused CUDA kernel (which keeps tiles in SRAM).

    Args:
        Q: Query tensor [B, N, d]
        K: Key tensor [B, N, d]
        V: Value tensor [B, N, d]
        block_size: Size of each tile (B_r = B_c = block_size)

    Returns:
        Output tensor [B, N, d]

    Memory: O(N) — only block-sized intermediates, no N x N matrix.
    """
    B, N, d = Q.shape
    B_r = block_size  # row block size (for Q)
    B_c = block_size  # column block size (for K, V)
    scale = 1.0 / math.sqrt(d)

    # Step 1: Initialize output O, running sum l, running max m
    O = torch.zeros_like(Q)                                    # [B, N, d]
    l = torch.zeros(B, N, 1, device=Q.device, dtype=Q.dtype)  # [B, N, 1]
    m = torch.full((B, N, 1), float("-inf"), device=Q.device, dtype=Q.dtype)  # [B, N, 1]

    # Step 2: Split K, V into column blocks; Q into row blocks
    num_col_blocks = math.ceil(N / B_c)
    num_row_blocks = math.ceil(N / B_r)

    # Step 3: Outer loop over K, V column blocks
    for j in range(num_col_blocks):
        j_start = j * B_c
        j_end = min(j_start + B_c, N)
        K_j = K[:, j_start:j_end, :]  # [B, B_c, d]
        V_j = V[:, j_start:j_end, :]  # [B, B_c, d]

        # Step 4: Inner loop over Q row blocks
        for i in range(num_row_blocks):
            i_start = i * B_r
            i_end = min(i_start + B_r, N)
            Q_i = Q[:, i_start:i_end, :]  # [B, B_r, d]

            # Load current running statistics for this row block
            m_i = m[:, i_start:i_end, :]  # [B, B_r, 1]
            l_i = l[:, i_start:i_end, :]  # [B, B_r, 1]
            O_i = O[:, i_start:i_end, :]  # [B, B_r, d]

            # Step 5: Compute block attention scores (only B_r x B_c, not N x N)
            S_ij = (Q_i @ K_j.transpose(-2, -1)) * scale  # [B, B_r, B_c]

            # Step 6: Block-local row max
            m_ij = S_ij.max(dim=-1, keepdim=True).values  # [B, B_r, 1]

            # Step 7: Block-local softmax numerator
            P_ij = torch.exp(S_ij - m_ij)  # [B, B_r, B_c]

            # Step 8: Block-local row sum
            l_ij = P_ij.sum(dim=-1, keepdim=True)  # [B, B_r, 1]

            # Step 9: Update running statistics using online softmax
            m_new = torch.maximum(m_i, m_ij)                           # [B, B_r, 1]
            l_new = torch.exp(m_i - m_new) * l_i + torch.exp(m_ij - m_new) * l_ij  # [B, B_r, 1]

            # Rescale accumulated output and add new contribution
            O_new = (
                torch.exp(m_i - m_new) * l_i * O_i  # rescale previous output
                + torch.exp(m_ij - m_new) * (P_ij @ V_j)  # add new block's contribution
            ) / l_new

            # Write back updated statistics
            m[:, i_start:i_end, :] = m_new
            l[:, i_start:i_end, :] = l_new
            O[:, i_start:i_end, :] = O_new

    return O


def pytorch_sdpa(Q, K, V):
    """
    PyTorch 2.0+ fused scaled dot-product attention.

    Dispatches to the best available backend:
    - FlashAttention-2 (CUDA, bf16/fp16)
    - Memory-efficient attention (xFormers)
    - Math fallback (standard)

    Args:
        Q: Query tensor [B, N, d]
        K: Key tensor [B, N, d]
        V: Value tensor [B, N, d]

    Returns:
        Output tensor [B, N, d]
    """
    return F.scaled_dot_product_attention(Q, K, V)
