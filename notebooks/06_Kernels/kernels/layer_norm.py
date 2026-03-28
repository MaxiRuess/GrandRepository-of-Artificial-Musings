"""
LayerNorm & RMSNorm — Fused Normalization Kernels.

This module implements both LayerNorm and RMSNorm using Triton. These are the
kernels AI labs ship first — every transformer block has 1-2 normalization calls,
and fusing them eliminates 3+ redundant HBM round-trips.

LayerNorm demonstrates:
1. Two reductions in one pass — mean AND variance (softmax only needed max and sum)
2. Learnable parameters — first kernel that loads external weights (gamma, beta)
3. The same fusion principle as softmax, applied to normalization

RMSNorm is the modern alternative (used by LLaMA, Mistral, Gemma, Qwen):
- Only 1 reduction (mean of squares) instead of 2
- No mean subtraction, no beta parameter — simpler and faster
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# LayerNorm Kernel
# ---------------------------------------------------------------------------

@triton.jit
def layer_norm_kernel(
    input_ptr,          # pointer to input matrix [N x D]
    output_ptr,         # pointer to output matrix [N x D]
    weight_ptr,         # pointer to gamma (learnable scale) [D]
    bias_ptr,           # pointer to beta (learnable shift) [D]
    n_cols,             # number of columns (feature dimension D)
    input_row_stride,   # stride between rows in input
    output_row_stride,  # stride between rows in output
    eps,                # epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,  # must be >= n_cols, power of 2
):
    """Each program normalizes one full row."""
    row_idx = tl.program_id(axis=0)

    # Pointers to this row
    row_start_input = input_ptr + row_idx * input_row_stride
    row_start_output = output_ptr + row_idx * output_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load the row — padding with 0.0 (zeros don't affect mean/var)
    x = tl.load(row_start_input + col_offsets, mask=mask, other=0.0)

    # === Fused LayerNorm (1 pass instead of 4+) ===
    # Reduction 1: mean
    mean = tl.sum(x, axis=0) / n_cols

    # Reduction 2: variance
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / n_cols

    # Normalize
    x_norm = x_centered / tl.sqrt(var + eps)

    # Load learnable parameters and apply scale + shift
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0)
    y = weight * x_norm + bias

    # Store result
    tl.store(row_start_output + col_offsets, y, mask=mask)


# ---------------------------------------------------------------------------
# LayerNorm Python Wrapper
# ---------------------------------------------------------------------------

def layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
               eps: float = 1e-5) -> torch.Tensor:
    """LayerNorm using our Triton kernel."""
    assert x.is_cuda, "Input must be on GPU"
    assert x.ndim == 2, "Input must be 2D (n_rows, n_cols)"

    n_rows, n_cols = x.shape
    assert weight.shape == (n_cols,), f"Weight shape mismatch: {weight.shape} vs ({n_cols},)"
    assert bias.shape == (n_cols,), f"Bias shape mismatch: {bias.shape} vs ({n_cols},)"

    output = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # One program per row (same as softmax)
    grid = (n_rows,)

    layer_norm_kernel[grid](
        x, output, weight, bias,
        n_cols,
        x.stride(0), output.stride(0),
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


# ---------------------------------------------------------------------------
# RMSNorm Kernel
# ---------------------------------------------------------------------------

@triton.jit
def rms_norm_kernel(
    input_ptr,          # pointer to input matrix [N x D]
    output_ptr,         # pointer to output matrix [N x D]
    weight_ptr,         # pointer to gamma (learnable scale) [D]
    n_cols,             # number of columns (feature dimension D)
    input_row_stride,   # stride between rows in input
    output_row_stride,  # stride between rows in output
    eps,                # epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,
):
    """Each program applies RMSNorm to one full row. Simpler than LayerNorm:
    only 1 reduction (mean of squares), no mean subtraction, no beta."""
    row_idx = tl.program_id(axis=0)

    row_start_input = input_ptr + row_idx * input_row_stride
    row_start_output = output_ptr + row_idx * output_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    x = tl.load(row_start_input + col_offsets, mask=mask, other=0.0)

    # === Fused RMSNorm — only 1 reduction ===
    rms = tl.sqrt(tl.sum(x * x, axis=0) / n_cols + eps)
    x_norm = x / rms

    # Load gamma and scale (no beta in RMSNorm)
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0)
    y = weight * x_norm

    tl.store(row_start_output + col_offsets, y, mask=mask)


# ---------------------------------------------------------------------------
# RMSNorm Python Wrapper
# ---------------------------------------------------------------------------

def rms_norm(x: torch.Tensor, weight: torch.Tensor,
             eps: float = 1e-5) -> torch.Tensor:
    """RMSNorm using our Triton kernel."""
    assert x.is_cuda, "Input must be on GPU"
    assert x.ndim == 2, "Input must be 2D (n_rows, n_cols)"

    n_rows, n_cols = x.shape
    assert weight.shape == (n_cols,), f"Weight shape mismatch: {weight.shape} vs ({n_cols},)"

    output = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    grid = (n_rows,)

    rms_norm_kernel[grid](
        x, output, weight,
        n_cols,
        x.stride(0), output.stride(0),
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


# ---------------------------------------------------------------------------
# Correctness checks (run this file directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    N, D = 128, 512
    x = torch.randn(N, D, device="cuda")
    weight = torch.randn(D, device="cuda")
    bias = torch.randn(D, device="cuda")
    eps = 1e-5

    # LayerNorm check
    output_triton = layer_norm(x, weight, bias, eps)
    output_torch = torch.layer_norm(x, [D], weight, bias, eps)

    max_diff = (output_triton - output_torch).abs().max().item()
    print(f"LayerNorm max difference: {max_diff:.2e}")
    assert torch.allclose(output_triton, output_torch, atol=1e-5), "LayerNorm mismatch!"
    print("✓ Triton layer_norm matches torch.layer_norm")

    # RMSNorm check (no torch built-in — compute reference manually)
    rms_ref = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    output_ref = weight * (x / rms_ref)
    output_triton_rms = rms_norm(x, weight, eps)

    max_diff_rms = (output_triton_rms - output_ref).abs().max().item()
    print(f"RMSNorm max difference: {max_diff_rms:.2e}")
    assert torch.allclose(output_triton_rms, output_ref, atol=1e-5), "RMSNorm mismatch!"
    print("✓ Triton rms_norm matches reference")
