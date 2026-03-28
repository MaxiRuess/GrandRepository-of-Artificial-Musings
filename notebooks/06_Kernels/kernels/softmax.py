"""
Softmax — A Fused Reduction Kernel.

This module implements a numerically stable, row-wise softmax using Triton.
Unlike vector_add (element-wise, no reduction), softmax demonstrates:

1. Row-wise reductions (tl.max, tl.sum) — computing across elements
2. Numerical stability — subtracting the max before exp() to prevent overflow
3. Kernel fusion — PyTorch does 3 passes over HBM; this kernel does 1

Each Triton program processes one entire row of the input matrix,
loading it into fast on-chip SRAM, performing max → subtract → exp → sum → divide,
and writing the result back in a single pass.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton Kernel
# ---------------------------------------------------------------------------

@triton.jit
def softmax_kernel(
    input_ptr,          # pointer to input matrix
    output_ptr,         # pointer to output matrix
    n_rows,             # number of rows
    n_cols,             # number of columns
    input_row_stride,   # stride between rows in input (in elements)
    output_row_stride,  # stride between rows in output (in elements)
    BLOCK_SIZE: tl.constexpr,  # must be >= n_cols, power of 2
):
    """Each program instance processes one full row."""
    # Which row am I?
    row_idx = tl.program_id(axis=0)

    # Compute pointers to the start of this row
    row_start_input = input_ptr + row_idx * input_row_stride
    row_start_output = output_ptr + row_idx * output_row_stride

    # Column offsets within the row
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load the row — padding with -inf (not 0!)
    # -inf → exp(-inf) = 0, so padded columns vanish from the sum
    # Using 0 would give exp(0) = 1, adding phantom probability mass
    x = tl.load(row_start_input + col_offsets, mask=mask, other=-float('inf'))

    # === Fused softmax (1 pass instead of 3) ===
    # Step 1: Numerical stability — subtract max to prevent exp() overflow
    x_max = tl.max(x, axis=0)
    x = x - x_max

    # Step 2: Exponentiate
    numerator = tl.exp(x)

    # Step 3: Normalize
    denominator = tl.sum(numerator, axis=0)
    y = numerator / denominator

    # Store result
    tl.store(row_start_output + col_offsets, y, mask=mask)


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

def softmax(x: torch.Tensor) -> torch.Tensor:
    """Row-wise softmax using our Triton kernel."""
    assert x.is_cuda, "Input must be on GPU"
    assert x.ndim == 2, "Input must be 2D (n_rows, n_cols)"

    n_rows, n_cols = x.shape
    output = torch.empty_like(x)

    # BLOCK_SIZE must be a power of 2 and >= n_cols
    # so each program can load the entire row
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # One program per row
    grid = (n_rows,)

    softmax_kernel[grid](
        x, output,
        n_rows, n_cols,
        x.stride(0), output.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


# ---------------------------------------------------------------------------
# Correctness check (run this file directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(128, 512, device="cuda")

    output_triton = softmax(x)
    output_torch = torch.softmax(x, dim=-1)

    max_diff = (output_triton - output_torch).abs().max().item()
    print(f"Max difference: {max_diff:.2e}")
    assert torch.allclose(output_triton, output_torch, atol=1e-6), "Results don't match!"
    print("✓ Triton softmax matches torch.softmax")

    # Verify rows sum to 1
    row_sums = output_triton.sum(dim=-1)
    print(f"Row sums — min: {row_sums.min():.6f}, max: {row_sums.max():.6f}")
