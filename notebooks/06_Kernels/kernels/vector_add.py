"""
Vector Add — The "Hello World" of GPU kernel programming.

This module implements a simple element-wise vector addition using Triton.
It demonstrates the core concepts: grid/block launches, pointer arithmetic,
masking for out-of-bounds access, and benchmarking against PyTorch.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton Kernel
# ---------------------------------------------------------------------------

@triton.jit
def vector_add_kernel(
    x_ptr,        # pointer to first input vector
    y_ptr,        # pointer to second input vector
    output_ptr,   # pointer to output vector
    n_elements,   # total number of elements
    BLOCK_SIZE: tl.constexpr,  # number of elements each program instance processes
):
    """Each program instance processes BLOCK_SIZE elements."""
    # Which block am I?
    pid = tl.program_id(axis=0)

    # Compute the range of elements this block will process
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Mask to guard against out-of-bounds access
    mask = offsets < n_elements

    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Compute and store
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Element-wise addition using our Triton kernel."""
    assert x.is_cuda and y.is_cuda, "Inputs must be on GPU"
    assert x.shape == y.shape, "Shape mismatch"

    output = torch.empty_like(x)
    n_elements = output.numel()

    # Triton handles the grid launch — we just say how many programs to run
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    vector_add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output


# ---------------------------------------------------------------------------
# Correctness check (run this file directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    size = 100_000
    x = torch.rand(size, device="cuda")
    y = torch.rand(size, device="cuda")

    output_triton = vector_add(x, y)
    output_torch = x + y

    print(f"Max difference: {(output_triton - output_torch).abs().max().item()}")
    assert torch.allclose(output_triton, output_torch), "Results don't match!"
    print("✓ Triton vector_add matches PyTorch")
