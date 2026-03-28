"""
LoRA (Low-Rank Adaptation) implementations for educational purposes.

Provides:
- LoRALinear: A single linear layer wrapped with LoRA adapters
- LoRAModel: Applies LoRA to target modules in any nn.Module

Reference: Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models.
https://arxiv.org/abs/2106.09685
"""

import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA: output = original(x) + (x @ A^T @ B^T) * (alpha / r)

    The original weights are frozen. Only the low-rank matrices A and B are trainable.
    B is initialized to zeros and A to random normal, so ΔW = BA = 0 at initialization.

    Args:
        original_linear: The nn.Linear layer to wrap
        r: Rank of the low-rank decomposition
        alpha: Scaling factor (effective scale = alpha / r)
    """

    def __init__(self, original_linear, r=8, alpha=16):
        super().__init__()
        self.original = original_linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # Freeze original weights
        for param in self.original.parameters():
            param.requires_grad = False

        # LoRA matrices: W' = W + BA where B is out×r, A is r×in
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # Initialize A with Kaiming uniform (same as nn.Linear default)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        # Original frozen forward
        original_out = self.original(x)
        # LoRA path: x @ A^T @ B^T * scaling
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return original_out + lora_out

    def extra_repr(self):
        return f"r={self.r}, alpha={self.alpha}, scaling={self.scaling:.2f}"


class LoRAModel(nn.Module):
    """
    Wrap any model, replacing target nn.Linear layers with LoRA variants.

    Args:
        model: The base model to wrap
        target_modules: List of module name substrings to match (e.g., ["q_proj", "v_proj"])
        r: LoRA rank
        alpha: LoRA scaling factor
    """

    def __init__(self, model, target_modules, r=8, alpha=16):
        super().__init__()
        self.model = model
        self.target_modules = target_modules
        self.r = r
        self.alpha = alpha

        # Freeze all base model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace matching linear layers with LoRA versions
        self._replaced = 0
        self._apply_lora(self.model, target_modules, r, alpha)

    def _apply_lora(self, module, target_modules, r, alpha):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear) and any(t in name for t in target_modules):
                setattr(module, name, LoRALinear(child, r=r, alpha=alpha))
                self._replaced += 1
            else:
                self._apply_lora(child, target_modules, r, alpha)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def trainable_parameters(self):
        """Count trainable vs total parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable, total

    def print_trainable_parameters(self):
        trainable, total = self.trainable_parameters()
        pct = 100 * trainable / total
        print(f"Trainable: {trainable:,} / {total:,} ({pct:.2f}%)")
