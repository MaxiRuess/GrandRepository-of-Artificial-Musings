import torch
import numpy as np


def set_device():
    """
    Set the device to use for training and inference.
    Returns 'mps', 'cuda', or 'cpu'.
    """
    print(f"PyTorch version: {torch.__version__}")
    print(f"Is MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
    print(f"Is MPS available? {torch.backends.mps.is_available()}")

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    return device


def set_deterministic():
    """Set deterministic behavior for reproducibility."""
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif torch.backends.mps.is_available():
        pass
    print("Set deterministic behavior")


def set_seed(seed):
    """Set seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Set seed for reproducibility: {seed}")
