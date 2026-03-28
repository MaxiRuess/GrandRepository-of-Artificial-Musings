"""
Modal GPU runner for DeepLearning_101.

Run training jobs and attention benchmarks on remote NVIDIA GPUs via Modal.

Usage from notebooks:
    from src.infra.modal_runner import run_training, run_attention_benchmark

    results = run_training.remote(
        model_factory_name="make_resnet18_cifar10",
        trainer_name="train_model_amp",
        dataset="cifar10",
        epochs=5,
    )

Setup:
    pip install modal
    modal token set
"""

from pathlib import Path

import modal

# --- Modal configuration ---

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent  # src/infra/../../ = project root

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision",
        "torchmetrics",
        "tqdm",
        "numpy",
        "matplotlib",
        "mlxtend",
    )
    .copy_local_dir(str(PROJECT_DIR / "src"), "/root/project/src")
    .env({"PYTHONPATH": "/root/project"})
)

dataset_volume = modal.Volume.from_name("dl101-datasets", create_if_missing=True)

app = modal.App("dl101-training")


# --- Helpers (run inside container) ---

def _build_transforms(dataset_name):
    """Build train/test transforms for a given dataset."""
    from torchvision import transforms

    if dataset_name == "cifar10":
        train_t = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
        ])
        test_t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
        ])
    elif dataset_name in ("mnist", "fashionmnist"):
        train_t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        test_t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return train_t, test_t


def _build_dataloaders(dataset_name, batch_size, data_path="/root/data"):
    """Build dataloaders inside the Modal container."""
    from src.data.loaders import (
        build_dataloaders_cifar,
        build_dataloaders_mnist,
        build_dataloaders_fmnist,
    )

    train_t, test_t = _build_transforms(dataset_name)
    num_workers = 4

    loader_map = {
        "cifar10": build_dataloaders_cifar,
        "mnist": build_dataloaders_mnist,
        "fashionmnist": build_dataloaders_fmnist,
    }

    if dataset_name not in loader_map:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_dl, test_dl, _, class_names = loader_map[dataset_name](
        data_path, train_t, test_t, num_workers, batch_size
    )
    return train_dl, test_dl, class_names


def _get_model(name):
    """Resolve a model factory name to an nn.Module instance."""
    from src.models.resnet import make_resnet18_cifar10, CheckpointedResNet18

    registry = {
        "make_resnet18_cifar10": make_resnet18_cifar10,
        "CheckpointedResNet18": lambda: CheckpointedResNet18(),
    }

    if name not in registry:
        raise ValueError(f"Unknown model: {name}. Available: {list(registry.keys())}")

    return registry[name]()


def _get_trainer(name):
    """Resolve a trainer function name."""
    from src.training.trainer import train_model, train_model_amp, train_model_grad_accum

    registry = {
        "train_model": train_model,
        "train_model_amp": train_model_amp,
        "train_model_grad_accum": train_model_grad_accum,
    }

    if name not in registry:
        raise ValueError(f"Unknown trainer: {name}. Available: {list(registry.keys())}")

    return registry[name]


# --- Modal functions ---

@app.function(
    gpu="A100",
    image=image,
    volumes={"/root/data": dataset_volume},
    timeout=1800,
)
def run_training(
    model_factory_name: str,
    trainer_name: str,
    dataset: str = "cifar10",
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 0.001,
    accumulation_steps: int = 1,
    use_amp: bool = False,
) -> dict:
    """
    Train a model on a remote GPU and return results.

    Args:
        model_factory_name: "make_resnet18_cifar10" or "CheckpointedResNet18"
        trainer_name: "train_model", "train_model_amp", or "train_model_grad_accum"
        dataset: "cifar10", "mnist", or "fashionmnist"
        epochs: Number of training epochs
        batch_size: Mini-batch size
        lr: Learning rate
        accumulation_steps: For train_model_grad_accum only
        use_amp: For train_model_grad_accum only

    Returns:
        dict with training results + GPU metadata
    """
    import torch
    import torch.nn as nn
    from src.utils.device import set_seed, set_deterministic

    device = "cuda"
    set_seed(42)
    set_deterministic()

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Model: {model_factory_name}, Trainer: {trainer_name}")
    print(f"Dataset: {dataset}, Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")

    # Build components
    model = _get_model(model_factory_name)
    train_dl, test_dl, class_names = _build_dataloaders(dataset, batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    trainer_fn = _get_trainer(trainer_name)

    # Build kwargs based on trainer type
    kwargs = dict(
        model=model,
        train_dataloader=train_dl,
        test_dataloader=test_dl,
        epochs=epochs,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
    )

    if trainer_name == "train_model_grad_accum":
        kwargs["accumulation_steps"] = accumulation_steps
        kwargs["use_amp"] = use_amp

    # Train
    _, results = trainer_fn(**kwargs)

    # Commit dataset volume so downloads are cached
    dataset_volume.commit()

    # Add GPU metadata
    results["gpu_name"] = torch.cuda.get_device_name()
    results["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_mem / 1024**3
    results["peak_memory_mb"] = torch.cuda.max_memory_allocated() / 1024**2

    return results


@app.function(
    gpu="A100",
    image=image,
    timeout=600,
)
def run_attention_benchmark(
    seq_lengths: list[int] = None,
    d: int = 64,
    batch_size: int = 4,
    num_warmup: int = 3,
    num_trials: int = 10,
    flash_max_N: int = 1024,
) -> dict:
    """
    Benchmark attention implementations on a remote GPU.

    Returns:
        dict with timing results per sequence length + GPU metadata
    """
    import torch
    import time
    import math
    import numpy as np
    from src.models.attention import naive_attention, flash_attention, pytorch_sdpa

    if seq_lengths is None:
        seq_lengths = [128, 256, 512, 1024, 2048, 4096]

    device = "cuda"
    print(f"GPU: {torch.cuda.get_device_name()}")

    def benchmark_fn(fn, Q, K, V):
        for _ in range(num_warmup):
            _ = fn(Q, K, V)
        torch.cuda.synchronize()

        times = []
        for _ in range(num_trials):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = fn(Q, K, V)
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)

        return float(np.median(times))

    results = {"seq_lengths": [], "naive_ms": [], "flash_ms": [], "sdpa_ms": []}

    for N in seq_lengths:
        Q = torch.randn(batch_size, N, d, device=device)
        K = torch.randn(batch_size, N, d, device=device)
        V = torch.randn(batch_size, N, d, device=device)

        t_naive = benchmark_fn(naive_attention, Q, K, V)

        if N <= flash_max_N:
            t_flash = benchmark_fn(flash_attention, Q, K, V)
        else:
            t_flash = float("nan")

        t_sdpa = benchmark_fn(pytorch_sdpa, Q, K, V)

        results["seq_lengths"].append(N)
        results["naive_ms"].append(t_naive)
        results["flash_ms"].append(t_flash)
        results["sdpa_ms"].append(t_sdpa)

        flash_str = f"{t_flash:.2f} ms" if not math.isnan(t_flash) else "skipped"
        print(f"N={N:5d} | Naive: {t_naive:8.2f} ms | Flash: {flash_str:>12s} | SDPA: {t_sdpa:8.2f} ms")

    results["gpu_name"] = torch.cuda.get_device_name()
    return results
