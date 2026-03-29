# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Educational deep learning repository exploring neural network architectures through Jupyter notebooks and PyTorch. Notebooks focus on explanation and experimentation; reusable code lives in `src/`.

## Setup

```bash
python3.12 -m venv dl_101
source dl_101/bin/activate
pip install -e .
```

For GPU training via Modal:
```bash
pip install -e ".[gpu]"
modal token set
```

The `pyproject.toml` defines all dependencies. Editable install (`-e .`) makes `src/` importable from any notebook location.

## Running Notebooks

```bash
jupyter notebook
# or
jupyter lab
```

Notebooks are the primary artifact. Open and run cells interactively. Training notebooks include "Run on GPU (Modal)" sections at the end for remote NVIDIA GPU execution. Notebooks also include Colab support — they auto-clone the repo when running in Google Colab.

## Architecture

```
src/                            # Importable Python package
├── utils/device.py             # set_device(), set_seed(), get_amp_config()
├── data/loaders.py             # Dataloader builders (CIFAR-10, MNIST, FashionMNIST)
├── training/trainer.py         # train_model(), train_model_amp(), train_model_grad_accum()
├── training/evaluation.py      # evaluate_model() with confusion matrix
├── models/attention.py         # naive_attention(), flash_attention(), pytorch_sdpa()
├── models/resnet.py            # make_resnet18_cifar10(), CheckpointedResNet18
├── models/lora.py              # LoRALinear, LoRAModel (from-scratch LoRA)
├── models/dqn.py               # QNetwork, ReplayBuffer (DQN components)
├── models/ppo.py               # ActorCritic, ContinuousActorCritic, RolloutBuffer
├── models/grpo.py              # compute_group_advantages(), compute_per_token_kl()
├── models/dpo.py               # compute_dpo_loss()
└── infra/modal_runner.py       # Modal GPU runner: run_training(), run_attention_benchmark()

notebooks/
├── 01_Fundamentals/            # Micrograd, MakeMore series, Attention basics
├── 02_Architectures/
│   ├── CNNs/                   # AlexNet, VGG-16, LeNet-5, ResNet-18, MoE
│   ├── RNNs/                   # RNN/LSTM sentiment analysis
│   ├── Transformers/           # GPT-2, Transformers from scratch, ViT, BERT
│   └── hugging_face/           # HuggingFace playgrounds (NLP, image detection)
├── 03_Training_Techniques/     # Flash Attention, Mixed Precision, Gradient Accumulation, LoRA/QLoRA
├── 04_Reinforcement_Learning/  # DQN, PPO (discrete + continuous), GRPO, PPO Pendulum
├── 05_Papers/                  # DPO, RoPE, Attention Is All You Need, Mixture of Experts
├── 06_Kernels/                 # Triton GPU kernels: Vector Add, Softmax, MatMul, LayerNorm
├── 07_Model_Replications/      # DeepSeek-V3, Nemotron-3 Super, Qwen 3.5, Kimi K2
└── 08_JAX/                     # JAX experiments

data/                           # Auto-downloaded datasets (CIFAR-10, MNIST, FashionMNIST)
```

### Key patterns

- **Notebooks import from `src/`** — each notebook adds the project root to `sys.path` at the top, then imports shared utilities from `src.*`. Colab-aware: auto-clones repo when in Google Colab.
- **New shared code goes in `src/`**, not inline in notebooks. Models → `src/models/`, training utils → `src/training/`, data loading → `src/data/`.
- **Modal for GPU training** — `src/infra/modal_runner.py` provides `run_training()` and `run_attention_benchmark()` that run on remote NVIDIA GPUs. Called via `.remote()` from notebooks.
- **Model Replications** build scaled-down versions of frontier LLMs (DeepSeek-V3, Nemotron-3, Qwen 3.5, Kimi K2) from scratch, reusing concepts from earlier notebooks.
- **All formulas in notebooks use `$...$`** (single dollar, left-aligned), never `$$...$$` (centered display math).
- **Static comparison tables use markdown**, not pandas DataFrames.
- Device detection prioritizes Apple MPS, then CUDA, then CPU (`src/utils/device.py`).
- Datasets auto-download to `data/` on first run via torchvision.

## Dependencies

Core stack: PyTorch, torchvision, transformers (HuggingFace), pytorch-lightning, torchmetrics, matplotlib, numpy, pandas, tqdm, mlxtend.

Optional: `modal` (for remote GPU training — install with `pip install -e ".[gpu]"`).

Optional: `peft`, `bitsandbytes`, `trl`, `datasets`, `accelerate` (for LLM fine-tuning — install with `pip install -e ".[llm]"`).

Optional: `gymnasium` (for RL notebooks — install with `pip install -e ".[rl]"`).

Optional: `triton` (for GPU kernel notebooks — install with `pip install -e ".[kernels]"`).
