# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Educational deep learning repository exploring neural network architectures through Jupyter notebooks and PyTorch. Notebooks focus on explanation and experimentation; reusable code lives in `src/`.

## Setup

```bash
python -m venv dl_101
source dl_101/bin/activate
pip install -e .
```

The `pyproject.toml` defines all dependencies. Editable install (`-e .`) makes `src/` importable from any notebook location.

## Running Notebooks

```bash
jupyter notebook
# or
jupyter lab
```

Notebooks are the primary artifact. Open and run cells interactively.

## Architecture

```
src/                            # Importable Python package
├── utils/device.py             # set_device(), set_deterministic(), set_seed()
├── data/loaders.py             # Dataloader builders (CIFAR-10, MNIST, FashionMNIST)
├── training/trainer.py         # Generic train_model() loop
├── training/evaluation.py      # evaluate_model() with confusion matrix
└── models/                     # Reusable model definitions (new modules go here)

notebooks/
├── 01_Fundamentals/            # Micrograd, MakeMore series, Attention basics
├── 02_Architectures/
│   ├── CNNs/                   # AlexNet, VGG-16, LeNet-5, ResNet-18, MoE
│   ├── RNNs/                   # RNN/LSTM sentiment analysis
│   ├── Transformers/           # GPT-2, Transformers from scratch, ViT, BERT
│   └── hugging_face/           # HuggingFace playgrounds (NLP, image detection)
├── 03_Training/                # Training techniques, fine-tuning (Flash Attention, LLaMA, LoRA)
├── 04_Reinforcement_Learning/  # RL algorithms and experiments
└── 05_Papers/                  # Paper reimplementations

data/                           # Auto-downloaded datasets (CIFAR-10, MNIST, FashionMNIST)
```

### Key patterns

- **Notebooks import from `src/`** — each notebook adds the project root to `sys.path` at the top, then imports shared utilities from `src.*`.
- **New shared code goes in `src/`**, not inline in notebooks. Models → `src/models/`, training utils → `src/training/`, data loading → `src/data/`.
- Device detection prioritizes Apple MPS, then CUDA, then CPU (`src/utils/device.py`).
- Datasets auto-download to `data/` on first run via torchvision.

## Dependencies

Core stack: PyTorch, torchvision, transformers (HuggingFace), pytorch-lightning, torchmetrics, matplotlib, numpy, pandas, tqdm, mlxtend.
