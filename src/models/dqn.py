"""
DQN (Deep Q-Network) components for educational purposes.

Provides:
- QNetwork: Simple MLP for Q-value approximation
- ReplayBuffer: Fixed-size experience replay buffer

Reference: Mnih et al. (2015). Human-level control through deep reinforcement learning.
https://www.nature.com/articles/nature14236
"""

import random
from collections import deque

import torch
import torch.nn as nn
import numpy as np


class QNetwork(nn.Module):
    """
    Simple MLP for Q-value approximation.

    Maps state vectors to Q-values for each action.
    Architecture: state → hidden → ReLU → hidden → ReLU → actions

    Args:
        state_size: Dimension of the state space
        action_size: Number of discrete actions
        hidden_size: Width of hidden layers
    """

    def __init__(self, state_size, action_size, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """
    Fixed-size experience replay buffer with uniform random sampling.

    Stores transitions (state, action, reward, next_state, done) and
    returns batched tensors for training.

    Args:
        capacity: Maximum number of transitions to store
    """

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Store a transition."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size, device="cpu"):
        """Sample a random batch and return as tensors."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.tensor(np.array(states), dtype=torch.float32, device=device),
            torch.tensor(actions, dtype=torch.long, device=device),
            torch.tensor(rewards, dtype=torch.float32, device=device),
            torch.tensor(np.array(next_states), dtype=torch.float32, device=device),
            torch.tensor(dones, dtype=torch.float32, device=device),
        )

    def __len__(self):
        return len(self.buffer)
