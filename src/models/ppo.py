"""
PPO (Proximal Policy Optimization) components for educational purposes.

Provides:
- ActorCritic: Shared-trunk network with policy and value heads (discrete actions)
- ContinuousActorCritic: Gaussian policy for continuous action spaces
- RolloutBuffer: On-policy trajectory storage with GAE computation

Reference: Schulman et al. (2017). Proximal Policy Optimization Algorithms.
https://arxiv.org/abs/1707.06347
"""

import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical, Normal


class ActorCritic(nn.Module):
    """
    Shared-trunk actor-critic network for discrete action spaces.

    Architecture:
        state → shared(128 → ReLU → 128 → ReLU) → actor_head(action_size)  [logits]
                                                  → critic_head(1)           [value]

    Args:
        state_size: Dimension of the state space
        action_size: Number of discrete actions
        hidden_size: Width of shared hidden layers
    """

    def __init__(self, state_size, action_size, hidden_size=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.actor_head = nn.Linear(hidden_size, action_size)
        self.critic_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """Returns (action_logits, state_value)."""
        features = self.shared(x)
        logits = self.actor_head(features)
        value = self.critic_head(features).squeeze(-1)
        return logits, value

    def act(self, state):
        """
        Sample an action from the policy and return (action, log_prob, value).

        Args:
            state: numpy array or tensor of shape [state_size]

        Returns:
            action: int
            log_prob: float (log probability of the action)
            value: float (estimated state value)
        """
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        state = state.unsqueeze(0)

        with torch.no_grad():
            logits, value = self.forward(state)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.item()


class ContinuousActorCritic(nn.Module):
    """
    Actor-critic for continuous action spaces using a Gaussian policy.

    Uses SEPARATE networks for actor and critic to avoid competing gradients.
    log_std is clamped to prevent exploration collapse or explosion.

    Architecture:
        state → actor(256 → Tanh → 256 → Tanh) → mean_head(action_size)
        state → critic(256 → Tanh → 256 → Tanh) → critic_head(1)
        log_std: learnable parameter, clamped to [LOG_STD_MIN, LOG_STD_MAX]

    Args:
        state_size: Dimension of the state space
        action_size: Number of continuous action dimensions
        hidden_size: Width of hidden layers
        action_low: Lower bound for action clipping
        action_high: Upper bound for action clipping
    """

    LOG_STD_MIN = -2.0   # σ min ≈ 0.14
    LOG_STD_MAX = 0.5    # σ max ≈ 1.65

    def __init__(self, state_size, action_size, hidden_size=256,
                 action_low=-2.0, action_high=2.0):
        super().__init__()
        self.action_low = action_low
        self.action_high = action_high

        self.actor = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden_size, action_size)

        self.critic = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.critic_head = nn.Linear(hidden_size, 1)

        self.log_std = nn.Parameter(torch.full((action_size,), -0.5))

        # Orthogonal initialization
        for net in [self.actor, self.critic]:
            for layer in net:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.zeros_(self.critic_head.bias)

    def forward(self, x):
        """Returns (action_mean, action_std, state_value)."""
        mean = self.mean_head(self.actor(x))
        log_std = torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp().expand_as(mean)
        value = self.critic_head(self.critic(x)).squeeze(-1)
        return mean, std, value

    def act(self, state):
        """
        Sample an action from the Gaussian policy.

        Returns:
            action: numpy array clipped to [action_low, action_high]
            log_prob: float (sum of per-dimension log probs)
            value: float (estimated state value)
        """
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        state = state.unsqueeze(0)

        with torch.no_grad():
            mean, std, value = self.forward(state)
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)
            action = torch.clamp(action, self.action_low, self.action_high)

        return action.squeeze(0).numpy(), log_prob.item(), value.item()

    def evaluate(self, states, actions):
        """
        Evaluate actions under the current policy.

        Args:
            states: [batch, state_size]
            actions: [batch, action_size]

        Returns:
            log_probs: [batch] (sum of per-dim log probs)
            entropy: scalar (mean entropy)
            values: [batch]
        """
        mean, std, values = self.forward(states)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1).mean()
        return log_probs, entropy, values


class RolloutBuffer:
    """
    Stores on-policy trajectories and computes returns + GAE advantages.

    Unlike a replay buffer (DQN), this stores sequential transitions from
    the current policy and is cleared after each update. PPO is on-policy:
    data must come from the current policy.
    """

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.advantages = None
        self.returns = None

    def push(self, state, action, reward, value, log_prob, done):
        """Store a single transition."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def compute_returns_and_advantages(self, last_value, gamma=0.99, gae_lambda=0.95):
        """
        Compute GAE advantages and discounted returns.

        GAE: A_t = sum_{l=0}^{T-t} (gamma * lambda)^l * delta_{t+l}
        where delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)

        Args:
            last_value: V(s_T) for the final state (0 if terminal)
            gamma: Discount factor
            gae_lambda: GAE smoothing parameter (0=TD, 1=Monte Carlo)
        """
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]

            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + np.array(self.values, dtype=np.float32)
        self.advantages = advantages
        self.returns = returns

    def get_batches(self, batch_size, device="cpu"):
        """
        Yield random mini-batches of the stored trajectory.

        Returns dicts with tensors: states, actions, log_probs, returns, advantages
        """
        n = len(self.states)
        indices = np.random.permutation(n)

        states = torch.tensor(np.array(self.states), dtype=torch.float32, device=device)
        # Auto-detect action dtype: int→long (discrete), float/array→float32 (continuous)
        action_dtype = torch.long if isinstance(self.actions[0], (int, np.integer)) else torch.float32
        actions = torch.tensor(np.array(self.actions), dtype=action_dtype, device=device)
        log_probs = torch.tensor(self.log_probs, dtype=torch.float32, device=device)
        returns = torch.tensor(self.returns, dtype=torch.float32, device=device)
        advantages = torch.tensor(self.advantages, dtype=torch.float32, device=device)

        # Normalize advantages (standard practice for stability)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]
            yield {
                "states": states[idx],
                "actions": actions[idx],
                "old_log_probs": log_probs[idx],
                "returns": returns[idx],
                "advantages": advantages[idx],
            }

    def clear(self):
        """Reset the buffer for the next rollout."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        self.advantages = None
        self.returns = None

    def __len__(self):
        return len(self.states)
