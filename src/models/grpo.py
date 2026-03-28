"""
GRPO (Group Relative Policy Optimization) components for educational purposes.

Provides:
- compute_group_advantages: Normalize rewards within a group of completions
- compute_per_token_kl: KL divergence between policy and reference model

Reference: Shao et al. (2024). DeepSeekMath: Pushing the Limits of Mathematical
Reasoning in Open Language Models. https://arxiv.org/abs/2402.03300
"""

import torch
import torch.nn.functional as F
import numpy as np


def compute_group_advantages(rewards, eps=1e-8):
    """
    Compute group-relative advantages from a list of reward scores.

    GRPO's key insight: instead of a learned value baseline (PPO), use the
    group mean reward as the baseline. This eliminates the need for a critic.

    A_i = (r_i - mean(r)) / (std(r) + eps)

    Args:
        rewards: list or array of G reward scores for G completions of the same prompt
        eps: small constant for numerical stability

    Returns:
        numpy array of normalized advantages (same length as rewards)
    """
    rewards = np.array(rewards, dtype=np.float32)
    mean = rewards.mean()
    std = rewards.std()
    advantages = (rewards - mean) / (std + eps)
    return advantages


def compute_per_token_kl(logprobs_policy, logprobs_ref):
    """
    Compute per-token KL divergence: KL(policy || ref).

    Used as a penalty to prevent the policy from diverging too far from the
    reference model (reward hacking prevention).

    KL(p || q) = sum_x p(x) * (log p(x) - log q(x))

    For a single token with known log-probs:
    KL_t = exp(logprob_policy) * (logprob_policy - logprob_ref)

    In practice, we approximate with: KL_t ≈ logprob_policy - logprob_ref
    (this is the "unbiased" estimator used in most implementations).

    Args:
        logprobs_policy: tensor of log probabilities from the current policy
        logprobs_ref: tensor of log probabilities from the reference model

    Returns:
        tensor of per-token KL divergence values
    """
    return logprobs_policy - logprobs_ref
