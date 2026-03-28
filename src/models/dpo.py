"""
DPO (Direct Preference Optimization) components for educational purposes.

Provides:
- compute_dpo_loss: The DPO loss function from preference pairs

Reference: Rafailov et al. (2023). Direct Preference Optimization:
Your Language Model is Secretly a Reward Model.
https://arxiv.org/abs/2305.18290
"""

import torch
import torch.nn.functional as F


def compute_dpo_loss(policy_chosen_logps, policy_rejected_logps,
                     ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """
    Compute the DPO loss from log-probabilities of chosen and rejected completions.

    L = -log σ(β * (log π_θ(y_c|x)/π_ref(y_c|x) - log π_θ(y_r|x)/π_ref(y_r|x)))

    The loss encourages the policy to increase the log-probability gap between
    chosen and rejected completions, relative to the reference model.

    Args:
        policy_chosen_logps: Log-probs of chosen completions under current policy [batch]
        policy_rejected_logps: Log-probs of rejected completions under current policy [batch]
        ref_chosen_logps: Log-probs of chosen completions under reference model [batch]
        ref_rejected_logps: Log-probs of rejected completions under reference model [batch]
        beta: Temperature parameter controlling deviation from reference (default 0.1)

    Returns:
        loss: scalar DPO loss
        chosen_rewards: implicit rewards for chosen completions (for logging)
        rejected_rewards: implicit rewards for rejected completions (for logging)
    """
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)

    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

    return loss, chosen_rewards.mean().item(), rejected_rewards.mean().item()
