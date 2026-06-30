"""CPO unary loss — pure-torch, no transformers/datasets dependency.

Kept in its own module so it can be imported and unit-tested without pulling
in the model/data stack (see test_loss.py). cpo_mini.py imports from here.
"""

from typing import List

import torch


def cpo_unary_loss(
    policy_target_logps: torch.Tensor,
    ref_target_logps: torch.Tensor,
    policy_kl_logps: torch.Tensor,
    ref_kl_logps: torch.Tensor,
    status: List[str],
    target_cluster_ids: torch.Tensor,
    kl_cluster_ids: torch.Tensor,
    K: int,
    beta: float,
    desirable_weight: float,
    undesirable_weight: float,
):
    """Unary CPO loss with per-cluster z_k baseline. Matches HALOs CPOTrainer.loss
    when restricted to a single device (no cross-device reduce needed).
    """
    device = policy_target_logps.device

    # Rewards are log-ratios (no beta); beta enters inside the sigmoid.
    target_rewards = policy_target_logps - ref_target_logps                       # [2n]
    kl_rewards = (policy_kl_logps - ref_kl_logps).detach()                        # [n]

    # Split target by status
    chosen_mask = torch.tensor([s == "chosen" for s in status], device=device)
    rejected_mask = ~chosen_mask
    chosen_rewards = target_rewards[chosen_mask]
    rejected_rewards = target_rewards[rejected_mask]

    target_cluster_ids = target_cluster_ids.to(device)
    kl_cluster_ids = kl_cluster_ids.to(device)
    chosen_clusters = target_cluster_ids[chosen_mask]
    rejected_clusters = target_cluster_ids[rejected_mask]

    # Per-cluster z_k: scatter-add the KL log-ratios into K bins
    nonempty = (kl_rewards.abs() != 0).float()
    sum_per_k = torch.zeros(K, device=device, dtype=kl_rewards.dtype)
    count_per_k = torch.zeros(K, device=device, dtype=kl_rewards.dtype)
    sum_per_k.scatter_add_(0, kl_cluster_ids, kl_rewards * nonempty)
    count_per_k.scatter_add_(0, kl_cluster_ids, nonempty)
    z = (sum_per_k / count_per_k.clamp(min=1)).clamp(min=0)   # [K]

    z_chosen = z[chosen_clusters]
    z_rejected = z[rejected_clusters]

    chosen_losses = desirable_weight * (1 - torch.sigmoid(beta * (chosen_rewards - z_chosen)))
    rejected_losses = undesirable_weight * (1 - torch.sigmoid(beta * (z_rejected - rejected_rewards)))

    losses = torch.cat([chosen_losses, rejected_losses])

    metrics = {
        "loss": losses.mean().item(),
        "chosen_reward": chosen_rewards.mean().item() if chosen_rewards.numel() else 0.0,
        "rejected_reward": rejected_rewards.mean().item() if rejected_rewards.numel() else 0.0,
        "margin": (
            chosen_rewards.mean().item() - rejected_rewards.mean().item()
            if chosen_rewards.numel() and rejected_rewards.numel()
            else 0.0
        ),
        "z_mean": z.mean().item(),
        "z_std": z.std().item() if z.numel() > 1 else 0.0,
        "z_per_cluster": [round(v, 4) for v in z.detach().cpu().tolist()],
    }
    return losses.mean(), metrics
