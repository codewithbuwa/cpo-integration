"""Verify the CPO loss math in cpo_loss.py with synthetic tensors.

Three sanity tests:
  T1. K=1 reduces to KTO (single cluster: z is a scalar, applied uniformly).
  T2. Loss is finite and non-negative on random inputs.
  T3. Per-cluster z values respond to per-cluster KL signal (mechanism check).

Imports the real loss from cpo_loss (pure-torch, no transformers dependency)
so this test cannot drift from the code it claims to verify.
"""

import random

import torch

from cpo_loss import cpo_unary_loss

torch.manual_seed(0)
random.seed(0)

# Construct a synthetic microbatch
n = 8
K = 4

# Status: first n are chosen, next n are rejected
status = ["chosen"] * n + ["rejected"] * n

# Per-example cluster assignments — explicit pattern
cluster_ids_target = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3] * 2, dtype=torch.long)
cluster_ids_kl = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long)

# Log-probs (synthetic — we don't need to actually train a model for this test)
policy_target = torch.randn(2 * n) * 0.1 - 5.0   # log-probs are negative
ref_target = torch.randn(2 * n) * 0.1 - 5.0
policy_kl = torch.randn(n) * 0.1 - 5.0
ref_kl = torch.randn(n) * 0.1 - 5.0

# ---------------- T1: K=1 reduces to KTO behavior ---------------- #
print("T1: K=1 — z is a scalar, applied uniformly across the batch")
cluster_ids_target_K1 = torch.zeros(2 * n, dtype=torch.long)
cluster_ids_kl_K1 = torch.zeros(n, dtype=torch.long)
loss_K1, metrics_K1 = cpo_unary_loss(
    policy_target, ref_target, policy_kl, ref_kl,
    status, cluster_ids_target_K1, cluster_ids_kl_K1,
    K=1, beta=0.1, desirable_weight=1.0, undesirable_weight=1.0,
)
print(f"   loss={loss_K1.item():.4f}")
print(f"   z_per_cluster={metrics_K1['z_per_cluster']}  (length 1)")
assert len(metrics_K1["z_per_cluster"]) == 1
# At K=1, z is exactly the mean of nonzero kl_rewards, clamped >= 0
kl_rewards_manual = (policy_kl - ref_kl).detach()
nonempty_manual = (kl_rewards_manual.abs() != 0).float()
z_manual = (kl_rewards_manual * nonempty_manual).sum() / nonempty_manual.sum().clamp(min=1)
z_manual = z_manual.clamp(min=0)
assert abs(metrics_K1["z_per_cluster"][0] - z_manual.item()) < 1e-3, (
    f"K=1 z mismatch: {metrics_K1['z_per_cluster'][0]} vs {z_manual.item():.6f}"
)
print(f"   ✓ K=1 z matches manual KTO formula\n")

# ---------------- T2: Loss finite and reasonable ---------------- #
print("T2: K=4 with normal inputs — loss should be finite, in (0, 2)")
loss, metrics = cpo_unary_loss(
    policy_target, ref_target, policy_kl, ref_kl,
    status, cluster_ids_target, cluster_ids_kl,
    K=K, beta=0.1, desirable_weight=1.0, undesirable_weight=1.0,
)
assert torch.isfinite(loss).all(), "loss is not finite"
assert 0 < loss.item() < 2.0, f"loss out of expected range: {loss.item()}"
print(f"   loss={loss.item():.4f}  margin={metrics['margin']:+.4f}")
print(f"   z_per_cluster={metrics['z_per_cluster']}")
print(f"   ✓ finite and in range\n")

# ---------------- T3: Per-cluster z responds to per-cluster KL signal ---------------- #
print("T3: inject large KL signal into cluster 2 only — z[2] should dominate")
policy_kl_t3 = ref_kl.clone()
policy_kl_t3[cluster_ids_kl == 2] += 5.0   # large positive log-ratio for cluster 2
loss_t3, metrics_t3 = cpo_unary_loss(
    policy_target, ref_target, policy_kl_t3, ref_kl,
    status, cluster_ids_target, cluster_ids_kl,
    K=K, beta=0.1, desirable_weight=1.0, undesirable_weight=1.0,
)
print(f"   z_per_cluster={metrics_t3['z_per_cluster']}")
z_per_k = metrics_t3["z_per_cluster"]
assert z_per_k[2] > 4.0, f"z[2] should be ~5.0, got {z_per_k[2]}"
other_zs = [z_per_k[k] for k in range(K) if k != 2]
assert max(other_zs) < 1.0, f"non-injected clusters should be small, got {other_zs}"
print(f"   ✓ z[2]={z_per_k[2]:.4f} dominates; others={other_zs}\n")

print("All three sanity tests PASSED.")
