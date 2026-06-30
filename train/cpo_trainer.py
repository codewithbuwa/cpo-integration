# CPO Trainer for HALOs — Stage 1 (unary CPO with per-cluster z_k).
#
# Mounts on top of HALOs' UnpairedPreferenceTrainer. The only structural
# difference from KTOTrainer is that the global KL baseline z_0 is replaced
# by a per-cluster vector z_k of length K (num_clusters), with per-example
# baselines z[cluster_id(example)].
#
# Two design notes:
#   1. The KL term is aggregated per-cluster across all devices in a single
#      accelerator.reduce call, mirroring KTO's existing pattern (one reduce
#      per step, no extra collective comms).
#   2. The reference model is updated via in-place EMA on local FSDP shards
#      (no unwrap / re-shard), triggered by overriding sync_reference_with_policy
#      so the existing train-loop hook fires it. Set sync_reference: true
#      and ema_tau: <float> in the loss config to enable.

from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn

from .trainers import UnpairedPreferenceTrainer
from .utils import delete_dicts


class CPOTrainer(UnpairedPreferenceTrainer):
    """Cluster-referenced Preference Optimization, unary regime (alpha = 0).

    Loss (per example, with cluster k = cluster_id(x)):
        chosen:   L = w_d * (1 - sigmoid(beta * (r_chosen   - z_k)))
        rejected: L = w_u * (1 - sigmoid(beta * (z_k      - r_rejected)))

    where z_k is the running estimate of E_{(x,y') ~ p_data, x in cluster k} [r(x, y')],
    estimated each step by the KL-mismatched sequences (same construction as KTO).

    Required config.loss fields:
        beta:               float  — sigmoid temperature (e.g. 0.1)
        num_clusters:       int    — K. Use 1 to reproduce KTO exactly.
        desirable_weight:   float
        undesirable_weight: float

    Optional config.loss fields:
        ema_tau: float | None — EMA rate for reference model. Requires
                                sync_reference: true. None ⇒ fall back to
                                hard sync (HALOs default) when sync_reference
                                is true; ignored when sync_reference is false.

    Batch contract (added by the patched UnpairedPreferenceDataLoader):
        batch['cluster_id']: List[int] of length microbatch_size, values in [0, K).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        K = self.config.loss.num_clusters
        assert K >= 1, f"num_clusters must be >= 1, got {K}"

    @staticmethod
    def _status_indices(status, device=None):
        """Split a batch's status list into chosen/rejected index lists.

        Returns python lists by default (for direct tensor indexing); pass a
        device to get LongTensors instead.
        """
        chosen = [i for i, s in enumerate(status) if s == 'chosen']
        rejected = [i for i, s in enumerate(status) if s == 'rejected']
        if device is None:
            return chosen, rejected
        return (
            torch.tensor(chosen, device=device, dtype=torch.long),
            torch.tensor(rejected, device=device, dtype=torch.long),
        )

    # ------------------------------------------------------------------ #
    # Forward: identical to KTOTrainer.forward — three returns.          #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        model: nn.Module,
        batch: Dict[str, Union[List, torch.LongTensor]],
        use_cache: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        with self.accelerator.autocast():
            with torch.no_grad():
                if use_cache:
                    KL_logps = model(batch['KL_combined_input_ids']).to(
                        self.policy_dtype
                    ).to(self.accelerator.device)
                else:
                    KL_logits = model(
                        batch['KL_combined_input_ids'],
                        attention_mask=batch['KL_combined_attention_mask'],
                    ).logits.to(self.policy_dtype)
                    KL_logps = self.get_batch_logps(KL_logits, batch['KL_labels'])

            if use_cache:
                target_logps = model(batch['target_combined_input_ids']).to(
                    self.policy_dtype
                ).to(self.accelerator.device)
            else:
                target_logits = model(
                    batch['target_combined_input_ids'],
                    attention_mask=batch['target_combined_attention_mask'],
                ).logits.to(self.policy_dtype)
                target_logps = self.get_batch_logps(
                    target_logits, batch['target_labels']
                )

        assert target_logps.shape[0] == len(batch['status'])
        chosen_idx, rejected_idx = self._status_indices(batch['status'])
        return target_logps[chosen_idx, ...], target_logps[rejected_idx, ...], KL_logps

    # ------------------------------------------------------------------ #
    # Loss: per-cluster z_k.                                              #
    # ------------------------------------------------------------------ #
    def loss(
        self,
        batch: Dict,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        policy_KL_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_KL_logps: torch.FloatTensor,
        *args,
    ) -> Tuple[
        torch.FloatTensor, torch.FloatTensor, torch.FloatTensor,
        torch.FloatTensor, torch.FloatTensor, torch.FloatTensor,
    ]:
        device = self.accelerator.device
        dtype = self.policy_dtype
        K = self.config.loss.num_clusters
        beta = self.config.loss.beta

        # ---- sequence rewards ---- #
        def rewards_or_empty(policy_logps, reference_logps):
            """get_sequence_rewards, or empty tensors when this status is absent."""
            if policy_logps.shape[0] != 0:
                return self.get_sequence_rewards(policy_logps, reference_logps)
            return (
                torch.empty(0, dtype=dtype, device=device),
                torch.tensor([1.0], dtype=dtype, device=device),
            )

        chosen_rewards, chosen_unclamped = rewards_or_empty(
            policy_chosen_logps, reference_chosen_logps
        )
        rejected_rewards, rejected_unclamped = rewards_or_empty(
            policy_rejected_logps, reference_rejected_logps
        )

        KL_rewards, _ = self.get_sequence_rewards(
            policy_KL_logps.detach(), reference_KL_logps.detach()
        )

        # ---- cluster bookkeeping ---- #
        # batch['cluster_id'] is a python list of length microbatch_size
        cluster_ids_all = torch.as_tensor(
            batch['cluster_id'], device=device, dtype=torch.long
        )
        assert cluster_ids_all.shape[0] == KL_rewards.shape[0], (
            f"cluster_id len {cluster_ids_all.shape[0]} != KL_rewards len "
            f"{KL_rewards.shape[0]}; dataloader and trainer disagree on batch layout"
        )
        # Validate cluster range
        assert (cluster_ids_all >= 0).all() and (cluster_ids_all < K).all(), (
            f"cluster_id values outside [0, {K}); got min={cluster_ids_all.min().item()}, "
            f"max={cluster_ids_all.max().item()}"
        )

        chosen_idx_t, rejected_idx_t = self._status_indices(batch['status'], device=device)
        cluster_ids_chosen = cluster_ids_all[chosen_idx_t]
        cluster_ids_rejected = cluster_ids_all[rejected_idx_t]

        # ---- per-cluster KL aggregation (scatter-add then cross-device reduce) ---- #
        nonempty = (KL_rewards.abs() != 0).to(dtype)            # mask out padding-only sequences
        local_sum = torch.zeros(K, dtype=dtype, device=device)
        local_count = torch.zeros(K, dtype=dtype, device=device)
        local_sum.scatter_add_(0, cluster_ids_all, KL_rewards * nonempty)
        local_count.scatter_add_(0, cluster_ids_all, nonempty)

        # Single collective; same call site as KTO's existing stats reduce.
        cluster_stats = self.accelerator.reduce(
            torch.stack([local_sum, local_count], dim=0), reduction="sum"
        )
        z = (cluster_stats[0] / cluster_stats[1].clamp(min=1)).clamp(min=0)   # [K]

        z_chosen = z[cluster_ids_chosen]
        z_rejected = z[cluster_ids_rejected]

        # ---- per-example unary loss ---- #
        # Both sides are weight * (1 - sigmoid(beta * (a - b))), differing only
        # in which term is the reward and which is the cluster baseline z.
        def unary_loss(a, b, weight):
            if a.shape[0] == 0:
                return torch.empty(0, dtype=dtype, device=device)
            return weight * (1 - torch.sigmoid(beta * (a - b)))

        chosen_losses = unary_loss(
            chosen_rewards, z_chosen, self.config.loss.desirable_weight
        )
        rejected_losses = unary_loss(
            z_rejected, rejected_rewards, self.config.loss.undesirable_weight
        )

        losses = torch.cat((chosen_losses, rejected_losses), 0)

        # Return z as the per-cluster diagnostic (KTO returned a scalar KL here).
        return (
            losses,
            chosen_rewards.detach(),
            rejected_rewards.detach(),
            z.detach(),
            chosen_unclamped,
            rejected_unclamped,
        )

    # ------------------------------------------------------------------ #
    # Batch metrics: same shape as KTOTrainer.get_batch_metrics, plus    #
    # per-cluster z logging.                                              #
    # ------------------------------------------------------------------ #
    def get_batch_metrics(
        self, batch: Dict[str, Union[List, torch.LongTensor]], mode: str = 'train'
    ):
        metrics = {}

        policy_chosen_logps, policy_rejected_logps, policy_KL_logps = self.forward(
            self.policy, batch
        )
        with torch.no_grad():
            reference_chosen_logps, reference_rejected_logps, reference_KL_logps = (
                self.forward(
                    self.reference_model,
                    batch,
                    use_cache=self.config.cache_reference_logprobs,
                )
            )

        losses, chosen_rewards, rejected_rewards, z, chosen_unclamped, rejected_unclamped = (
            self.loss(
                batch,
                policy_chosen_logps, policy_rejected_logps, policy_KL_logps,
                reference_chosen_logps, reference_rejected_logps, reference_KL_logps,
            )
        )

        combined_rewards = torch.cat(
            (chosen_rewards.detach(), rejected_rewards.detach()), 0
        )
        combined_statuses = torch.tensor(
            [1] * len(chosen_rewards) + [0] * len(rejected_rewards),
            device=self.accelerator.device,
        )

        stats = self.accelerator.gather({
            'rewards': combined_rewards,
            'statuses': combined_statuses,
            'losses': losses.detach(),
        })

        all_rewards = stats['rewards']
        chosen_idx_g = [i for i, s in enumerate(stats['statuses']) if s.item() == 1]
        rejected_idx_g = [i for i, s in enumerate(stats['statuses']) if s.item() == 0]

        metrics[f'rewards_{mode}/chosen'] = all_rewards[chosen_idx_g]
        metrics[f'rewards_{mode}/rejected'] = all_rewards[rejected_idx_g]
        metrics[f'rewards_{mode}/margins'] = torch.tensor([
            (all_rewards[chosen_idx_g].mean().nan_to_num(0)
             - all_rewards[rejected_idx_g].mean().nan_to_num(0)).item()
        ])
        metrics[f'loss/{mode}'] = stats['losses'].mean()
        metrics[f'rewards_{mode}/chosen_unclamped'] = self.accelerator.gather(chosen_unclamped)
        metrics[f'rewards_{mode}/rejected_unclamped'] = self.accelerator.gather(rejected_unclamped)

        # z is already reduced across devices inside .loss(); log it once.
        # Mean as a scalar headline; per-cluster as a small dict-like log.
        metrics[f'z_{mode}/mean'] = z.mean()
        for k in range(z.shape[0]):
            metrics[f'z_{mode}/cluster_{k}'] = z[k]

        del (policy_chosen_logps, policy_rejected_logps, policy_KL_logps,
             reference_chosen_logps, reference_rejected_logps, reference_KL_logps,
             combined_rewards, combined_statuses, all_rewards)
        delete_dicts(stats)

        return losses.mean(), metrics

    # ------------------------------------------------------------------ #
    # EMA reference update.                                               #
    # Overrides BasicTrainer.sync_reference_with_policy so the existing  #
    # train-loop hook (fired when config.loss.sync_reference is true)    #
    # does an in-place EMA instead of a full state_dict copy.            #
    # ------------------------------------------------------------------ #
    def sync_reference_with_policy(self):
        tau = self.config.loss.get('ema_tau', None)
        if tau is None:
            # No EMA configured — fall back to hard sync (HALOs default behavior).
            return super().sync_reference_with_policy()

        assert 0.0 < tau <= 1.0, f"ema_tau must be in (0, 1], got {tau}"

        # In-place EMA on local (sharded) parameters. Under FSDP, policy and
        # reference are sharded identically (same arch + same prepare call),
        # so zip over .parameters() lines up shard-for-shard without any
        # collective comm. Buffers (e.g. RMSNorm running stats — none in
        # current Llama configs) would need a separate pass if added later.
        with torch.no_grad():
            for p_pol, p_ref in zip(
                self.policy.parameters(), self.reference_model.parameters()
            ):
                if p_pol.shape != p_ref.shape:
                    raise RuntimeError(
                        f"EMA shape mismatch: policy {p_pol.shape} vs reference "
                        f"{p_ref.shape}. Reference and policy must be prepared identically."
                    )
                # p_ref ← (1 - tau) * p_ref + tau * p_pol
                p_ref.data.mul_(1.0 - tau).add_(p_pol.data, alpha=tau)

        self.accelerator.wait_for_everyone()
