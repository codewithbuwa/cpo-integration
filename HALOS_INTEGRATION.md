# HALOs integration

How to drop `CPOTrainer` into a HALOs checkout for GPU-cluster runs. Stage 2
(mixed $\alpha \in (0, 1)$ regime with the DPO pairwise term) is intentionally
deferred; this scaffold leaves the door open for it.

## Files

| Path | Purpose |
|---|---|
| `train/cpo_trainer.py` | `CPOTrainer(UnpairedPreferenceTrainer)` with per-cluster $z_k$ and EMA hook |
| `train/cpo_cluster.py` | Sidecar-JSON cluster loader (pluggable; replace later) |
| `config/loss/cpo.yaml` | Loss config; reproduces KTO when `num_clusters: 1` and `ema_tau: null` |
| `scripts/launch_llama_cpo.sh` | SLURM launch wrapper |

## What to drop into your HALOs checkout

Copy the four files into the matching locations in your HALOs repo:

```
HALOs/
├── train/
│   ├── cpo_trainer.py          ← new
│   └── cpo_cluster.py          ← new
├── config/loss/
│   └── cpo.yaml                ← new
└── scripts/
    └── launch_llama_cpo.sh     ← new
```

Then apply four small edits to existing files.

### Edit 1 — `train/trainers.py`: register `CPOTrainer`

Append one line at the bottom of the file. `launch.py` resolves the trainer via `getattr(trainers, config.loss.trainer)`, so `CPOTrainer` must be visible as an attribute of the `train.trainers` module.

```python
# At the very end of train/trainers.py
from .cpo_trainer import CPOTrainer       # noqa: F401  — registered for launch.py
```

### Edit 2 — `train/data.py`: add `cluster_id` to `Example`

In the `Example` dataclass (around line 38), add one field. Default `0` makes existing code paths a no-op (single-cluster ↔ KTO).

```python
@dataclass
class Example:
    prompt: List = field(default_factory=list)
    prompt_id: int = -1
    generations: List = field(default_factory=list)
    sft_index: int = -1
    scores: List[float] = field(default_factory=list)
    pairs: List[Tuple[int, int]] = field(default_factory=list)
    desirable: List[bool] = field(default_factory=list)
    dataset_name: str = ''
    original_prompt: str = ''
    cluster_id: int = 0                       # ← ADD THIS LINE
```

No `__setattr__` changes needed — `cluster_id` doesn't interact with the prompt-id hashing logic.

### Edit 3 — `train/dataloader.py`: assign clusters and carry them through the batch

Two changes in this file.

**(a)** At the top of `train/dataloader.py`, add the import:

```python
from .cpo_cluster import load_cluster_map, assign_cluster
```

**(b)** In `DataLoader.__init__`, immediately after the existing line `print(f"Total prompts loaded: {len(self.full_data)}")` (currently around line 105), assign cluster IDs to every loaded `Example`:

```python
        if process_index == 0:
            print(f"Total prompts loaded: {len(self.full_data)}")

        # ---- CPO cluster assignment ---- #
        # Loss configs that don't set these fields will get sensible defaults
        # (single cluster, all examples → cluster 0, equivalent to KTO).
        cluster_map_path = kwargs.get("cluster_map_path", None)
        num_clusters = kwargs.get("num_clusters", 1)
        cluster_map = load_cluster_map(cluster_map_path, num_clusters)
        for prompt_key, example in self.full_data.items():
            example.cluster_id = assign_cluster(example.prompt_id, cluster_map)

        self.num_training_steps = self.get_num_training_steps()
```

**(c)** In `UnpairedPreferenceDataLoader.__iter__`, in the loop that builds each `batch_element` (around line 427–432), add one line so `cluster_id` rides on the batch element and gets passed through `collate` as a plain python list:

```python
                batch_element = self.tokenize_batch_element(example.prompt, generation, prefix='target')
                batch_element['status'] = status
                batch_element['conversation'] = example.prompt
                batch_element['generation'] = generation
                batch_element['prompt_id'] = example.prompt_id
                batch_element['score'] = score
                batch_element['cluster_id'] = example.cluster_id    # ← ADD THIS LINE
                example_queue.append(batch_element)
```

`collate` already passes any non-tensor field through unchanged (line 152), so `batch['cluster_id']` arrives in the trainer as a python list of ints.

### Edit 4 — `launch.py`: forward loss-level kwargs to the dataloader

`launch.py` builds a `data_iterator_kwargs` dict (around line 137) and passes it to both train and eval data loaders. `config.loss.*` fields are not auto-forwarded, so add two lines:

```python
    data_iterator_kwargs = dict(
        process_index=accelerator.process_index,
        num_processes=accelerator.num_processes,
        max_length=config.model.max_length,
        max_prompt_length=config.model.max_prompt_length,
        seed=config.seed,
        frac_unique_desirable=config.frac_unique_desirable,
        frac_unique_undesirable=config.frac_unique_undesirable,
        control_tokens=config.loss.get("control_tokens", {}),
        cluster_map_path=config.loss.get("cluster_map_path", None),   # ← ADD
        num_clusters=config.loss.get("num_clusters", 1),               # ← ADD
    )
```

Both train and eval iterators pick these up via `**data_iterator_kwargs`.

## Configuration knobs

All in `config/loss/cpo.yaml`:

| Field | Meaning | Recommended |
|---|---|---|
| `beta` | sigmoid temperature | $0.1$ (same as KTO) |
| `num_clusters` | $K$ | start $K = 1$ for smoke test; then $\{8, 32, 128\}$ ablation |
| `cluster_map_path` | sidecar JSON | `null` until you decide the source |
| `desirable_weight`, `undesirable_weight` | class weights | `1.1, 1.0` (HALOs default for ultrafeedback) |
| `sync_reference` | fire EMA hook every step | `true` to enable EMA |
| `ema_tau` | EMA rate $\tau$ | start $10^{-3}$; ablate $\{10^{-4}, 10^{-3}, 10^{-2}\}$ |

## Smoke tests, in this order

These match the synthetic-experiment validation pattern from Part 2.

**Test 1 — $K = 1$ reproduces KTO bit-for-bit.** With a single cluster and `ema_tau: null`, every example's $z_k = z_0$ and the EMA branch is inactive. Loss should match KTOTrainer to numerical precision on identical seeds and batches.

```bash
# in config/loss/cpo.yaml: num_clusters: 1, sync_reference: false, ema_tau: null
# Run the same exp once with loss=kto and once with loss=cpo, identical seed
# Compare wandb logs / per-step loss values
```

**Test 2 — overfit 10 examples.** Standard sanity. Set `n_examples=10`, `num_clusters=2` (alternating), confirm loss → 0.

**Test 3 — per-cluster $z_k$ logging is non-degenerate.** Run with $K = 8$ and a meaningful `cluster_map_path`. Watch the `z_train/cluster_*` metrics in wandb; they should differ across clusters. If they're all identical, the cluster assignment isn't reaching the trainer (most likely culprit: `cluster_id` not in `batch_element`, see Edit 3c).

**Test 4 — EMA stability.** With `ema_tau: 0.001`, watch `loss/train` for unusual oscillation in the first few hundred steps. A reference drifting too fast destabilizes the unary term because $z_k$ collapses toward zero. If unstable, lower $\tau$ by an order of magnitude.

## Open items (Stage 2 and beyond)

1. **Cluster source.** Three options: sidecar JSON on prompt_id (currently scaffolded), dataset field (cleanest if you build the dataset), online embedding-based (couples training to an embedding model — last resort). Decide before $K = 8$ ablation.
2. **Mixed CPO ($\alpha > 0$).** Requires a hybrid dataloader yielding paired tensors and a unary view of the same pairs, plus a `MixedCPOTrainer(BasicTrainer)` running two forwards. The architecture sketch is in the planning notes — the unary path is exactly `CPOTrainer.loss` as written.
3. **Buffers in EMA.** Llama variants used in HALOs don't carry stateful buffers (no BatchNorm running stats; RMSNorm parameters are weights). If you switch to an architecture with running buffers, extend `sync_reference_with_policy` to iterate `.buffers()` as well.
4. **`max_prompt_count` interaction.** When the unary dataloader subsamples pairs via `max_prompt_count`, all subsampled examples inherit the same prompt-level `cluster_id`. This is the intended semantics (cluster is a property of the prompt) but worth confirming on your data.

## Why this is a thin patch

| Component | Reused as-is |
|---|---|
| `BasicTrainer.train` loop, gradient accumulation, FSDP plumbing | ✓ |
| `UnpairedPreferenceTrainer.forward` (overridden, but same signature shape as KTO) | structurally |
| `UnpairedPreferenceDataLoader` (one-line addition) | ✓ |
| KL-mismatched-sequence construction (lines 443–449) | ✓ |
| `get_sequence_rewards`, `accelerator.reduce`, `accelerator.gather` patterns | ✓ |
| Save / checkpoint / eval | ✓ |

The only structural delta is: a scalar `KL` becomes a length-$K$ vector $z$ indexed by `cluster_id`, and `sync_reference_with_policy` does an in-place EMA on local FSDP shards instead of a full state_dict copy.
