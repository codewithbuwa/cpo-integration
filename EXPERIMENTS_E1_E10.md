# CPO Stage 1 — Experiments E1–E10

Each card is independently runnable given its dependencies. All training runs use HALOs at 8B Llama-3-Instruct, UltraFeedback-armorm, $n_{\text{examples}} = 20{,}000$, microbatch 64, $\beta = 0.1$, FSDP on 4 GPUs unless noted. Per-run wall-clock $\approx 8$ h, i.e. 32 GPU-h.

## Sequencing

```
E1 (smoke) ──┬─→ E2 (random control)  ─┐
             ├─→ E3 (topic)            ├─→ E6 (K sweep) ─→ E7 (τ sweep) ─→ E8 (vs baselines) ─→ E9 (capabilities)
             ├─→ E4 (difficulty)       │                                                       └─→ E10 (dataset transfer)
             └─→ E5 (task-type)        ┘
```

E2–E5 run in parallel. E6 picks the winning scheme from E2–E5. E10 is conditional on E8 landing the headline.

## Budget summary

| Exp | Runs | GPU-h |
|---|---|---|
| E1 | 4 (small) | ~24 |
| E2–E5 | 8 (4 × 2 seeds) | 256 |
| E6 | 6 | 192 |
| E7 | 5 | 160 |
| E8 | 12 (4 × 3 seeds) | 384 |
| E9 | eval-only on E8 ckpts | ~48 |
| E10 | 6 (2 × 3 seeds) | 192 |
| **Total** | **41** | **~1256** |

Lean variant (drop E10, cut E8 to 2 seeds): ~870 GPU-h.

---

## E1 — Infrastructure equivalence

**Question.** Is the CPO scaffold structurally correct?

**Setup.** Four small sanity tests, not benchmark runs. Use `n_examples=200`, single seed.

**Configs.**
- E1a: `loss=cpo num_clusters=1 ema_tau=null sync_reference=false` vs `loss=kto`, same seed → per-step train loss matches to $< 10^{-5}$ for 100 steps.
- E1b: `n_examples=10 num_clusters=2`, alternating cluster IDs → loss $\to 0$ in $\leq 50$ steps.
- E1c: `num_clusters=8` with non-trivial sidecar JSON → $\geq 5$ of 8 `z_train/cluster_*` distinguishable in W&B (var $> 10^{-4}$).
- E1d: `ema_tau=1e-3 sync_reference=true` for 200 steps → no NaNs; $\|\theta_{\text{ref}} - \theta_0\|$ grows monotonically; grad norm within $2\times$ frozen baseline.

**Pass criterion.** All four must pass before proceeding.

**Dependencies.** None.

---

## E2 — Random clustering control

**Question.** Does the per-cluster $z_k$ mechanism give a gain from genuine structure, not from added capacity?

**Hypothesis.** Random cluster assignment yields $\Delta_{\text{LC}}({\text{CPO vs KTO}}) \approx 0$ within seed CI.

**Setup.** Assign each prompt to a uniformly random cluster in $[0, 8)$, fixed seed. Train CPO at $K = 8$, frozen reference. 2 seeds.

**Configs.** `loss=cpo num_clusters=8 cluster_map_path=random_K8.json ema_tau=null sync_reference=false`.

**Metric.** AlpacaEval 2.0 LC win-rate vs SFT; report $\Delta$ vs KTO baseline (use a KTO run from E8 or one dedicated KTO run on the same seeds).

**Decision.** This is a negative control. If $\Delta > 1$ pt on either seed, there is a confounder in the scaffold and **E6 is blocked** until diagnosed. Most likely causes: cluster IDs not reaching the trainer (verify `z_train/cluster_*` are all $\approx$ equal in W&B), or the data shuffling is reading the cluster sidecar in a leak-inducing way.

**Dependencies.** E1 pass.

---

## E3 — Topic clustering

**Question.** Does topic-based clustering give a positive $\Delta$ over KTO?

**Setup.** $K$-means ($K = 8$) on prompt embeddings from `BAAI/bge-large-en-v1.5`. Compute offline, store as `topic_K8.json`. 2 seeds.

**Configs.** `loss=cpo num_clusters=8 cluster_map_path=topic_K8.json ema_tau=null sync_reference=false`.

**Metrics.** AlpacaEval LC; per-cluster reward margin std$_k$; cluster size distribution (report imbalance).

**Decision.** Record $\Delta_{\text{LC}}$ for the cross-scheme comparison at the end of E5.

**Dependencies.** E1 pass. Pre-compute the sidecar JSON (one-time GPU job, ~30 min on 1 GPU).

---

## E4 — Difficulty clustering

**Question.** Does difficulty-based clustering give a positive $\Delta$?

**Setup.** Bin prompts into 8 quantiles by the gap between top-1 and median ArmoRM score (already in UltraFeedback-armorm). Store as `difficulty_K8.json`. 2 seeds.

**Configs.** `loss=cpo num_clusters=8 cluster_map_path=difficulty_K8.json ema_tau=null sync_reference=false`.

**Metrics.** Same as E3, plus: per-cluster mean ArmoRM gap (sanity — should be monotone in $k$).

**Decision.** Record $\Delta_{\text{LC}}$.

**Dependencies.** E1 pass.

---

## E5 — Task-type clustering

**Question.** Does dataset-metadata clustering give a positive $\Delta$?

**Setup.** Cluster by UltraFeedback's source-category metadata. If categories number > 8, take the 7 largest and pool the rest into cluster 7. Store as `tasktype_K8.json`. 2 seeds.

**Configs.** `loss=cpo num_clusters=8 cluster_map_path=tasktype_K8.json ema_tau=null sync_reference=false`.

**Metrics.** Same as E3.

**Decision rule for E2–E5 jointly.** Pick scheme $S^\star = \arg\max_S \Delta_{\text{LC}}$, **conditional on** the random control E2 giving $\Delta_{\text{LC}} \leq $ smallest seed-CI half-width across schemes. If E2 fails this, mechanism is confounded and the headline claim cannot be made from this protocol.

**Dependencies.** E1 pass.

---

## E6 — $K$ ablation

**Question.** What $K^\star$ maximizes alignment gain under scheme $S^\star$?

**Setup.** Sweep $K \in \{1, 4, 8, 32, 128\}$. Re-cluster $S^\star$ at each $K$ (re-run $k$-means or re-bin). 1 seed at $K \in \{1, 4, 32, 128\}$, 2 seeds at $K = 8$ (matches prior).

**Configs.** `loss=cpo num_clusters={K} cluster_map_path={S*}_K{K}.json ema_tau=null sync_reference=false`.

**Metrics.** AlpacaEval LC vs SFT; $\bar n_k$ (mean effective sample size per cluster); std$_k(z_k)$ at end of training.

**Decision.** $K^\star = \arg\max$ LC, ties broken toward smaller $K$. If LC is monotone up to $K = 128$, run E6.5 at $K = 512$.

**Dependencies.** E2–E5 complete; $S^\star$ identified.

---

## E7 — EMA $\tau$ ablation

**Question.** What $\tau^\star$ for the EMA reference?

**Setup.** Sweep $\tau \in \{0, 10^{-4}, 10^{-3}, 10^{-2}, 1\}$ at $(S^\star, K^\star)$. 1 seed each. $\tau = 0$ is the frozen reference (sync_reference=false); $\tau = 1$ is hard sync every step (HALOs default with sync_reference=true and ema_tau=null).

**Configs.** `loss=cpo num_clusters={K*} cluster_map_path={S*}_K{K*}.json sync_reference=true ema_tau={τ}` (with `ema_tau=null` for $\tau \in \{0, 1\}$ and `sync_reference` adjusted accordingly).

**Metrics.** AlpacaEval LC; $\|\theta_{\text{ref}} - \theta_0\|$ over training; mean $z_k$ over training (should decay toward 0 as $\tau$ grows).

**Decision.** $\tau^\star = $ smallest $\tau$ not significantly worse than the best. Prefer small $\tau$ on the principle that frozen-reference is the cleanest objective.

**Dependencies.** E6 complete.

---

## E8 — Main comparison vs baselines

**Question.** Does CPO with $(S^\star, K^\star, \tau^\star)$ beat KTO at matched compute?

**Setup.** Four methods, 3 seeds each. Identical data, compute, $\beta = 0.1$, lr, batch size.

**Configs.**
- `loss=sft` — capability floor
- `loss=dpo`
- `loss=kto desirable_weight=1.1`
- `loss=cpo` with $(S^\star, K^\star, \tau^\star)$

**Metrics.** AlpacaEval 2.0 LC win-rate vs GPT-4 turbo (mean $\pm$ SE across 3 seeds); held-out reward margin; for CPO only, std$_k(\text{margin})$ as a mechanism check.

**Decision.** Headline claim succeeds if CPO mean LC exceeds KTO mean LC by $\geq 1$ pt and the seed-CI gap is non-overlapping.

**Dependencies.** E7 complete.

---

## E9 — Capability preservation

**Question.** Does CPO degrade general capabilities relative to baselines?

**Setup.** Run `lm_eval_harness` on every E8 final checkpoint (4 methods × 3 seeds = 12 checkpoints). No new training.

**Tasks.** `arc_easy, arc_challenge, winogrande, bbh_cot_fewshot, gsm8k_cot` (already wired in the HALOs launch script).

**Metric.** Mean across tasks, plus per-task. Compare CPO vs KTO mean; bound the regression at $\leq 1$ pt average across tasks.

**Decision.** If CPO loses on capability while winning on AlpacaEval, the writeup must acknowledge the trade.

**Dependencies.** E8 final checkpoints saved.

---

## E10 — Dataset transfer (conditional)

**Question.** Does the CPO gain transfer to a second preference dataset?

**Setup.** Only if E8 lands the headline. Repeat E8 minimal slice (CPO vs KTO only) on **PKU-SafeRLHF** or **Anthropic HH**. 3 seeds. Use the same $(S^\star, K^\star, \tau^\star)$ from E6/E7 — do not re-tune. The clustering sidecar must be rebuilt for the new dataset's prompts.

**Configs.** Switch `train_datasets`/`test_datasets` in launch.py; rebuild cluster sidecar with the same scheme $S^\star$.

**Metrics.** AlpacaEval LC and dataset-native eval (e.g., safety win-rate for PKU-SafeRLHF).

**Decision.** Reports as either "transfers" (Δ in same direction, half magnitude or more) or "dataset-specific" (Δ flips sign or collapses). Either outcome is publishable; not transferring just bounds the claim.

**Dependencies.** E8 succeeded.

---

## What goes in the paper

If E2 passes (random control near zero), E5/E4/E3 best scheme wins by $\geq 1$ pt, and E8 lands:

- E1, E2 in the appendix as scaffold validation.
- E3/E4/E5 as Figure 1 (scheme comparison).
- E6 as Figure 2 ($K$ ablation).
- E7 as Figure 3 ($\tau$ ablation, narrow result).
- E8 as Table 1 (headline).
- E9 as Table 2 (capability preservation).
- E10 as Figure 4 or Section 5 (transfer).

The unary-only headline is the +11.4 synthetic finding replicated at LLM scale — if it survives E2, that's the paper.
