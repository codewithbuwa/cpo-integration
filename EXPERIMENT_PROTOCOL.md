# CPO Stage 1 — LLM-scale experiment protocol

Scope: unary CPO ($\alpha = 0$) on 8B Llama with HALOs.
Stage 2 (mixed $\alpha > 0$) is out of scope and tracked separately.

## Goals

In priority order:

1. **Headline claim.** Establish whether the +11.4 unary-only gain from CPO Part 2 synthetic transfers to LLM scale. Falsifiable: $\geq 1$ pt LC win-rate gain over KTO at matched compute, with non-overlapping 2-seed CIs.
2. **Pin the cluster source.** Decide between topic / difficulty / task-type / random clustering with controlled ablation.
3. **Pin $K$.** Find the operating point in $K \in \{1, 4, 8, 32, 128\}$ before any cross-scheme comparison.
4. **Pin $\tau$ for the EMA reference.** Find a stable range for $\tau$ at LLM scale — not validated by the synthetic experiments.

## Non-goals (do not run at this stage)

- Mixed-regime ablations ($\alpha$ sweep). Settled by synthetic at $\alpha^\star = 0.5$ and architecturally deferred to Stage 2.
- Pair-budget sweeps. Same reason.
- 30-seed replication. Compute-prohibitive at LLM scale; 2–3 seeds is the working norm.
- Per-cluster $\beta_k$ ablation. Synthetic kept $\beta$ global; LLM-scale extends this.

## Compute budget

Single run assumption: 8B Llama, 4×H100 (or A100 80GB), FSDP via HALOs' `fsdp_4gpu.yaml`, 20k UltraFeedback-armorm examples, batch size 64.

| Phase | Runs | Per-run wall-clock | GPU-hours |
|---|---|---|---|
| 0. Infra smoke | 3 | 1 h (n_examples=200) | 12 |
| 1. Clustering scheme | 8 (4 × 2 seeds) | 8 h | 256 |
| 2. $K$ ablation | 6 (5 K × 1 seed; 2 seeds at $K{=}8$) | 8 h | 192 |
| 3. $\tau$ ablation | 4 | 8 h | 128 |
| 4. Main comparison | 12 (4 methods × 3 seeds) | 8 h | 384 |
| **Total** | **33** | — | **~970 GPU-h** |

On 4×H100 that is ≈10 calendar days if everything runs serially. Realistic with two SLURM job-arrays in parallel: ~5 days.

Lean variant (cuts Phase 4 to 2 seeds, Phase 1 to 4 runs): ~600 GPU-h, ~3 days parallel.

## Phase 0 — Infrastructure smoke tests

**Purpose.** Verify the CPO scaffold is wired correctly before any real run. These are sanity checks, not experiments; no metrics reported in the paper.

| Test | Config | Pass criterion |
|---|---|---|
| 0.1 KTO equivalence | `loss=cpo num_clusters=1 ema_tau=null sync_reference=false`, identical seed to KTO baseline | per-step train loss matches `loss=kto` to $< 10^{-5}$ for 100 steps |
| 0.2 Overfit-10 | `n_examples=10 num_clusters=2`, alternating cluster IDs | train loss $\to 0$ within 50 steps |
| 0.3 Cluster diversity | $K = 8$ + any non-trivial cluster map, log `z_train/cluster_*` | $\geq 5$ of 8 $z_k$ values distinguishable in TensorBoard (variance $> 10^{-4}$) |
| 0.4 EMA stability | `ema_tau=1e-3 sync_reference=true` for 200 steps | no NaNs, gradient norm stays within $2\times$ of frozen-reference run |

**Go/no-go.** All four must pass before Phase 1. If 0.1 fails the trainer has a structural bug; if 0.3 fails the cluster pipeline isn't end-to-end.

## Phase 1 — Clustering scheme ablation

**Hypothesis.** The per-cluster $z_k$ mechanism extracts gain from genuine structure in the prompt distribution, not from added capacity. Random clustering should give $\approx 0$ improvement over KTO; structured clustering should give a measurable improvement.

**Schemes to compare** (all at $K = 8$, $\tau$ frozen, identical training budget):

| Scheme | How clusters are assigned |
|---|---|
| **Topic** | $K$-means on prompt embeddings from a sentence encoder (`BAAI/bge-large-en-v1.5` or `sentence-transformers/all-mpnet-base-v2`); pre-computed offline, stored as sidecar JSON |
| **Difficulty** | Bin prompts into 8 quantiles by the gap between top-1 and median scores under an external reward model (e.g., the ArmoRM scores already in UltraFeedback-armorm) |
| **Task-type** | If UltraFeedback metadata exposes source dataset / category, use those directly (8 most common categories, "other" rolled up) |
| **Random** (control) | Uniform random assignment; same seed for all examples |

**Configs.** Each scheme × 2 seeds = 8 runs. Reference frozen. `ema_tau: null`. `num_clusters: 8`. Train budget: 20k examples.

**Reported metrics.**

| Metric | Why |
|---|---|
| AlpacaEval 2.0 LC win-rate vs SFT baseline | Headline alignment metric |
| $\Delta$ vs KTO baseline (same seed, same model) | Isolates the cluster mechanism |
| Per-cluster reward margin at end of training | Diagnostic — does the mechanism actually equalize across clusters? |
| Per-cluster $z_k$ trajectory | Sanity — random scheme should show $z_k$ all $\approx$ equal; structured schemes should not |

**Decision rule.** Pick the scheme with the largest $\Delta$ vs KTO that is also non-overlapping with the random-clustering control on 2-seed CIs. If random clustering is non-zero, the scaffold has a confounder and Phase 2 is blocked until it's diagnosed.

**Expected falsifier.** If all four schemes (including random) give the same $\Delta$, the per-cluster $z_k$ mechanism isn't doing what the synthetic analysis predicts. This would contradict the +11.4 pt finding and is the most informative possible outcome — it would mean the synthetic story doesn't transfer.

## Phase 2 — $K$ ablation

**Hypothesis.** There exists an interior optimum $K^\star$. Too few clusters under-resolves the structure ($z_k \to z_0$); too many makes each $z_k$ noisy ($n_k$ shrinks like $N / K$).

Sweep $K \in \{1, 4, 8, 32, 128\}$ with the winning scheme from Phase 1. One seed for $K \in \{1, 4, 32, 128\}$, two seeds for $K = 8$ (the prior).

**Reported metrics.**

| Metric | Notes |
|---|---|
| AlpacaEval LC win-rate vs SFT | Headline |
| End-of-training std$_k$($z_k$) | Diagnostic for whether clusters separate |
| Mean per-cluster effective sample size $\bar n_k$ | Plot against $K$; expect $1/K$ falloff |
| Per-cluster margin spread, std$_k$(margin) | Tests the equalization hypothesis directly |

**Decision rule.** Pick $K^\star = \arg\max_K$ LC win-rate, breaking ties toward smaller $K$ (Occam). Use $K^\star$ in all downstream runs.

**Expected falsifier.** Monotone improvement up to $K = 128$ with no plateau would indicate the variance argument is wrong at this scale — possible if $N = 20\text{k}$ is enough that even $\bar n_k = 156$ is well-resolved. In that case re-run at $K = 512$ to find the actual ceiling.

## Phase 3 — EMA $\tau$ ablation

**Hypothesis.** A slowly-tracking reference ($\tau \in [10^{-4}, 10^{-3}]$) tightens the unary objective without destabilizing it. Faster updates collapse $z_k$ toward zero; frozen reference is the synthetic-experiment regime.

Sweep $\tau \in \{0\text{ (frozen)},\ 10^{-4},\ 10^{-3},\ 10^{-2},\ 1.0\text{ (hard sync)}\}$ at $K^\star$ from Phase 2. One seed each.

**Reported metrics.**

| Metric | Notes |
|---|---|
| AlpacaEval LC win-rate vs SFT | Headline |
| $\|\theta_{\text{ref}} - \theta_0\|$ over training | How fast the reference is drifting |
| Mean $z_k$ over training | Should decay toward 0 as $\tau$ grows |
| Loss curve smoothness | Subjective check for instability |

**Decision rule.** Pick the smallest $\tau$ that does not lose vs $\tau = 0$, preferring small $\tau$ on the principle that frozen-reference is the cleanest objective and EMA is a regularizer.

## Phase 4 — Main comparison

**Setup.** Run CPO with $(\text{scheme}^\star, K^\star, \tau^\star)$ from Phases 1–3, against three baselines, three seeds each. All methods on identical data (UltraFeedback-armorm, 20k examples), identical compute, identical hyperparameters where shared ($\beta = 0.1$, lr, batch size 64, etc.).

| Method | Config |
|---|---|
| SFT (no preference tuning) | `loss=sft` |
| DPO | `loss=dpo` |
| KTO | `loss=kto desirable_weight=1.1` |
| CPO | `loss=cpo` with phase-1/2/3 winners |

**Reported metrics.**

| Metric | Source |
|---|---|
| AlpacaEval 2.0 LC win-rate vs GPT-4 turbo | Primary alignment headline |
| lm-eval-harness: ARC, BBH-CoT-fewshot, GSM8K, WinoGrande | Capability headline (already in launch script) |
| Mean reward margin on held-out UltraFeedback test | Intrinsic |
| Per-cluster margin uniformity (std$_k$) — CPO only | CPO-specific mechanism check |

Report means $\pm$ standard error across 3 seeds. State whether CPO's CI overlaps each baseline.

**Headline question.** Does CPO beat KTO at matched compute, and is the gap meaningfully larger than KTO-vs-DPO?

## Metrics & logging

Per-step logging in W&B for all phases:

- `loss/train`, `loss/eval`
- `rewards_train/chosen`, `rewards_train/rejected`, `rewards_train/margins`
- `z_train/mean`, `z_train/cluster_0` … `z_train/cluster_{K-1}` (CPO only)
- `grad_norm`, `counters/examples`
- `rewards_train/chosen_unclamped`, `rewards_train/rejected_unclamped`

Eval-time:

- AlpacaEval 2.0 outputs go to `outputs/{exp_name}.json`; judge with `alpaca_eval_gpt-4.1.yaml` (HALOs default) for parity with the existing launch script.
- lm-eval-harness on the final checkpoint.

Per-cluster diagnostics (CPO only, computed on held-out):

- Mean and std of $\hat r(x, y_w) - \hat r(x, y_l)$ stratified by $k$.
- $z_k$ value at end of training, plotted alongside per-cluster prompt count.

## Risk log

| Risk | Likelihood | Mitigation |
|---|---|---|
| Random clustering control is non-zero | Medium | Diagnose data leakage between cluster assignment and dataset shuffle; check seed plumbing |
| All cluster schemes give same gain | Low | Would falsify mechanism; treat as headline result regardless |
| EMA with $\tau \geq 10^{-2}$ NaNs | Medium | Cap at $10^{-3}$; document; this is why Phase 3 is conservative |
| AlpacaEval judge variance dominates seed variance | High | Use the same judge config across all comparisons; run 3 seeds; note this is the standard regime |
| $K = 128$ is starved at $N = 20\text{k}$ | High | Acknowledge in $K$ ablation; consider Phase 2.5 at 64k for $K \geq 32$ if budget allows |
| FSDP EMA breaks under bf16 numerical drift | Low | Already lerp in float; verify reference checkpoint after 1k steps in Phase 0 |
| Cluster sidecar JSON missing for eval split | Medium | Default unmapped prompts to cluster 0; log fraction unmapped; should be < 1% for serious schemes |

## Schedule sketch

| Week | Activity |
|---|---|
| 1 | Phase 0 + start Phase 1 (build cluster sidecars for all 4 schemes) |
| 2 | Phase 1 complete, decision; start Phase 2 |
| 3 | Phase 2 complete, decision; start Phase 3 + Phase 4 in parallel |
| 4 | Phase 4 complete, write-up |

Tondji as rehearsal audience for the Phase 1 decision and the Phase 4 results.

## What this protocol does not test

- Robustness across base models (only 8B Llama-3-Instruct).
- Robustness across datasets (only UltraFeedback-armorm).
- Long-form generation quality beyond AlpacaEval.
- Safety / refusal behavior.

These belong in a follow-up protocol if Phase 4 lands the headline result.
