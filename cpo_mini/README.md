# cpo-mini

Minimal CPO trainer for M1 prototyping. One file (`cpo_mini.py`), single-device,
no FSDP/accelerate/Hydra. Loss math matches the HALOs `CPOTrainer` exactly;
the only differences are (a) no cross-device reduce because there's only one
device, and (b) trimmed bookkeeping.

Use this before burning cluster hours. Anything that goes wrong here would
also go wrong at LLM scale, just slower and more expensively.

## Install

```bash
# from the repo root, on Python 3.12
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # torch, transformers, datasets (pinned)
```

PyTorch will use the MPS backend automatically on M-series Macs (and CUDA on a
GPU box). fp32 throughout; bf16 has gaps on MPS.

## Default model and footprint

`HuggingFaceTB/SmolLM2-135M-Instruct` — 135M params, modern Llama-style architecture,
already instruction-tuned (so no SFT warmup needed for the chat-style UltraFeedback
data).

Memory at batch_size 4, max_length 512:

| | Approx |
|---|---|
| Policy params (fp32) | 540 MB |
| Reference params (fp32) | 540 MB |
| AdamW optimizer state | 1.08 GB |
| Activations | ~500 MB |
| **Total** | **~3 GB** |

Fits comfortably on 8 GB M1. On 16+ GB, bump to `SmolLM2-360M-Instruct` (`--model HuggingFaceTB/SmolLM2-360M-Instruct`).

## Three smoke tests, in order

These mirror E1–E5 from the LLM-scale protocol. Each takes 10–25 min on M1.

### S1. KTO equivalence (single cluster, K=1)

```bash
python cpo_mini.py --scheme single --K 1 --n_steps 200 --out_dir runs/s1_single
```

Expect: loss decreases from ~0.5 toward ~0.3. `margin` becomes positive. `z_std`
should be exactly 0 (only one cluster). This validates the unary CPO path works
at all, since with K=1 the per-cluster mechanism reduces to KTO.

### S2. Random clustering control (K=8)

```bash
python cpo_mini.py --scheme random --K 8 --n_steps 200 --out_dir runs/s2_random
```

Expect: loss and margin trajectory near-identical to S1. `z_std` will be nonzero
but small. **This is the load-bearing negative control.** If random clustering
visibly improves over S1, your scaffold has a confounder.

Compare end-state margin numerically:

```bash
python -c "
import json
for name in ['s1_single', 's2_random']:
    d = json.load(open(f'runs/{name}/log.json'))
    last = d['log'][-1]
    print(f'{name:12s}  loss={last[\"loss\"]:.4f}  margin={last[\"margin\"]:+.4f}')
"
```

Margin gap between S1 and S2 should be within seed noise.

### S3. Length-binned clustering (toy structured)

```bash
python cpo_mini.py --scheme length --K 8 --n_steps 200 --out_dir runs/s3_length
```

Expect: `z_per_cluster` to show non-trivial spread (`z_std > 0.01`), since long
prompts tend to be harder and have lower log-ratios. Margin may or may not
improve — length is a weak proxy for structure. The point of S3 is to validate
that the per-cluster mechanism *can* show signal when clusters reflect something
real, not to claim length is a good clustering scheme.

## EMA smoke test

```bash
python cpo_mini.py --scheme single --K 1 --n_steps 200 --ema_tau 1e-3 --out_dir runs/s4_ema
```

Expect: stable training, no NaNs, slightly different trajectory from S1.
Reference-policy distance grows monotonically.

## What to look at in the logs

Each run writes `runs/<name>/log.json` with per-step metrics. Quick plot:

```python
import json, matplotlib.pyplot as plt
runs = ["s1_single", "s2_random", "s3_length"]
for r in runs:
    log = json.load(open(f"runs/{r}/log.json"))["log"]
    plt.plot([x["step"] for x in log], [x["margin"] for x in log], label=r)
plt.xlabel("step"); plt.ylabel("reward margin"); plt.legend(); plt.show()
```

## What this is and isn't

Is:
- a faithful single-device CPO unary trainer
- the right place to debug the cluster-id plumbing and EMA hook
- the right place to confirm S2 (random-cluster control) behaves as theory predicts
- fast enough to iterate in minutes, not hours

Isn't:
- a benchmark setup — SmolLM2-135M is too small to read off AlpacaEval-relevant gains
- a paper result — for that you run the E1–E10 protocol on H100s with the HALOs scaffold
- a clustering tool — the schemes here are deliberately toy; build the real ones offline

If a smoke test passes on M1 but the analogous LLM-scale run fails, the bug is
most likely in HALOs' multi-device reduce or its data loader. If it fails here
first, the bug is in the loss or the cluster plumbing.

## Adding a new cluster scheme

In `assign_clusters`, add a branch. The function takes the example list and
mutates `ex.cluster_id` in place. Any deterministic function of the example
metadata or the prompt content works.

```python
elif scheme == "my_scheme":
    for ex in examples:
        ex.cluster_id = compute_cluster_id(ex)
```

For real schemes (BGE k-means, ArmoRM quantiles), compute the assignment
offline, save as a sidecar `{prompt_id: cluster_id}` JSON, and add a
`from_json` scheme that reads it. Same pattern as the HALOs `cpo_cluster.py`.

## Known M1 caveats

- First run is slow: `datasets` downloads UltraFeedback (all splits, ~hundreds of
  MB) and HF caches the SmolLM2 weights (~540 MB fp32). Subsequent runs hit the
  cache and start in seconds.
- `scatter_add_` on MPS is supported but occasionally falls back to CPU silently.
  If you see warnings, ignore them — correctness is unaffected.
- `torch.compile` is not currently reliable on MPS; this script doesn't use it.
- The default lr=5e-7 is conservative. SmolLM2-135M is small enough that 1e-6
  is usually fine, but 1e-5 risks divergence in the first 50 steps.
