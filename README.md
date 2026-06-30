# CPO integration for HALOs

Cluster-referenced Preference Optimization (CPO), unary regime — KTO-style
training with the global KL baseline replaced by a per-cluster baseline $z_k$.

Two parts:

- **`cpo_mini/`** — single-device prototype. Runs on a laptop (MPS) or a GPU
  box (CUDA), no HALOs/FSDP. **Start here.**
- **`train/`, `config/`, `scripts/`** — drop-in patch for a HALOs checkout
  (GPU cluster).

## Quick start

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python cpo_mini/cpo_mini.py --scheme single --K 1
```

## Docs

- [cpo_mini/README.md](cpo_mini/README.md) — the local prototype: smoke tests, models, tuning.
- [HALOS_INTEGRATION.md](HALOS_INTEGRATION.md) — how to wire `CPOTrainer` into HALOs for cluster runs.

## License

MIT — see [LICENSE](LICENSE).
