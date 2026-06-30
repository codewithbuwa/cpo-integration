#!/usr/bin/env python3
"""CPO-mini — minimal unary CPO trainer for M1 prototyping.

Single-device implementation. The loss math is bit-equivalent to HALOs
CPOTrainer.loss when num_clusters and cluster assignment are matched.
No FSDP, no accelerate, no Hydra — meant for fast local iteration on a
Mac before committing GPU-cluster time.

Default config: SmolLM2-135M-Instruct, 500 UltraFeedback examples, K=8,
~10-25 min on an M1 (8GB or higher).

Usage:
    python cpo_mini.py --scheme single --K 1       # KTO-equivalent smoke test
    python cpo_mini.py --scheme random --K 8       # negative control
    python cpo_mini.py --scheme length --K 8       # toy structured clustering
"""

import argparse
import hashlib
import json
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from cpo_loss import cpo_unary_loss


# ----------------------- Device ----------------------- #

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ----------------------- Data ----------------------- #

@dataclass
class Example:
    prompt: str
    chosen: str
    rejected: str
    cluster_id: int = 0
    prompt_id: str = ""


def load_ultrafeedback_subset(n: int, seed: int) -> List[Example]:
    """Load a subset of UltraFeedback-binarized. Caches to ~/.cache/huggingface."""
    from datasets import load_dataset
    ds = load_dataset(
        "HuggingFaceH4/ultrafeedback_binarized",
        split="train_prefs",
    )
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    examples = []
    for row in ds:
        # The 'chosen' and 'rejected' fields are lists of chat turns;
        # take the final assistant turn as the completion.
        chosen = row["chosen"][-1]["content"]
        rejected = row["rejected"][-1]["content"]
        examples.append(
            Example(
                prompt=row["prompt"],
                chosen=chosen,
                rejected=rejected,
                # Stable across processes/runs (Python's hash() is salted per
                # process). This must match the prompt_id digest used by the
                # real HALOs dataloader, since it is the join key into
                # cluster_map (see train/cpo_cluster.py).
                prompt_id=hashlib.sha1(row["prompt"].encode("utf-8")).hexdigest()[:8],
            )
        )
    return examples


def assign_clusters(examples: List[Example], scheme: str, K: int, seed: int) -> None:
    """In-place cluster assignment. Schemes: single | random | length."""
    if scheme == "single":
        assert K == 1, "scheme='single' requires K=1"
        for ex in examples:
            ex.cluster_id = 0

    elif scheme == "random":
        rng = random.Random(seed + 1)
        for ex in examples:
            ex.cluster_id = rng.randrange(K)

    elif scheme == "length":
        # Bin prompts into K quantiles by character length.
        # Crude proxy for "difficulty" — meant for local sanity, not science.
        order = sorted(range(len(examples)), key=lambda i: len(examples[i].prompt))
        for rank, i in enumerate(order):
            examples[i].cluster_id = (rank * K) // len(examples)

    else:
        raise ValueError(f"unknown cluster scheme: {scheme}")


# ----------------------- Tokenization ----------------------- #

def tokenize_pair(
    prompt: str, completion: str, tokenizer, max_length: int
) -> Dict[str, List[int]]:
    """Tokenize prompt+completion as a single sequence; label only completion tokens."""
    p_ids = tokenizer.encode(prompt, add_special_tokens=False)
    eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    c_ids = tokenizer.encode(completion, add_special_tokens=False) + eos
    full = p_ids + c_ids
    labels = [-100] * len(p_ids) + c_ids[:]
    if len(full) > max_length:
        full = full[:max_length]
        labels = labels[:max_length]
    return {
        "input_ids": full,
        "attention_mask": [1] * len(full),
        "labels": labels,
    }


def collate(rows: List[Dict[str, List[int]]], pad_id: int) -> Dict[str, torch.Tensor]:
    max_len = max(len(r["input_ids"]) for r in rows)

    def pad(seq, val):
        return seq + [val] * (max_len - len(seq))

    return {
        "input_ids": torch.tensor([pad(r["input_ids"], pad_id) for r in rows], dtype=torch.long),
        "attention_mask": torch.tensor([pad(r["attention_mask"], 0) for r in rows], dtype=torch.long),
        "labels": torch.tensor([pad(r["labels"], -100) for r in rows], dtype=torch.long),
    }


def build_batch(
    examples: List[Example], tokenizer, max_length: int, rng: random.Random
) -> Dict:
    """Two tokenized views per microbatch:
       * target view: each example contributes its chosen and its rejected as unary items
       * KL view: prompts paired with a permuted set of completions (mismatched x, y')
    """
    prompts = [ex.prompt for ex in examples]
    chosen = [ex.chosen for ex in examples]
    rejected = [ex.rejected for ex in examples]
    cluster_ids = [ex.cluster_id for ex in examples]
    n = len(examples)

    # Target view: chosen items first, then rejected
    target_rows = [
        tokenize_pair(prompts[i], chosen[i], tokenizer, max_length) for i in range(n)
    ] + [
        tokenize_pair(prompts[i], rejected[i], tokenizer, max_length) for i in range(n)
    ]
    target_cluster_ids = cluster_ids + cluster_ids
    status = ["chosen"] * n + ["rejected"] * n

    # KL view: prompts in original order, completions permuted across the batch
    perm = list(range(n))
    rng.shuffle(perm)
    kl_completions = [chosen[p] for p in perm]   # always use chosen completions
    kl_rows = [
        tokenize_pair(prompts[i], kl_completions[i], tokenizer, max_length)
        for i in range(n)
    ]
    kl_cluster_ids = cluster_ids   # cluster follows prompt, not completion

    pad_id = tokenizer.pad_token_id
    target = collate(target_rows, pad_id)
    kl = collate(kl_rows, pad_id)

    return {
        "target_input_ids": target["input_ids"],
        "target_attention_mask": target["attention_mask"],
        "target_labels": target["labels"],
        "kl_input_ids": kl["input_ids"],
        "kl_attention_mask": kl["attention_mask"],
        "kl_labels": kl["labels"],
        "status": status,
        "target_cluster_ids": torch.tensor(target_cluster_ids, dtype=torch.long),
        "kl_cluster_ids": torch.tensor(kl_cluster_ids, dtype=torch.long),
    }


# ----------------------- Log-probs ----------------------- #

def batch_logps(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Sum log p(label_t | x_<t) over t where label_t != -100, per example. Returns shape [B]."""
    # Standard shift: logits at position t predict token at t+1
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:].clone()
    mask = (shift_labels != -100).float()
    shift_labels[shift_labels == -100] = 0   # safe-gather placeholder
    per_tok_logps = torch.gather(
        F.log_softmax(shift_logits, dim=-1),
        dim=-1,
        index=shift_labels.unsqueeze(-1),
    ).squeeze(-1)
    return (per_tok_logps * mask).sum(dim=-1)


def forward_view(model, input_ids, attention_mask, labels) -> torch.Tensor:
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    return batch_logps(out.logits, labels)


# ----------------------- Training loop ----------------------- #

def train(args):
    device = get_device()
    print(f"[setup] device={device}  model={args.model}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Models — fp32 on MPS for stability; bf16 has gaps on M1
    dtype = torch.float32
    print("[setup] loading policy ...")
    policy = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    print("[setup] loading reference ...")
    reference = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    for p in reference.parameters():
        p.requires_grad_(False)
    reference.eval()

    # Data
    print(f"[setup] loading {args.n_examples} examples from UltraFeedback ...")
    examples = load_ultrafeedback_subset(n=args.n_examples, seed=args.seed)
    assign_clusters(examples, args.scheme, args.K, seed=args.seed)
    sizes = Counter(e.cluster_id for e in examples)
    print(f"[setup] scheme={args.scheme}  K={args.K}  cluster sizes={dict(sorted(sizes.items()))}")

    # Optimizer
    optim = torch.optim.AdamW(policy.parameters(), lr=args.lr)

    # Training loop
    policy.train()
    rng = random.Random(args.seed + 7)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = []
    t0 = time.time()

    for step in range(args.n_steps):
        # Microbatch
        batch_examples = rng.sample(examples, args.batch_size)
        batch = build_batch(batch_examples, tokenizer, args.max_length, rng)
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device)

        # Policy forwards
        policy_target_logps = forward_view(
            policy, batch["target_input_ids"], batch["target_attention_mask"], batch["target_labels"]
        )
        with torch.no_grad():
            policy_kl_logps = forward_view(
                policy, batch["kl_input_ids"], batch["kl_attention_mask"], batch["kl_labels"]
            )

        # Reference forwards (always no_grad)
        with torch.no_grad():
            ref_target_logps = forward_view(
                reference, batch["target_input_ids"], batch["target_attention_mask"], batch["target_labels"]
            )
            ref_kl_logps = forward_view(
                reference, batch["kl_input_ids"], batch["kl_attention_mask"], batch["kl_labels"]
            )

        loss, metrics = cpo_unary_loss(
            policy_target_logps, ref_target_logps,
            policy_kl_logps, ref_kl_logps,
            batch["status"], batch["target_cluster_ids"], batch["kl_cluster_ids"],
            K=args.K, beta=args.beta,
            desirable_weight=args.desirable_weight,
            undesirable_weight=args.undesirable_weight,
        )

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optim.step()

        # Optional EMA reference update
        if args.ema_tau is not None and args.ema_tau > 0:
            with torch.no_grad():
                for p_pol, p_ref in zip(policy.parameters(), reference.parameters()):
                    p_ref.data.mul_(1.0 - args.ema_tau).add_(p_pol.data, alpha=args.ema_tau)

        log.append({"step": step, **metrics})
        if step % args.log_every == 0 or step == args.n_steps - 1:
            t = time.time() - t0
            z_head = metrics["z_per_cluster"][: min(4, args.K)]
            print(
                f"step {step:4d} | loss {metrics['loss']:.4f} | "
                f"margin {metrics['margin']:+.4f} | "
                f"z_mean {metrics['z_mean']:.4f} | z_std {metrics['z_std']:.4f} | "
                f"z[:4]={z_head} | t={t:.0f}s"
            )

    # Save log
    with open(out_dir / "log.json", "w") as f:
        json.dump({"args": vars(args), "log": log}, f, indent=2)
    print(f"[done] total {time.time() - t0:.0f}s  log → {out_dir / 'log.json'}")


# ----------------------- CLI ----------------------- #

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct",
                   help="Any HF causal LM; defaults to SmolLM2-135M-Instruct (~540MB).")
    p.add_argument("--n_examples", type=int, default=500)
    p.add_argument("--n_steps", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=4,
                   help="Effective per-step pairs. Two views ⇒ ~3× memory of batch_size alone.")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--lr", type=float, default=5e-7)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--scheme", default="random", choices=["single", "random", "length"])
    p.add_argument("--desirable_weight", type=float, default=1.0)
    p.add_argument("--undesirable_weight", type=float, default=1.0)
    p.add_argument("--ema_tau", type=float, default=None,
                   help="EMA rate τ for reference; None = frozen reference.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default="runs/cpo_mini")
    p.add_argument("--log_every", type=int, default=10)
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.scheme == "single":
        args.K = 1
    train(args)


if __name__ == "__main__":
    main()
