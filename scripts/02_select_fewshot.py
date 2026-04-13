#!/usr/bin/env python
"""
Step 2: build sparse activation features from the JSONL produced by step 1
and run the NEUFS dual-criteria selector.

Writes a JSON file containing the selected few-shot records (a subset of the
original unlabeled pool) in the order they were chosen.
"""

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
from transformers import AutoConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neufs.features import build_features, load_neuron_jsonl
from neufs.select import neufs_select


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_pool(path: str):
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            return [json.loads(l) for l in f if l.strip()]
        return json.load(f)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True,
                   help="Same HF model id used in step 1 (needed for num_layers / hidden_size)")
    p.add_argument("--neuron_jsonl", required=True,
                   help="Output of scripts/01_collect_activations.py")
    p.add_argument("--pool_path", required=True,
                   help="Original unlabeled pool (matches the order used in step 1)")
    p.add_argument("--output_path", required=True,
                   help="JSON file to write the selected few-shot set")
    p.add_argument("--n_shots", type=int, required=True)
    p.add_argument("--tau", type=float, default=0.5,
                   help="Weight on neuron consensus Q(x). tau=0 => pure diversity, tau=1 => pure consensus")
    p.add_argument("--topk_per_sample", type=int, default=5000,
                   help="Global top-k contribution filter applied before clustering")
    p.add_argument("--n_init", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    print(f"[NEUFS] config: {vars(args)}")

    config = AutoConfig.from_pretrained(args.model_name)
    num_layers = config.num_hidden_layers
    hidden_size = config.intermediate_size  # FFN up-proj dim = neuron dim

    records = load_neuron_jsonl(args.neuron_jsonl)
    pool = load_pool(args.pool_path)

    if len(records) != len(pool):
        print(
            f"[WARN] neuron jsonl has {len(records)} rows but pool has {len(pool)}. "
            "Selection will index into the first min(...) rows."
        )
    n = min(len(records), len(pool))
    records = records[:n]
    pool = pool[:n]

    active_mask, score_map, consensus_count, topk_active = build_features(
        records,
        num_layers=num_layers,
        hidden_size=hidden_size,
        topk_per_sample=args.topk_per_sample,
    )
    print(
        f"[NEUFS] features: {topk_active.shape}  "
        f"consensus min/mean/max = {consensus_count.min()}/{consensus_count.mean():.1f}/{consensus_count.max()}"
    )

    selected_idx = neufs_select(
        features=topk_active,
        consensus_count=consensus_count,
        n_shots=args.n_shots,
        tau=args.tau,
        n_init=args.n_init,
        verbose=args.verbose,
    )
    print(f"[NEUFS] selected indices: {selected_idx}")

    selected_samples = [pool[i] for i in selected_idx]
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(selected_samples, f, ensure_ascii=False, indent=2)

    print(f"[NEUFS] wrote {len(selected_samples)}-shot selection -> {args.output_path}")


if __name__ == "__main__":
    main()
