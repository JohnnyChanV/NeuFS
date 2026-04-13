#!/usr/bin/env python
"""
Step 1: run the target LLM over the unlabeled pool, capture per-sample
contribution-scored neuron activations, and dump them to JSONL.

Expected input JSON/JSONL: a list of records, each with
    {
        "id":     <any unique id>,           # optional
        "input":  <raw text fed into prompt_template>,
        "label":  <gold label string or index, optional>
    }

The prompt is built as
    system = <system_prompt_file content>
    user   = prompt_template.format(input)
    assistant = "<answer>{candidate}</answer>"   (one per candidate)
"""

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neufs.collate import CandidateCollator
from neufs.collect import NeuronActivationCollector


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


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_dataset(data, tokenizer, system_prompt, prompt_template):
    def _map(row):
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_template.format(row["input"])},
        ]
        out = {"messages": msgs, "sem_label": row.get("label")}
        return out

    ds = Dataset.from_list(data)
    return ds.map(_map)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True, help="HF model id or local path")
    p.add_argument("--pool_path", required=True, help="JSON/JSONL unlabeled pool")
    p.add_argument("--system_prompt_file", required=True)
    p.add_argument("--prompt_template", default="{}")
    p.add_argument("--candidates", required=True, nargs="+",
                   help="Candidate label strings, space-separated")
    p.add_argument("--output_path", required=True,
                   help="Where to write the neuron JSONL")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--top_k_per_layer", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    print(f"[NEUFS] config: {vars(args)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, device_map="auto", torch_dtype=dtype
    )
    model.eval()

    pool = load_pool(args.pool_path)
    system_prompt = read_text(args.system_prompt_file)
    dataset = build_dataset(pool, tokenizer, system_prompt, args.prompt_template)

    collator = CandidateCollator(tokenizer, args.candidates)
    collector = NeuronActivationCollector(
        model=model,
        tokenizer=tokenizer,
        candidates=args.candidates,
        top_k_per_layer=args.top_k_per_layer,
    )
    collector.run(
        dataset=dataset,
        collator=collator,
        output_path=args.output_path,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
