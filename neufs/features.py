"""
Feature construction from the neuron JSONL produced by `collect.py`.

For each sample we build:
  * `active_mask`   : (L, hidden)  binary mask of unique activated neurons,
                      unioned across all predicted-answer positions. Used to
                      compute `consensus_count` (= |N_act(x)|).
  * `score_map`     : (L, hidden)  per-neuron contribution scores (largest
                      score wins on collision). Used for the optional
                      top-K re-filter before clustering.
  * `consensus_count`: int, number of uniquely activated neurons.

If `topk_per_sample` is provided, we further sparsify `active_mask` to the
top-K globally-ranked contribution scores per sample (positive entries only),
matching the two-stage filter described in the paper.
"""

import json
from typing import List, Optional, Tuple

import numpy as np


def load_neuron_jsonl(path: str) -> List[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_features(
    records: List[dict],
    num_layers: int,
    hidden_size: int,
    topk_per_sample: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        active_mask       : (N, L, H) float32, 0/1
        score_map         : (N, L, H) float32
        consensus_count   : (N,) int64    -- |N_act(x)|, used as Q(x)
        topk_active_feat  : (N, L, H) float32, 0/1 (top-k filtered if requested,
                            otherwise same as active_mask)
    """
    N = len(records)
    active_mask = np.zeros((N, num_layers, hidden_size), dtype=np.float32)
    score_map = np.zeros((N, num_layers, hidden_size), dtype=np.float32)
    consensus_count = np.zeros(N, dtype=np.int64)

    for idx, rec in enumerate(records):
        neuron_meta = rec.get("top_neurons", [])
        if not neuron_meta:
            continue

        # NOTE: matches the original notebook exactly -- `range(max_pos)` is
        # half-open, so the last position (== max_pos) is intentionally dropped.
        max_pos = max(item["position"] for item in neuron_meta)
        unique_neurons = set()
        local_scores = {}

        for pos in range(max_pos):
            pos_items = [it for it in neuron_meta if it["position"] == pos]
            pos_items.sort(key=lambda x: x["score"], reverse=True)
            for it in pos_items:
                key = (it["layer"], it["neuron"])
                unique_neurons.add(key)
                # keep the strongest score we've seen for this (layer, neuron)
                prev = local_scores.get(key, float("-inf"))
                if it["score"] > prev:
                    local_scores[key] = it["score"]

        consensus_count[idx] = len(unique_neurons)
        for (layer, neuron), score in local_scores.items():
            if 0 <= layer < num_layers and 0 <= neuron < hidden_size:
                active_mask[idx, layer, neuron] = 1.0
                score_map[idx, layer, neuron] = score

    if topk_per_sample is None:
        topk_active = active_mask.copy()
    else:
        topk_active = _topk_filter(score_map, topk_per_sample)

    return active_mask, score_map, consensus_count, topk_active


def _topk_filter(score_map: np.ndarray, k: int) -> np.ndarray:
    N, L, H = score_map.shape
    flat = score_map.reshape(N, -1)
    k = min(k, flat.shape[1])
    topk_idx = np.argsort(flat, axis=1)[:, -k:]
    out = np.zeros_like(flat)
    np.put_along_axis(out, topk_idx, 1.0, axis=1)
    out = out * (flat > 0)
    return out.reshape(N, L, H)
