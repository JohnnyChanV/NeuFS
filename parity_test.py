"""
Parity test: run the original notebook's selection logic and NEUFS on the
same essay_comments cache, compare selected indices for several n_shots.

Both code paths share the same:
  - topk_active_feat (5000 per-sample top-k, zero-score filtered)
  - activated_neuron_num (consensus Q)
  - seed / init / n_init / max_iter / tau

The only difference is which JaccardKMedoids + selection loop runs them.
"""

import os
import sys

import numpy as np
import torch

ORIG_DIR = "/ihome/xli/zhc195/ix1_dir/ExpCLS/ExpCLS/PromptTuning/ActiveLearning"
NEW_DIR = "/ihome/xli/zhc195/ix1_dir/ExpCLS/NEUFS"

sys.path.insert(0, ORIG_DIR)
sys.path.insert(0, NEW_DIR)

from AL_algos.Jaccard_KM import JaccardKMedoids_MultiStart  # original
from neufs.select import neufs_select                         # new


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_topk_feat(score_neuron_features, k=5000):
    L, layer_num, hidden_size = score_neuron_features.shape
    flat_scores = score_neuron_features.reshape(L, -1)
    topk_indices = np.argsort(flat_scores, axis=1)[:, -k:]
    topk_active_feat = np.zeros_like(flat_scores)
    np.put_along_axis(topk_active_feat, topk_indices, 1, axis=1)
    mask_nonzero = flat_scores > 0
    topk_active_feat = topk_active_feat * mask_nonzero
    return topk_active_feat.reshape(L, layer_num, hidden_size)


def original_select(X, activated_neuron_num, n_shots, tau=0.5, seed=2025, n_init=10):
    """Replicate the notebook selection loop verbatim."""
    set_seed(seed)
    kmedoids = JaccardKMedoids_MultiStart(
        n_clusters=n_shots, max_iter=100, init="k-means++",
        verbose=False, n_init=n_init,
    )
    kmedoids.fit(X)
    cluster_labels = kmedoids.labels_

    indices_per_cluster = {}
    for idx, cl in enumerate(cluster_labels.tolist()):
        indices_per_cluster.setdefault(cl, []).append(idx)

    alpha = 0
    selected_indices = []
    for cluster_id in range(n_shots):
        if cluster_id not in indices_per_cluster:
            continue
        cluster_global_indices = np.array(indices_per_cluster[cluster_id])
        medoid_global_idx = kmedoids.medoid_indices_[cluster_id]

        this_cluster_neuron_nums = activated_neuron_num[cluster_global_indices]
        min_val, max_val = this_cluster_neuron_nums.min(), this_cluster_neuron_nums.max()
        normalized_neuron_nums = (this_cluster_neuron_nums - min_val) / (max_val - min_val + 1e-8)

        cluster_data_np = X[cluster_global_indices]
        medoid_data_np = X[medoid_global_idx].reshape(1, -1)
        cluster_tensor = kmedoids._to_tensor(cluster_data_np)
        medoid_tensor = kmedoids._to_tensor(medoid_data_np)

        with torch.no_grad():
            sim_tensor = kmedoids._compute_jaccard_sim_matrix(cluster_tensor, medoid_tensor)
            sims = sim_tensor.cpu().numpy().flatten()

        dists = 1.0 - sims
        d_min, d_max = dists.min(), dists.max()
        if d_max - d_min > 1e-8:
            norm_dists = (dists - d_min) / (d_max - d_min)
        else:
            norm_dists = np.zeros_like(dists)

        sample_scores = (1 - tau) * (1 - norm_dists) + tau * normalized_neuron_nums
        best_local = int(np.argmax(sample_scores))
        selected_indices.append(int(cluster_global_indices[best_local]))

    return selected_indices


def neufs_run(topk_active_feat, activated_neuron_num, n_shots, tau=0.5, seed=2025, n_init=10):
    set_seed(seed)
    X = topk_active_feat.reshape(len(topk_active_feat), -1)
    return neufs_select(
        features=X,
        consensus_count=activated_neuron_num,
        n_shots=n_shots,
        tau=tau,
        n_init=n_init,
        verbose=False,
    )


def main():
    cache_path = os.path.join(
        ORIG_DIR,
        "neuron_cache/essay_comments/Qwen3-4B-Instruct-2507_neuron_cache.npz",
    )
    print(f"[parity] loading {cache_path}")
    cache = np.load(cache_path, allow_pickle=True)
    score_neuron_features = cache["score_neuron_features"]
    flat_features = cache["flat_features"]
    activated_neuron_num = flat_features.sum(-1).astype(np.int64)
    print(f"[parity] N={len(flat_features)} score_shape={score_neuron_features.shape}")

    topk_active_feat = build_topk_feat(score_neuron_features, k=5000)
    X = np.asarray(topk_active_feat.reshape(len(topk_active_feat), -1))
    print(f"[parity] X={X.shape}  per-sample active counts: "
          f"min={X.sum(1).min()} mean={X.sum(1).mean():.0f} max={X.sum(1).max()}")

    for tau in [0.0, 0.5, 1.0]:
        for n_shots in [5, 10, 20, 30]:
            orig = original_select(X, activated_neuron_num, n_shots, tau=tau)
            new = neufs_run(topk_active_feat, activated_neuron_num, n_shots, tau=tau)
            orig_s, new_s = sorted(orig), sorted(new)
            match = orig_s == new_s
            print(f"[parity] tau={tau} n_shots={n_shots}  "
                  f"match={match}  overlap={len(set(orig_s) & set(new_s))}/{n_shots}")
            if not match:
                print(f"    orig: {orig_s}")
                print(f"    new : {new_s}")


if __name__ == "__main__":
    main()
