"""
NEUFS dual-criteria few-shot selection.

Given sparse neuron activation features and per-sample consensus counts:
  1. Cluster into C = n_shots groups via Jaccard K-Medoids.
  2. For every cluster:
       * normalize (min-max) the activation distance to the medoid
       * normalize (min-max) the consensus count Q(x) within the cluster
       * score(x) = tau * Q_norm(x) + (1 - tau) * (1 - D_norm(x))
     Pick the sample with the highest score as the cluster's representative.

Returns the global indices of the selected samples.
"""

from typing import List

import numpy as np
import torch

from .kmedoids import JaccardKMedoids


def neufs_select(
    features: np.ndarray,          # (N, L, H) or (N, D) binary-ish
    consensus_count: np.ndarray,   # (N,)
    n_shots: int,
    tau: float = 0.5,
    n_init: int = 10,
    max_iter: int = 100,
    init: str = "k-means++",
    verbose: bool = False,
) -> List[int]:
    X = features.reshape(features.shape[0], -1).astype(np.float32)

    kmedoids = JaccardKMedoids(
        n_clusters=n_shots,
        max_iter=max_iter,
        n_init=n_init,
        init=init,
        verbose=verbose,
    ).fit(X)

    labels = kmedoids.labels_
    data_tensor = kmedoids._data_tensor

    clusters_to_indices = {}
    for idx, cl in enumerate(labels.tolist()):
        clusters_to_indices.setdefault(cl, []).append(idx)

    selected = []
    for cluster_id in range(n_shots):
        cluster_indices = np.array(clusters_to_indices.get(cluster_id, []))
        if cluster_indices.size == 0:
            continue

        medoid_idx = int(kmedoids.medoid_indices_[cluster_id])

        # --- normalized consensus count Q_norm ---
        cluster_q = consensus_count[cluster_indices].astype(np.float32)
        q_min, q_max = cluster_q.min(), cluster_q.max()
        q_norm = (cluster_q - q_min) / (q_max - q_min + 1e-8)

        # --- normalized Jaccard distance D_norm (to medoid) ---
        with torch.no_grad():
            cluster_vecs = data_tensor[torch.as_tensor(cluster_indices, device=data_tensor.device)]
            medoid_vec = data_tensor[medoid_idx:medoid_idx + 1]
            sims = JaccardKMedoids.compute_jaccard_sim_matrix(cluster_vecs, medoid_vec)
            sims = sims.squeeze(1).cpu().numpy()
        dists = 1.0 - sims
        d_min, d_max = dists.min(), dists.max()
        if d_max - d_min > 1e-8:
            d_norm = (dists - d_min) / (d_max - d_min)
        else:
            d_norm = np.zeros_like(dists)

        # Higher is better: tau * Q_norm + (1 - tau) * (1 - D_norm)
        scores = tau * q_norm + (1.0 - tau) * (1.0 - d_norm)
        best_local = int(np.argmax(scores))
        selected.append(int(cluster_indices[best_local]))

    return selected
