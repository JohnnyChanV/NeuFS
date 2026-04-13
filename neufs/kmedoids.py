"""
Jaccard K-Medoids with multi-start (GPU accelerated).

Used to cluster samples by their sparse neuron activation sets. Inputs are
expected to be (n_samples, n_features) binary-or-thresholdable vectors --
values > 0 are treated as 1.
"""

import time

import numpy as np
import scipy.sparse as sparse
import torch


class JaccardKMedoids:
    def __init__(
        self,
        n_clusters: int = 5,
        max_iter: int = 100,
        n_init: int = 10,
        init: str = "k-means++",
        verbose: bool = False,
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_method = init
        self.verbose = verbose

        self.medoid_indices_ = None
        self.labels_ = None
        self.inertia_ = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------ utils
    def _to_tensor(self, X):
        if sparse.issparse(X):
            data = X.toarray().astype(np.float32)
        else:
            data = np.asarray(X, dtype=np.float32)
        t = torch.from_numpy(data).to(self.device)
        return (t > 0).float()

    @staticmethod
    def compute_jaccard_sim_matrix(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        intersection = X @ Y.t()
        card_X = X.sum(dim=1)
        card_Y = Y.sum(dim=1)
        union = card_X.unsqueeze(1) + card_Y.unsqueeze(0) - intersection
        return intersection / union.clamp(min=1e-8)

    # ------------------------------------------------------------------ init
    def _kmeans_pp_init(self, data_tensor: torch.Tensor) -> np.ndarray:
        n_samples = data_tensor.shape[0]
        centers = [int(np.random.randint(n_samples))]
        closest = torch.ones(n_samples, device=self.device)

        for _ in range(1, self.n_clusters):
            cur = data_tensor[centers[-1]].unsqueeze(0)
            sim = self.compute_jaccard_sim_matrix(data_tensor, cur).squeeze(1)
            closest = torch.min(closest, 1.0 - sim)
            probs = closest ** 2
            total = probs.sum()
            if total <= 0:
                remaining = list(set(range(n_samples)) - set(centers))
                cand = int(np.random.choice(remaining))
            else:
                cand = int(torch.multinomial(probs / total, 1).item())
            centers.append(cand)
        return np.array(centers)

    # ------------------------------------------------------------------ core
    def _run_single(self, data_tensor: torch.Tensor):
        n_samples = data_tensor.shape[0]

        if self.init_method == "k-means++":
            medoid_indices = self._kmeans_pp_init(data_tensor)
        elif isinstance(self.init_method, (list, np.ndarray)):
            medoid_indices = np.array(self.init_method)
        else:
            medoid_indices = np.random.choice(n_samples, self.n_clusters, replace=False)

        labels = None
        for _ in range(self.max_iter):
            medoids_tensor = data_tensor[medoid_indices]
            sim_matrix = self.compute_jaccard_sim_matrix(data_tensor, medoids_tensor)
            _, labels = torch.max(sim_matrix, dim=1)

            new_medoids = np.zeros(self.n_clusters, dtype=int)
            for k in range(self.n_clusters):
                mask = labels == k
                if not mask.any():
                    new_medoids[k] = int(np.random.randint(n_samples))
                    continue
                cluster_data = data_tensor[mask]
                cluster_global = torch.nonzero(mask, as_tuple=True)[0]
                intra = self.compute_jaccard_sim_matrix(cluster_data, cluster_data)
                best = int(torch.argmax(intra.sum(dim=1)).item())
                new_medoids[k] = int(cluster_global[best].item())

            if np.sum(medoid_indices != new_medoids) == 0:
                break
            medoid_indices = new_medoids

        final_medoids = data_tensor[medoid_indices]
        final_sim = self.compute_jaccard_sim_matrix(data_tensor, final_medoids)
        max_sims, final_labels = torch.max(final_sim, dim=1)
        inertia = float(torch.sum(1.0 - max_sims).item())
        return medoid_indices, final_labels.cpu().numpy(), inertia

    def fit(self, X):
        data_tensor = self._to_tensor(X)
        if self.verbose:
            print(
                f"[JaccardKMedoids] n_clusters={self.n_clusters} n_init={self.n_init} "
                f"device={self.device}"
            )

        best_inertia, best_medoids, best_labels = float("inf"), None, None
        t0 = time.time()
        for run_id in range(self.n_init):
            medoids, labels, inertia = self._run_single(data_tensor)
            if self.verbose:
                marker = "*" if inertia < best_inertia else ""
                print(f"  run {run_id + 1}/{self.n_init} inertia={inertia:.4f} {marker}")
            if inertia < best_inertia:
                best_inertia = inertia
                best_medoids = medoids
                best_labels = labels

        self.medoid_indices_ = best_medoids
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        if self.verbose:
            print(f"[JaccardKMedoids] best inertia={best_inertia:.4f} ({time.time() - t0:.1f}s)")

        self._data_tensor = data_tensor
        return self
