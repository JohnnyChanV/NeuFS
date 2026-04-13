"""
Neuron activation collection (early unembedding).

For each sample in the unlabeled pool:
  1. Score every candidate by sequence log-prob, pick the argmax as the model's
     prediction y_hat.
  2. During the same forward pass, hook every layer's FFN activation-function
     output (the input to the down-projection).
  3. For each token position t in the predicted answer span, compute the
     contribution score  S^l_{y_t, i} = k^l_i * (w^l_{out,i} . e_{y_t})
     i.e. the element-wise product of the neuron activation and the neuron's
     direct projection onto the predicted-token embedding (Chen et al., 2025).
  4. Per layer, flatten (num_positions, hidden_size) and keep the top-K
     (position, neuron) entries by contribution score; dump to JSONL.

The resulting JSONL is the only input needed for the selection step
(see `scripts/02_select_fewshot.py`).

Reference implementation this is ported from:
  https://github.com/ALEX-nlp/MUI-Eval/blob/main/neuron_and_sae/get_performance/get_neuron.py
The hook target, the contribution formula (`activate_scores * token_projections`),
and the per-layer flatten-then-topk convention all follow MUI-Eval.
"""

import json
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


class NeuronActivationCollector:
    def __init__(self, model, tokenizer, candidates, top_k_per_layer: int = 2000):
        self.model = model
        self.tokenizer = tokenizer
        self.candidates = list(candidates)
        self.top_k_per_layer = top_k_per_layer

    @torch.no_grad()
    def run(self, dataset, collator, output_path: str, batch_size: int = 4):
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        open(output_path, "w", encoding="utf-8").close()

        loader = DataLoader(
            dataset, batch_size=batch_size, collate_fn=collator, shuffle=False
        )
        self.model.eval()

        num_layers = self.model.config.num_hidden_layers
        vocab_size = self.tokenizer.vocab_size
        unembedding = self.model.lm_head.weight.data.T.to(self.model.device)[:, :vocab_size]

        for batch in tqdm(loader, desc="Collecting activations"):
            self._process_batch(batch, num_layers, unembedding, output_path)

        print(f"[NEUFS] Wrote {output_path}")

    def _process_batch(self, batch, num_layers, unembedding, output_path):
        input_ids = batch["input_ids"].to(self.model.device)
        attention_mask = batch["attention_mask"].to(self.model.device)
        candidate_index = batch["candidate_index"]
        answer_ranges = batch["answer_range"]
        batch_labels = batch["labels"]

        B = input_ids.size(0)
        L = input_ids.size(1)

        # Hook act_fn outputs (post-activation FFN values -> input to down_proj).
        act = {n: [] for n in range(num_layers)}

        def make_hook(n):
            def fn(_, _input, output):
                act[n].append(output.detach())
            return fn

        handles = [
            self.model.model.layers[n].mlp.act_fn.register_forward_hook(make_hook(n))
            for n in range(num_layers)
        ]
        try:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
        finally:
            for h in handles:
                h.remove()

        logits = outputs.logits

        # -- sequence log-probs per candidate --
        seq_logprobs = []
        for i in range(B):
            s, e = answer_ranges[i]
            pred_logits = logits[i, s - 1:e - 1, :]
            target_ids = input_ids[i][s:e]
            log_probs = F.log_softmax(pred_logits, dim=-1)
            token_log_probs = log_probs[range(len(target_ids)), target_ids]
            seq_logprobs.append(token_log_probs.mean().item())

        # -- pick best candidate per example, compute entropy over candidates --
        score_dict = defaultdict(dict)
        for (ex_idx, cand_idx), score in zip(candidate_index, seq_logprobs):
            score_dict[ex_idx][cand_idx] = score

        preds, pred_entropies, pred_seq_indices = [], [], []
        for ex_idx in sorted(score_dict.keys()):
            cand_scores = score_dict[ex_idx]
            best_cand = max(cand_scores, key=cand_scores.get)
            preds.append(self.candidates[best_cand])

            scores_tensor = torch.tensor(
                [cand_scores[k] for k in cand_scores], dtype=torch.float32
            )
            log_p = scores_tensor - torch.logsumexp(scores_tensor, dim=0)
            p = torch.exp(log_p)
            pred_entropies.append(float(-(p * log_p).sum().item()))

            pred_seq_indices.append(candidate_index.index((ex_idx, best_cand)))

        if not pred_seq_indices:
            return

        this_B = len(pred_seq_indices)
        pred_answer_ranges = np.array(answer_ranges)[pred_seq_indices]
        messages_list = batch["messages"]

        # -- gather top-k contribution neurons --
        # Each entry: list over layers, each a list over this_B, each a list
        # of 1-d tensors of length hidden_size (one per position in predicted span).
        scores_logit = [[[] for _ in range(this_B)] for _ in range(num_layers)]
        scores_activate = [[[] for _ in range(this_B)] for _ in range(num_layers)]

        position_mask = torch.zeros((this_B, L), dtype=torch.bool)
        input_position_mask = torch.zeros((this_B, L), dtype=torch.bool)
        for i, (s, e) in enumerate(pred_answer_ranges):
            position_mask[i, s - 1:e - 1] = True
            input_position_mask[i, s:e] = True

        batch_idx_flat, pos_idx_flat = position_mask.nonzero(as_tuple=True)
        input_batch_idx_flat, input_pos_idx_flat = input_position_mask.nonzero(as_tuple=True)

        target_input_ids = input_ids[pred_seq_indices][input_batch_idx_flat, input_pos_idx_flat]

        for layer in range(num_layers):
            w_down_T = self.model.model.layers[layer].mlp.down_proj.weight.data.T.to(
                self.model.device
            )  # (hidden_size, d_model)
            # activation at the predicted-span positions
            activate_scores = act[layer][0][pred_seq_indices][batch_idx_flat, pos_idx_flat, :].to(
                self.model.device
            )
            # (hidden_size, num_target_tokens) -> transposed to (num_target_tokens, hidden_size)
            token_projections = torch.matmul(
                w_down_T.float(),
                unembedding[:, target_input_ids].float(),
            ).half().T

            contribution = activate_scores * token_projections  # (N_positions, hidden_size)
            for idx, b_idx in enumerate(batch_idx_flat.tolist()):
                scores_logit[layer][b_idx].append(contribution[idx].detach().cpu())
                scores_activate[layer][b_idx].append(activate_scores[idx].detach().cpu())

        # -- per case / per layer: top-k by contribution score --
        # Matches MUI-Eval's get_neuron.py: flatten (num_positions, hidden_size)
        # and take top-k over the full flattened dim, capped at top_k_per_layer.
        with open(output_path, "a", encoding="utf-8") as f:
            for i in range(this_B):
                top_neurons = []
                for layer in range(num_layers):
                    if not scores_logit[layer][i]:
                        continue
                    stacked = torch.stack(scores_logit[layer][i], dim=0)  # (num_positions, hidden_size)
                    num_positions, hidden_size = stacked.shape
                    flat = stacked.flatten()
                    top_values, top_indices = torch.topk(flat, self.top_k_per_layer)
                    for v, flat_idx in zip(top_values.tolist(), top_indices.tolist()):
                        top_neurons.append({
                            "layer": layer,
                            "position": flat_idx // hidden_size,
                            "neuron": flat_idx % hidden_size,
                            "score": float(v),
                        })

                record = {
                    "messages": messages_list[i],
                    "top_neurons": top_neurons,
                    "entropy": pred_entropies[i],
                    "pred": preds[i],
                    "label": batch_labels[i],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
