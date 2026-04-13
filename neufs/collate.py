"""
Candidate-based collator for multiple-choice / classification scoring.

For each example we build |candidates| sequences of
    <system> <user_prompt> <assistant>"<answer>{cand}</answer>"
and record the token range of `{cand}` so downstream code can compute
per-candidate log-probabilities.
"""

import torch
from torch.nn.utils.rnn import pad_sequence


class CandidateCollator:
    def __init__(self, tokenizer, candidates, pad_token_id=None):
        self.tokenizer = tokenizer
        self.candidates = list(candidates)
        self.pad_token_id = pad_token_id if pad_token_id is not None else tokenizer.pad_token_id

    def __call__(self, examples):
        all_ids = []
        candidate_index = []
        labels = []
        answer_ranges = []
        messages_list = []

        for ex_idx, example in enumerate(examples):
            messages_list.append(example["messages"])
            for cand_idx, cand in enumerate(self.candidates):
                prefix_length = self.tokenizer.apply_chat_template(
                    example["messages"],
                    add_generation_prompt=True,
                    enable_thinking=False,
                    return_tensors="pt",
                )[0].shape[0] + len(self.tokenizer.encode("<answer>", add_special_tokens=False))

                cand_len = len(self.tokenizer.encode(cand, add_special_tokens=False))

                ids = self.tokenizer.apply_chat_template(
                    example["messages"] + [
                        {"role": "assistant", "content": f"<answer>{cand}</answer>"}
                    ],
                    add_generation_prompt=False,
                    enable_thinking=False,
                    return_tensors="pt",
                )[0]

                answer_range = (prefix_length, prefix_length + cand_len)
                decoded = self.tokenizer.decode(ids[answer_range[0]:answer_range[1]])
                assert decoded == cand, (
                    f"Candidate token range mismatch: got '{decoded}' expected '{cand}'"
                )

                all_ids.append(ids)
                answer_ranges.append(answer_range)
                candidate_index.append((ex_idx, cand_idx))

            labels.append(example.get("sem_label", example.get("label")))

        padded = pad_sequence(all_ids, batch_first=True, padding_value=self.pad_token_id)
        attention_mask = (padded != self.pad_token_id).long()

        return {
            "messages": messages_list,
            "input_ids": padded,
            "attention_mask": attention_mask,
            "candidate_index": candidate_index,
            "answer_range": answer_ranges,
            "labels": labels,
        }
