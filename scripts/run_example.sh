#!/usr/bin/env bash
# End-to-end example: Essay-Feedback (binary) with Qwen3-4B-Instruct-2507.
# Adjust MODEL, POOL, CANDS to run on your own data.

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="Qwen/Qwen3-4B-Instruct-2507"
POOL="examples/essay_comments/pool.jsonl"
SYS_PROMPT="examples/essay_comments/system_prompt.txt"
NEURON_OUT="cache/essay_comments_qwen3-4b.jsonl"

python scripts/01_collect_activations.py \
    --model_name "$MODEL" \
    --pool_path "$POOL" \
    --system_prompt_file "$SYS_PROMPT" \
    --prompt_template '<text>{}</text> Is this text contains the explanation relation?' \
    --candidates "Without Explanation" "With Explanation" \
    --output_path "$NEURON_OUT" \
    --batch_size 4 \
    --top_k_per_layer 2000

for N in 5 10 20 30; do
    python scripts/02_select_fewshot.py \
        --model_name "$MODEL" \
        --neuron_jsonl "$NEURON_OUT" \
        --pool_path "$POOL" \
        --output_path "outputs/essay_comments/neufs_${N}shot.json" \
        --n_shots "$N" \
        --tau 0.5 \
        --topk_per_sample 4000 \
        --n_init 10 \
        --verbose
done
