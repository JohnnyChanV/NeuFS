# NEUFS — Neuron-Aware Active Few-Shot Learning for LLMs



[English](README.md) | **中文**

*Zhuowei Chen, Liwei Chen, Christian Schunn, Raquel Coelho, Xiang Lorraine Li*

**NEUFS** 的最小开源实现,对应论文 _Neuron-Aware Active Few-Shot Learning for LLMs_。
给定一个无标注候选池,NEUFS 利用目标 LLM 内部的 FFN 神经元激活,挑选
出一组小而多样、且对幻觉敏感的 few-shot 示例,用于 in-context learning。

![NEUFS 流程总览](assets/overview.png)

整条流程分为两个自包含的阶段:

1. **激活采集** — 让模型跑一遍候选池的每个样本,把 FFN 激活通过 LM head
   做早期 unembedding,记录每个样本贡献最大的若干神经元。
2. **双准则选样** — 以稀疏神经元集合之间的 Jaccard 相似度做聚类,然后
   在每个簇内按下式挑一个代表:
   `score(x) = tau · Q̃(x) + (1 - tau) · (1 - D̃(x))`,其中
   * `Q̃(x)` 是簇内按 min-max 归一化的神经元"共识计数"
     (越高 → 独特电路越多 → 越容易产生幻觉),
   * `D̃(x)` 是簇内到 medoid 的 Jaccard 距离(min-max 归一化),
     (越小 → 越能代表整簇)。

**为什么 Q(x) 能追踪幻觉?** 激活"独特神经元"越多的样本,经验上越
容易被模型答错——按 `#Unique Activations` 分箱后,准确率随之单调下降
(p<0.001)。这和已有工作的发现一致:神经元层面的 agreement / consensus
是判断 LLM 是否走在可靠推理路径上的一个可解释的机制层面信号
[[Li et al., 2025a]](https://arxiv.org/abs/2504.07440),
[[Li et al., 2025b]](https://arxiv.org/abs/2510.26277)。

## 目录结构

```
NEUFS/
├── neufs/                         核心库
│   ├── collate.py                 多选 / 候选项 collator
│   ├── collect.py                 FFN hook + early-unembedding top-k
│   ├── features.py                jsonl → (active_mask, score_map, Q(x))
│   ├── kmedoids.py                Jaccard K-Medoids(多起点,GPU 版)
│   └── select.py                  双准则逐簇选样
├── scripts/
│   ├── 01_collect_activations.py  Stage 1 CLI
│   ├── 02_select_fewshot.py       Stage 2 CLI
│   └── run_example.sh             essay_comments 端到端示例
└── examples/essay_comments/       示例候选池 + system prompt
```

## 安装

```bash
pip install -r requirements.txt
```

需要一块 GPU,显存足够在 bf16/fp16 下跑目标模型。

## 快速上手

候选池格式 —— 一个 JSON 或 JSONL 列表,每条记录一个候选样本:

```json
{"id": 0, "input": "Try to vary your sentence length...", "label": "With Explanation"}
```

严格只需要 `input`。`label` 会原样透传到输出 JSON
(当 NEUFS 作为"驱动标注"的选样器时有用)。

### Stage 1 —— 采集神经元激活

```bash
python scripts/01_collect_activations.py \
    --model_name Qwen/Qwen3-4B-Instruct-2507 \
    --pool_path examples/essay_comments/pool.jsonl \
    --system_prompt_file examples/essay_comments/system_prompt.txt \
    --prompt_template '<text>{}</text> Is this text contains the explanation relation?' \
    --candidates "Without Explanation" "With Explanation" \
    --output_path cache/essay_comments_qwen3-4b.jsonl \
    --batch_size 4 \
    --top_k_per_layer 2000
```

输出:每行一个样本的 JSON,字段为
`messages`, `top_neurons`, `entropy`, `pred`, `label`。

### Stage 2 —— 选 few-shot

```bash
python scripts/02_select_fewshot.py \
    --model_name Qwen/Qwen3-4B-Instruct-2507 \
    --neuron_jsonl cache/essay_comments_qwen3-4b.jsonl \
    --pool_path examples/essay_comments/pool.jsonl \
    --output_path outputs/essay_comments/neufs_5shot.json \
    --n_shots 5 \
    --tau 0.5 \
    --topk_per_sample 4000 \
    --n_init 10 \
    --verbose
```

输出:一个 JSON 数组,恰好 `n_shots` 条记录,顺序即选中顺序。直接塞进
你的 few-shot prompt 模板即可。

### 端到端示例

```bash
bash scripts/run_example.sh
```

## 程序化 API

```python
from neufs.features import load_neuron_jsonl, build_features
from neufs.select import neufs_select

records = load_neuron_jsonl("cache/essay_comments_qwen3-4b.jsonl")
_, _, consensus, feats = build_features(
    records, num_layers=36, hidden_size=12288, topk_per_sample=5000,
)
indices = neufs_select(feats, consensus, n_shots=10, tau=0.5)
```

## 注意事项

* FFN hook 的路径(`model.model.layers[n].mlp.act_fn`)假设模型是
  LLaMA 风格的架构(LLaMA-3, Qwen-3, Mistral 等)。非标准模型可能需要
  换一个 hook 目标。
* features 里的 `hidden_size` 指的是 `config.intermediate_size`
  (FFN 神经元数),**不是** `config.hidden_size`。
* 激活采集时,整批会把一层的 `act_fn` 输出留在显存里;遇到长 prompt
  或大模型时,把 `--batch_size` 调小。

## 代码参考

[`neufs/collect.py`](neufs/collect.py) 里的神经元激活采集部分直接移植自
**MUI-Eval** 的 `get_neuron`:
[ALEX-nlp/MUI-Eval – neuron_and_sae/get_performance/get_neuron.py](https://github.com/ALEX-nlp/MUI-Eval/blob/main/neuron_and_sae/get_performance/get_neuron.py)。
FFN hook 位置、贡献度公式 (`activate_scores * token_projections`)、
以及每层 `top_k = min(top_k_per_layer, num_positions * hidden_size)`
的 flatten-then-topk 规则都沿用 MUI-Eval。

## 引用

如果这份代码对你有帮助,请引用论文。
