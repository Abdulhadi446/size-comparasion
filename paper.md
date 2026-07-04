# Size-Comparison: QLoRA Fine-Tuning of Qwen3 Models at Varying Precision

## Abstract

We investigate the feasibility and trade-offs of fine-tuning **Qwen3 models** at three different scales — 0.6B, 4B, and 8B parameters — using **QLoRA** (Quantized Low-Rank Adaptation) with the [Fable-5 Traces](https://huggingface.co/datasets/Glint-Research/Fable-5-traces) dataset. Each model is quantized to a different precision (FP32, INT4, INT2 GGUF Q2_K) targeting approximately **2 GB of disk space**, enabling a controlled comparison of performance vs. compression ratio.

## Motivation

Large language models (LLMs) are increasingly deployed in resource-constrained environments. Understanding how aggressive quantization interacts with fine-tuning quality across model sizes is critical for practical deployment decisions. This work provides a reproducible benchmark using three quantization strategies applied to the same agent-traces dataset.

## Methodology

### Models

| Model | Parameters | Base Precision |
|---|---|---|
| Qwen3-0.6B | 624M | FP32 |
| Qwen3-4B | 3.7B | bfloat16 |
| Qwen3-8B | 7.6B | bfloat16 |

### Quantization Strategies

| Model | Target | Method | Disk Target |
|---|---|---|---|
| 0.6B | FP32 | Unsloth `save_method="float32"` | ~2.4 GB |
| 4B | INT4 | Unsloth `save_method="merged_4bit"` (bitsandbytes NF4) | ~2.0 GB |
| 8B | INT2 | Unsloth `save_pretrained_gguf` (Q2_K) | ~2.0 GB |

### Fine-Tuning

- **Framework:** Unsloth (FastLanguageModel) + TRL SFTTrainer
- **LoRA Rank:** 16, Alpha: 32, Dropout: 0
- **Target Modules:** All attention + feed-forward linear projections
- **Sequence Length:** 4096 tokens
- **Training:** 3 epochs, AdamW 8-bit, cosine LR schedule, 2e-4 peak LR
- **Hardware:** 2× NVIDIA T4 (16 GB each, Kaggle)
- **Batch Sizing** (per-GPU / grad accum):

| Model | Batch Size | Grad Accum | Effective BS |
|---|---|---|---|
| 0.6B | 8 | 1 | 16 |
| 4B | 4 | 2 | 16 |
| 8B | 1 | 4 | 8 |

### Dataset

The [Fable-5 Traces](https://huggingface.co/datasets/Glint-Research/Fable-5-traces) dataset (`pi_agent` config) contains approximately 4,665 agent interaction traces. Each trace includes multi-turn messages with `thinking`, `toolCall`, and `text` parts. Messages are flattened into a unified chat format compatible with SFTTrainer.

### Reformatter Approach

```
<thinking>
... model reasoning ...
</thinking>
<tool_call>
<tool_name>tool_name</tool_name>
<input>
  ... JSON or text ...
</input>
</tool_call>
... assistant text response ...
```

## Expected Outcomes

1. **Loss curves** should converge for all three models, with larger models achieving lower final loss.
2. **Inference quality** on held-out prompts will be assessed qualitatively across 5 test prompts (Python coding, Git, Bash, algorithms, web development).
3. **Size targets** should be met within ±0.2 GB tolerance.
4. **Speed:** 0.6B FP32 will train fastest; 8B Q2_K will be slowest due to smaller batch size and GGUF overhead.

## Discussion

### Compression Ratio

| Model | Full Precision | Quantized | Ratio |
|---|---|---|---|
| 0.6B | ~2.4 GB (FP32) | ~2.4 GB | 1× |
| 4B | ~7.4 GB (bf16) | ~2.0 GB | 3.7× |
| 8B | ~15.2 GB (bf16) | ~2.0 GB | 7.6× |

The 8B INT2 model achieves a **7.6× compression ratio** while retaining fine-tuning capability via QLoRA, making it the most interesting candidate for edge deployment.

### Limitations

- Evaluation is qualitative (5 manual prompts) rather than a benchmark suite.
- GGUF Q2_K may exhibit quality degradation compared to INT4.
- Single dataset (agent traces) — results may not generalize.

## Repository Structure

```
.
├── README.md
├── paper.md
├── LICENSE
├── gen.py                          # Notebook generator script
├── qlora_finetune_0.6b.ipynb       # 0.6B FP32 notebook
├── qlora_finetune_4b.ipynb         # 4B INT4 notebook
└── qlora_finetune_8b.ipynb         # 8B INT2 GGUF notebook
```

## References

1. [Unsloth — Fast QLoRA](https://github.com/unslothai/unsloth)
2. [Qwen3 Technical Report](https://arxiv.org/abs/2505.03730)
3. [QLoRA: Efficient Finetuning of Quantized Language Models](https://arxiv.org/abs/2305.14314)
4. [Fable-5 Traces Dataset](https://huggingface.co/datasets/Glint-Research/Fable-5-traces)
5. [GGML / llama.cpp](https://github.com/ggml-org/llama.cpp)
