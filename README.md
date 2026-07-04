# Size-Comparison

QLoRA fine-tuning of **Qwen3 models** (0.6B, 4B, 8B) on the [Fable-5 Traces](https://huggingface.co/datasets/Glint-Research/Fable-5-traces) dataset using [Unsloth](https://github.com/unslothai/unsloth).

Each model is quantized to a different precision to compare the trade-off between model size, quality, and inference cost.

## Notebooks

| Notebook | Model | Quant | Target Size | Batch / GradAcc |
|---|---|---|---|---|
| `qlora_finetune_0.6b.ipynb` | Qwen3-0.6B | FP32 | ~2.4 GB | 8 / 1 |
| `qlora_finetune_4b.ipynb` | Qwen3-4B | INT4 (bitsandbytes) | ~2.0 GB | 4 / 2 |
| `qlora_finetune_8b.ipynb` | Qwen3-8B | INT2 (GGUF Q2_K) | ~2.0 GB | 1 / 4 |

All notebooks are self-contained (pip installs inside) and designed for **Kaggle 2x T4 (16 GB each)**.

## Dataset

[Glint-Research/Fable-5-traces](https://huggingface.co/datasets/Glint-Research/Fable-5-traces) — `pi_agent` config (~4665 rows). Agent traces containing `thinking`, `toolCall`, and `text` parts reformatted into standard chat messages.

## Quick Start

1. Open on Kaggle (or any 2x T4 environment)
2. Run all cells in order
3. Model is saved to `./model/qwen3-{size}-{quant}/`

## Tech Stack

- [Unsloth](https://github.com/unslothai/unsloth) — fast QLoRA with memory-efficient kernels
- [TRL](https://huggingface.co/docs/trl/en/index) SFTTrainer — supervised fine-tuning
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) — 4-bit NF4 quantization
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — GGUF Q2_K for 8B model
