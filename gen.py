#!/usr/bin/env python3
"""Generate 3 Unsloth QLoRA notebooks. FINAL — all bugs fixed."""
import json, os

def md(s): return {"cell_type":"markdown","metadata":{},"source":s}
def cd(s): return {"cell_type":"code","metadata":{},"source":s,"outputs":[],"execution_count":None}

META = {"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
        "language_info":{"name":"python","version":"3.10.0"}}

def build(mid, short, qlabel, tgt, out, bs, ga, pip_extra, save_cell, eval_cell):
    c = []
    # 0: Title
    c.append(md([f"# QLoRA Fine-Tuning: {mid} on Fable-5 Traces (Unsloth)\n","\n",
                 f"- **Model:** {mid}\n", f"- **Dataset:** Glint-Research/Fable-5-traces (pi_agent)\n",
                 f"- **Quant:** {qlabel}\n", f"- **Target Size:** ~{tgt} GB\n",
                 f"- **Output:** {out}\n", f"- **Hardware:** Kaggle 2x T4 (16 GB each)\n",
                 f"- **Framework:** Unsloth\n"]))
    # 1: Setup heading
    c.append(md(["## 1. Setup\n"]))
    # 2: pip install
    c.append(cd(["!pip install -q unsloth datasets einops sentencepiece tiktoken\n",
                 pip_extra + "\n"]))
    # 3: imports
    c.append(cd(["import os, gc, json, torch, matplotlib.pyplot as plt\n",
                 "from datasets import load_dataset\n",
                 "from unsloth import FastLanguageModel\n",
                 "from trl import SFTTrainer\n",
                 "from transformers import TrainingArguments\n"]))
    # 4: GPU check
    c.append(cd(["print(f\"CUDA: {torch.cuda.is_available()}\")\n",
                 "print(f\"GPUs: {torch.cuda.device_count()}\")\n",
                 "for i in range(torch.cuda.device_count()):\n",
                 "    p = torch.cuda.get_device_properties(i)\n",
                 "    print(f\"  GPU {i}: {p.name}  VRAM: {p.total_memory / 1024**3:.2f} GB\")\n"]))

    # 5: Dataset heading
    c.append(md(["## 2. Dataset\n"]))
    # 6: load + inspect
    c.append(cd(['ds = load_dataset("Glint-Research/Fable-5-traces", "pi_agent", split="train")\n',
                 "print(f\"Rows: {len(ds)}  Columns: {ds.column_names}\")\n",
                 'print("\\nFirst row keys:")\n',
                 "for k, v in ds[0].items():\n",
                 '    if k == "messages":\n',
                 "        print(f\"  messages: list[{len(v)}]\")\n",
                 "    else:\n",
                 "        print(f\"  {k}: {type(v).__name__} = {str(v)[:200]}\")\n"]))
    # 7: reformat — import JSON at TOP before functions
    c.append(cd(["import json\n",
                 "def flatten_content(parts):\n",
                 "    if isinstance(parts, str):\n",
                 "        return parts\n",
                 "    result = []\n",
                 "    for part in parts:\n",
                 '        t = part["type"]\n',
                 '        if t == "thinking":\n',
                 "            result.append(f\"<thinking>\\n{part['content']}\\n</thinking>\")\n",
                 '        elif t == "text":\n',
                 "            result.append(part[\"text\"])\n",
                 '        elif t == "toolCall":\n',
                 "            tool = part[\"toolUse\"]\n",
                 "            inp = json.dumps(tool[\"input\"], indent=2) if isinstance(tool[\"input\"], dict) else str(tool[\"input\"])\n",
                 "            result.append(f\"<tool_call>\\n<tool_name>{tool['name']}</tool_name>\\n<input>\\n{inp}\\n</input>\\n</tool_call>\")\n",
                 '    return "\\n".join(result)\n',
                 "def reformat_row(row):\n",
                 "    messages = []\n",
                 "    for m in row[\"messages\"]:\n",
                 '        messages.append({"role": m["role"], "content": flatten_content(m["content"])})\n',
                 "    return {\"messages\": messages}\n",
                 "ds = ds.map(reformat_row, remove_columns=ds.column_names)\n",
                 "print(f\"Reformatted: {len(ds)} rows\")\n",
                 'print("\\nSample:")\n',
                 "for i, m in enumerate(ds[0][\"messages\"][:3]):\n",
                 "    c = m[\"content\"][:250].replace(\"\\n\", \" \")\n",
                 "    print(f\"  [{i}] {m['role']}: {c}\")\n"]))

    # 8: QLoRA heading
    c.append(md(["## 3. QLoRA Fine-Tuning (Unsloth)\n"]))
    # 9: load model
    c.append(cd([f'model, tokenizer = FastLanguageModel.from_pretrained(\n',
                 f'    model_name="{mid}",\n',
                 f'    max_seq_length=4096,\n',
                 f'    load_in_4bit=True,\n',
                 f'    dtype=None,\n',
                 f')\n',
                 "tokenizer.pad_token = tokenizer.eos_token\n",
                 'tokenizer.padding_side = "right"\n',
                 'MODEL_NAME = "' + mid + '"\n',
                 "print(\"Model loaded via Unsloth\")\n"]))
    # 10: LoRA
    c.append(cd(["model = FastLanguageModel.get_peft_model(\n",
                 "    model,\n",
                 "    r=16,\n",
                 '    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],\n',
                 "    lora_alpha=32,\n",
                 "    lora_dropout=0,\n",
                 '    bias="none",\n',
                 '    use_gradient_checkpointing="unsloth",\n',
                 "    random_state=42,\n",
                 ")\n",
                 "print(f\"Trainable: {model.num_parameters(only_trainable=True):,} / {model.num_parameters():,}\")\n"]))
    # 11: TrainingArgs
    c.append(cd([f"args = TrainingArguments(\n",
                 f'    output_dir="./ckpt-{short}",\n',
                 f"    per_device_train_batch_size={bs},\n", f"    gradient_accumulation_steps={ga},\n",
                 f"    num_train_epochs=3,\n", f"    learning_rate=2e-4,\n",
                 f"    fp16=True, bf16=False,\n",
                 f'    optim="adamw_8bit",\n',
                 f"    logging_steps=1, log_level=\"info\", logging_first_step=True,\n",
                 f'    save_strategy="no", report_to="none",\n',
                 f"    warmup_ratio=0.03, lr_scheduler_type=\"cosine\",\n",
                 f"    gradient_checkpointing=True,\n",
                 f"    dataloader_num_workers=2,\n",
                 f"    ddp_find_unused_parameters=False,\n", f"    remove_unused_columns=False,\n",
                 f")\n"]))
    # 12: SFTTrainer
    c.append(cd(["def fmt(example):\n",
                 '    return tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)\n',
                 "trainer = SFTTrainer(\n",
                 "    model=model, args=args, train_dataset=ds,\n",
                 "    formatting_func=fmt, tokenizer=tokenizer, max_seq_length=4096,\n",
                 ")\n"]))
    # 13: train
    c.append(cd(["trainer.train()\n"]))
    # 14: loss plot — uses MODEL_NAME, not {mid}
    c.append(cd(['logs = trainer.state.log_history\n',
                 'steps = [x["step"] for x in logs if "loss" in x]\n',
                 'losses = [x["loss"] for x in logs if "loss" in x]\n',
                 "plt.figure(figsize=(10,5))\n", "plt.plot(steps, losses)\n",
                 'plt.xlabel("Step"); plt.ylabel("Loss")\n',
                 'plt.title(f"Training Loss - {MODEL_NAME}")\n', "plt.grid(True); plt.show()\n"]))

    # 15: Merge + Quantize heading
    c.append(md(["## 4. Merge + Quantize (Unsloth built-in)\n"]))
    # 16: helper fns
    c.append(cd(["def get_dir_size(path):\n",
                 "    return sum(os.path.getsize(os.path.join(dp,f)) for dp,_,fn in os.walk(path) for f in fn) / (1024**3)\n",
                 "def get_file_size(path):\n",
                 "    return os.path.getsize(path) / (1024**3)\n"]))
    # 17: define OUT
    c.append(cd([f'OUT = "{out}"\n', "os.makedirs(OUT, exist_ok=True)\n",
                 "print(f\"Saving to {OUT}\")\n"]))
    # 18: save + size check
    c.append(save_cell)

    # 19: Save confirmation
    c.append(md(["## 5. Save Confirmation\n"]))
    # 20: confirm
    c.append(cd(["print(f\"Model saved to: {OUT}\")\n", "print(f\"Contents: {os.listdir(OUT)}\")\n"]))

    # 21: Eval heading
    c.append(md(["## 6. Quick Sanity Eval\n"]))
    # 22: eval
    c.append(eval_cell)

    return {"cells":c, "metadata":META, "nbformat":4, "nbformat_minor":5}


# =============================================================================
# Qwen3-0.6B -> FP32 (~2.4 GB)
# =============================================================================
n1 = build("Qwen/Qwen3-0.6B", "qwen3-0.6b", "FP32", 2.4, "./model/qwen3-0.6b-fp32",
           8, 1, "# no extra packages",
    cd(["model.save_pretrained_merged(OUT, tokenizer, save_method=\"float32\")\n",
        "print(\"FP32 model saved.\")\n",
        "sz = get_dir_size(OUT)\n",
        "print(f\"FP32 size: {sz:.2f} GB (target 2.4 GB)\")\n",
        "if abs(sz - 2.4) > 0.2:\n",
        "    print(f\"WARNING: deviates >0.2 GB from target\")\n",
        "else:\n",
        "    print(\"OK: within tolerance\")\n"]),
    cd(["from transformers import pipeline\n",
        "pipe = pipeline(\"text-generation\", model=OUT, tokenizer=OUT, device=0)\n",
        "prompts = [\n",
        '    "Write a Python function to sort dicts by a key.",\n',
        '    "Diff between git merge and rebase?",\n',
        '    "Bash one-liner: files > 100MB.",\n',
        '    "Implement LRU cache in Python.",\n',
        '    "CORS in Flask?",\n',
        "]\n",
        "for i, p in enumerate(prompts):\n",
        '    r = pipe(p, max_new_tokens=256, do_sample=True, temperature=0.7)[0]["generated_text"]\n',
        "    print(f\"\\n=== {i+1} ===\\nQ: {p}\\nA: {r[len(p):].strip()}\")\n"]),
)

# =============================================================================
# Qwen3-4B -> INT4 via bitsandbytes (~2.0 GB)
# =============================================================================
n2 = build("Qwen/Qwen3-4B", "qwen3-4b", "INT4 (bitsandbytes)", 2.0, "./model/qwen3-4b-int4",
           4, 2, "# no extra packages",
    cd(["print(\"Saving merged INT4 model via Unsloth...\")\n",
        "model.save_pretrained_merged(OUT, tokenizer, save_method=\"merged_4bit\")\n",
        "sz = get_dir_size(OUT)\n",
        "print(f\"INT4 size: {sz:.2f} GB (target 2.0 GB)\")\n",
        "if abs(sz - 2.0) > 0.2:\n",
        "    print(f\"WARNING: deviates >0.2 GB from target\")\n",
        "else:\n",
        "    print(\"OK: within tolerance\")\n"]),
    cd(["from transformers import pipeline\n",
        "pipe = pipeline(\"text-generation\", model=OUT, tokenizer=OUT, device=0)\n",
        "prompts = [\n",
        '    "Write a Python function to sort dicts by a key.",\n',
        '    "Diff between git merge and rebase?",\n',
        '    "Bash one-liner: files > 100MB.",\n',
        '    "Implement LRU cache in Python.",\n',
        '    "CORS in Flask?",\n',
        "]\n",
        "for i, p in enumerate(prompts):\n",
        '    r = pipe(p, max_new_tokens=256, do_sample=True, temperature=0.7)[0]["generated_text"]\n',
        "    print(f\"\\n=== {i+1} ===\\nQ: {p}\\nA: {r[len(p):].strip()}\")\n"]),
)

# =============================================================================
# Qwen3-8B -> INT2 GGUF Q2_K (~2.0 GB)
# =============================================================================
n3 = build("Qwen/Qwen3-8B", "qwen3-8b", "INT2 (GGUF Q2_K)", 2.0, "./model/qwen3-8b-int2",
           1, 4, "!pip install -q llama-cpp-python",
    cd(["print(\"Saving GGUF Q2_K via Unsloth...\")\n",
        "model.save_pretrained_gguf(OUT, tokenizer, quantization_method=\"q2_k\")\n",
        "gguf_files = [f for f in os.listdir(OUT) if f.endswith(\".gguf\")]\n",
        "if gguf_files:\n",
        "    GGUF_PATH = os.path.join(OUT, gguf_files[0])\n",
        "    sz = get_file_size(GGUF_PATH)\n",
        "    print(f\"GGUF Q2_K size: {sz:.2f} GB (target 2.0 GB)\")\n",
        "    if abs(sz - 2.0) > 0.2:\n",
        "        print(f\"WARNING: deviates >0.2 GB from target\")\n",
        "    else:\n",
        "        print(\"OK: within tolerance\")\n",
        "else:\n",
        "    print(\"WARNING: No .gguf file found in output directory\")\n"]),
    cd(["from llama_cpp import Llama\n",
        "gguf_files = [f for f in os.listdir(OUT) if f.endswith(\".gguf\")]\n",
        "assert gguf_files, \"No GGUF file found\"\n",
        "GGUF_PATH = os.path.join(OUT, gguf_files[0])\n",
        "llm = Llama(model_path=GGUF_PATH, n_ctx=4096, n_gpu_layers=-1, verbose=False)\n",
        "prompts = [\n",
        '    "Write a Python function to sort dicts by a key.",\n',
        '    "Diff between git merge and rebase?",\n',
        '    "Bash one-liner: files > 100MB.",\n',
        '    "Implement LRU cache in Python.",\n',
        '    "CORS in Flask?",\n',
        "]\n",
        "for i, p in enumerate(prompts):\n",
        "    r = llm(p, max_tokens=256, temperature=0.7, echo=False)\n",
        "    print(f\"\\n=== {i+1} ===\\nQ: {p}\\nA: {r['choices'][0]['text'].strip()}\")\n"]),
)

# Write
base = os.path.dirname(os.path.abspath(__file__))
for name, nb in {"qlora_finetune_0.6b.ipynb": n1, "qlora_finetune_4b.ipynb": n2, "qlora_finetune_8b.ipynb": n3}.items():
    with open(os.path.join(base, name), "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Wrote {name}  ({len(nb['cells'])} cells)")
print("Done.")
