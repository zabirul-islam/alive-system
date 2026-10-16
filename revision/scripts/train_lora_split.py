import os
from pathlib import Path
from datetime import datetime

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer

# ── CONFIG ────────────────────────────────────────────────────
MODEL_ID   = "meta-llama/Meta-Llama-3.1-8B-Instruct"
TRAIN_PATH = os.environ["TRAIN_PATH"]
VAL_PATH   = os.environ["VAL_PATH"]
OUTPUT_DIR = os.environ["OUTPUT_DIR"]
LOG_DIR    = os.environ["OUTPUT_DIR"]
MAX_SEQ    = 2048

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

# ── LOG START ─────────────────────────────────────────────────
start_time = datetime.now()
log_path   = Path(LOG_DIR) / "finetune.log"
with open(log_path, "a") as f:
    f.write(f"\n{'='*60}\n")
    f.write(f"TRAINING START: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"{'='*60}\n")
print(f"TRAINING START: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

# ── TOKENIZER ─────────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token        = tokenizer.eos_token
tokenizer.padding_side     = "right"
tokenizer.model_max_length = MAX_SEQ

# ── MODEL ─────────────────────────────────────────────────────
print("Loading model in bfloat16...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache      = False
model.config.pretraining_tp = 1

# Critical: enable input gradients before applying LoRA
model.enable_input_require_grads()

# ── LORA ──────────────────────────────────────────────────────
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
    bias="none",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── DATASET ───────────────────────────────────────────────────
def format_alpaca(example):
    if example.get("input", "").strip():
        return {"text": (
            "Below is an instruction that describes a task, paired with an input "
            "that provides further context.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Input:\n{example['input']}\n\n"
            f"### Response:\n{example['output']}"
        )}
    return {"text": (
        "Below is an instruction that describes a task.\n\n"
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Response:\n{example['output']}"
    )}

print("Loading datasets...")
train_ds = load_dataset("json", data_files=TRAIN_PATH, split="train")
val_ds   = load_dataset("json", data_files=VAL_PATH,   split="train")
train_ds = train_ds.map(format_alpaca, remove_columns=train_ds.column_names)
val_ds   = val_ds.map(format_alpaca,   remove_columns=val_ds.column_names)
print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")

# ── TRAINING ARGS ─────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=50,
    bf16=True,
    tf32=True,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="none",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataloader_num_workers=2,
    run_name="alive-lora-llama31-8b",
)

# ── TRAINER ───────────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tokenizer,
)

# ── TRAIN ─────────────────────────────────────────────────────
print("Starting training...")
trainer.train()

# ── SAVE ──────────────────────────────────────────────────────
print("Saving LoRA adapter...")
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Saved to {OUTPUT_DIR}/")

# ── LOG END ───────────────────────────────────────────────────
end_time   = datetime.now()
duration   = end_time - start_time
hours, rem = divmod(duration.seconds, 3600)
mins, secs = divmod(rem, 60)
with open(log_path, "a") as f:
    f.write(f"TRAINING END:   {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"DURATION:       {hours:02d}h {mins:02d}m {secs:02d}s\n")
    f.write(f"{'='*60}\n")
print(f"TRAINING END:   {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"DURATION:       {hours:02d}h {mins:02d}m {secs:02d}s")
