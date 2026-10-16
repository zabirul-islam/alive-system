#!/bin/bash
# Waits for both split retrains, then merge -> generate (base + adapted) -> score. Runs on axis2 in ~/alive_rev.
cd ~/alive_rev
for n in slide_grouped lecture_heldout; do
  while [ ! -f runs/$n/adapter_model.safetensors ] || pgrep -f "OUTPUT_DIR=runs/$n" >/dev/null || ! grep -q "TRAINING END" runs/$n/finetune.log 2>/dev/null; do sleep 30; done
  env/bin/python merge_adapter.py runs/$n runs/$n/merged > runs/$n/merge.log 2>&1
  CUDA_VISIBLE_DEVICES=$1 env/bin/python gen_2x2.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --label base --conds lecture --contexts splits/$n/test_ctx.jsonl --out runs/$n/gen_base.jsonl > runs/$n/gen_base.log 2>&1
  CUDA_VISIBLE_DEVICES=$1 env/bin/python gen_2x2.py --model runs/$n/merged --label adapted --conds lecture --contexts splits/$n/test_ctx.jsonl --out runs/$n/gen_new.jsonl > runs/$n/gen_new.log 2>&1
  CUDA_VISIBLE_DEVICES=$1 env/bin/python score_split.py splits/$n/test_ctx.jsonl splits/$n/train.jsonl runs/$n/gen_base.jsonl runs/$n/gen_new.jsonl runs/$n/score.json > runs/$n/score.log 2>&1
  echo "$n DONE" >> after_train.done
done
