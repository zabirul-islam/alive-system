"""Merge a LoRA adapter into Llama-3.1-8B-Instruct (bf16) for vLLM serving: merge_adapter.py <adapter_dir> <out_dir>"""
import sys, torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", torch_dtype=torch.bfloat16)
m = PeftModel.from_pretrained(base, sys.argv[1]).merge_and_unload()
m.save_pretrained(sys.argv[2], safe_serialization=True)
AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct").save_pretrained(sys.argv[2])
