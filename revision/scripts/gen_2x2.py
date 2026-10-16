"""R1-1: generate answers for one model under both context conditions (axis2, vLLM offline).

Prompt, stop tokens and decoding are copied verbatim from eval_llm.py / eval_rag.py.
Usage: CUDA_VISIBLE_DEVICES=2 python gen_2x2.py --model <path-or-hf-id> --label base|alive \
           --contexts contexts_2x2.jsonl --out gen_<label>.jsonl
"""
import argparse, json
from vllm import LLM, SamplingParams

SYSTEM = ("You are a teaching assistant answering a student. "
          "Use ONLY the lecture context provided. "
          "Answer in 1-3 clear, precise sentences.")


def build_prompt(question, ctx, system=SYSTEM, nocontext=False):  # default == eval_llm.build_prompt == eval_rag.build_prompt
    user = f"QUESTION:\n{question}\n\nANSWER:" if nocontext else f"LECTURE CONTEXT:\n{ctx}\n\nQUESTION:\n{question}\n\nANSWER:"
    return ("<|begin_of_text|>"
            f"<|start_header_id|>system<|end_header_id|>\n{system}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n{user}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n")


ap = argparse.ArgumentParser()
for a in ["--model", "--label", "--contexts", "--out"]:
    ap.add_argument(a, required=True)
ap.add_argument("--conds", default="lecture,retrieved")
ap.add_argument("--system", default=SYSTEM, help="override system prompt (E2 length ablation)")
ap.add_argument("--nocontext", action="store_true", help="ungrounded prompt without lecture context (E1)")
args = ap.parse_args()
rows = [json.loads(l) for l in open(args.contexts)]
llm = LLM(model=args.model, dtype="bfloat16", seed=42, gpu_memory_utilization=0.85, max_model_len=4096)
sp = SamplingParams(temperature=0.0, max_tokens=200, stop=["<|eot_id|>", "<|end_of_text|>", "\n\n"], seed=42)
out = {"retrieved": [""] * len(rows), "retrieved_prompt_tokens": [0] * len(rows)}
for cond in args.conds.split(","):
    prompts = [build_prompt(r["question"], r.get(f"ctx_{cond}", ""), args.system, args.nocontext) for r in rows]
    res = llm.generate(prompts, sp)
    out[cond] = [o.outputs[0].text.strip() for o in res]
    out[f"{cond}_prompt_tokens"] = [len(o.prompt_token_ids) for o in res]
with open(args.out, "w") as f:
    for i, r in enumerate(rows):
        f.write(json.dumps({"i": i, "label": args.label, "pred_lecture": out["lecture"][i], "pred_retrieved": out["retrieved"][i],
                            "prompt_tokens_lecture": out["lecture_prompt_tokens"][i],
                            "prompt_tokens_retrieved": out["retrieved_prompt_tokens"][i]}) + "\n")
print("wrote", args.out, len(rows))
