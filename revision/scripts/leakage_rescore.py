"""R1-2 diagnostic: Base vs ALIVE on the full test set and on near-duplicate-filtered subsets.

Uses existing per-example predictions (no regeneration). Max train-question cosine per test
item comes from audit_evidence.py (revision/results/test_max_train_cos.npy).
Output: revision/results/leakage_rescore.json
"""
import json
from pathlib import Path
import numpy as np
from rouge_score import rouge_scorer

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / "alive-eval-bundle/raw"
base, fine = json.load(open(B / "results_base.json")), json.load(open(B / "results_finetuned.json"))
mx = np.load(ROOT / "revision/results/test_max_train_cos.npy")
assert base["questions"] == fine["questions"] and len(mx) == len(base["questions"])
test_q = [json.loads(l)["instruction"] for l in open(ROOT / "alive-eval-bundle/data/test.jsonl")]
assert test_q == base["questions"], "test.jsonl order differs from results order"

sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
rl = lambda preds: np.array([sc.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, base["references"])])
M = {"bertscore_f1": (np.array(base["bertscore"]["f1_values"]), np.array(fine["bertscore"]["f1_values"])),
     "rougeL": (rl(base["predictions"]), rl(fine["predictions"]))}


def paired(a, b, idx, B_=2000):
    d = b[idx] - a[idx]; rng = np.random.default_rng(0)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(B_)]
    return {"n": int(len(idx)), "base": round(float(a[idx].mean()), 4), "alive": round(float(b[idx].mean()), 4),
            "delta": round(float(d.mean()), 4), "ci95": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]}


subsets = {"all": np.arange(len(mx)), "max_cos_lt_0.90": np.where(mx < .90)[0], "max_cos_lt_0.80": np.where(mx < .80)[0],
           "near_dup_ge_0.90": np.where(mx >= .90)[0]}
out = {s: {m: paired(*M[m], idx) for m in M} for s, idx in subsets.items()}
json.dump(out, open(ROOT / "revision/results/leakage_rescore.json", "w"), indent=1)
for s, v in out.items():
    print(s, {m: (x["n"], x["base"], x["alive"], x["delta"], x["ci95"]) for m, x in v.items()})
