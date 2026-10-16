"""Segment-aware NLI (external review): the context is split into windows of <=1,024 characters at
paragraph/sentence boundaries; an answer is 'entailed' if any window entails it, 'contradicted' if no window
entails and any window contradicts, else 'neutral'. Same BART-large-MNLI model and input format as before.
Usage: python nli_segments.py contexts.jsonl out.json label=gen.jsonl:cond [label=gen.jsonl:cond ...]
"""
import json, re, sys
import numpy as np
from scipy.stats import wilcoxon
from transformers import pipeline

ctx_f, out_f, specs = sys.argv[1], sys.argv[2], sys.argv[3:]
CL = [json.loads(l) for l in open(ctx_f)]
nli = pipeline("text-classification", model="facebook/bart-large-mnli", device=0, top_k=None)


def windows(text, n=1024):
    parts = [p for p in re.split(r"(?<=[.!?])\s+|\n\n", text) if p.strip()]
    out, cur = [], ""
    for p in parts:
        if len(cur) + len(p) + 1 > n and cur:
            out.append(cur); cur = ""
        cur = (cur + " " + p).strip()[:n]
    return out + ([cur] if cur else [])


def label(ans, ctx):
    if not ans.strip():
        return "contradiction"
    labs = []
    for w in windows(ctx):
        sc = {d["label"].lower(): d["score"] for d in nli(f"{w} </s></s> {ans}", truncation=True)[0]}
        labs.append(max(sc, key=sc.get))
    return "entailment" if "entailment" in labs else ("contradiction" if "contradiction" in labs else "neutral")


res = {}
for spec in specs:
    lab, rest = spec.split("=", 1); gf, cond = rest.split(":")
    GL = [json.loads(l) for l in open(gf)]  # generation rows are in the same order as the contexts file
    assert len(GL) == len(CL)
    ids = list(range(len(CL))); C = dict(enumerate(CL)); G = dict(enumerate(GL))
    L = [label(G[i][f"pred_{cond}"], C[i][f"ctx_{cond}"]) for i in ids]
    res[lab] = {"ids": ids, "labels": L, "entail": round(float(np.mean([l == "entailment" for l in L])), 4),
                "neutral": round(float(np.mean([l == "neutral" for l in L])), 4), "contra": round(float(np.mean([l == "contradiction" for l in L])), 4),
                "mean_windows": round(float(np.mean([len(windows(C[i][f"ctx_{cond}"])) for i in ids])), 2)}
    print(lab, {k: v for k, v in res[lab].items() if k not in ("ids", "labels")}, flush=True)
pairs = {}
for a in res:
    for b in res:
        if a.startswith("alive") and b == a.replace("alive", "base", 1) or (a.startswith("new") and b == a.replace("new", "base", 1)):
            x = np.array([l == "entailment" for l in res[a]["labels"]], float); y = np.array([l == "entailment" for l in res[b]["labels"]], float)
            d = x - y; rng = np.random.default_rng(0); bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
            pairs[f"{a} - {b}"] = {"delta": round(float(d.mean()), 4), "ci95": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
                                   "p_wilcoxon": float(wilcoxon(x, y).pvalue) if np.any(d != 0) else 1.0}
json.dump({"cells": {k: {kk: vv for kk, vv in v.items() if kk not in ("ids", "labels")} for k, v in res.items()}, "contrasts": pairs,
           "labels": {k: v["labels"] for k, v in res.items()}}, open(out_f, "w"), indent=1)
print(json.dumps(pairs, indent=1))
