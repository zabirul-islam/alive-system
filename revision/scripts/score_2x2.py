"""R1-1: score the 2x2 (Base/ALIVE x lecture-restricted/retrieved context) — run on axis2 GPU.

Metric code mirrors alive_scripts/evaluation/eval_llm.py (BERTScore deberta-xlarge-mnli, ROUGE,
BART-MNLI entailment, numeric-unit check). Also reports how often regenerated outputs match the
original submission's outputs for the three pre-existing cells.
Usage: python score_2x2.py contexts_2x2.jsonl gen_base.jsonl gen_alive.jsonl out.json [per_example.jsonl]
"""
import json, re, sys
import numpy as np
from bert_score import score as bscore
from rouge_score import rouge_scorer
from scipy.stats import wilcoxon
from transformers import pipeline

ctx_f, gb_f, ga_f, out_f = sys.argv[1:5]
ex_f = sys.argv[5] if len(sys.argv) > 5 else None
C = [json.loads(l) for l in open(ctx_f)]
G = {"base": [json.loads(l) for l in open(gb_f)], "alive": [json.loads(l) for l in open(ga_f)]}
refs = [c["reference"] for c in C]
cells = {(m, k): [g[f"pred_{k}"] for g in G[m]] for m in G for k in ["lecture", "retrieved"]}

NUM = re.compile(r"\b(\d+(?:\.\d+)?)\s*(HU|ms|MHz|kHz|GHz|MeV|keV|mm|cm|T1|T2|Tesla|T)\b", re.I)


def numeric_flags(preds, ctxs):  # eval_llm.slide_grounded_factual_error
    fl = []
    for p, c in zip(preds, ctxs):
        bad = any(m.group(1) not in c and f"{m.group(1)} {m.group(2)}" not in c and m.group(0) not in c
                  for m in NUM.finditer(p))
        fl.append(int(bad))
    return fl


nli = pipeline("text-classification", model="facebook/bart-large-mnli", device=0)
rs = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
per = {}
for (m, k), preds in cells.items():
    ctxs = [c[f"ctx_{k}"] for c in C]
    _, _, F = bscore(preds, refs, lang="en", model_type="microsoft/deberta-xlarge-mnli", verbose=False)
    r = [rs.score(ref, p) for p, ref in zip(preds, refs)]
    lab = []
    for p, c in zip(preds, ctxs):
        lab.append("contradiction" if not p.strip() else nli(f"{c[:1024]} </s></s> {p}", truncation=True)[0]["label"].lower())
    per[(m, k)] = {"bertscore_f1": F.numpy(), "rouge1": np.array([x["rouge1"].fmeasure for x in r]),
                   "rouge2": np.array([x["rouge2"].fmeasure for x in r]), "rougeL": np.array([x["rougeL"].fmeasure for x in r]),
                   "nli_entail": np.array([float("entail" in l) for l in lab]),
                   "nli_contra": np.array([float("contra" in l) for l in lab]),
                   "numeric_err": np.array(numeric_flags(preds, ctxs), float),
                   "words": np.array([len(p.split()) for p in preds], float)}


def boot(d, B=2000):
    rng = np.random.default_rng(0); bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)]
    return [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


out = {"n": len(C), "cells": {f"{m}|{k}": {mt: round(float(v.mean()), 4) for mt, v in per[(m, k)].items()} for (m, k) in per}}
con = {"model_effect|lecture": (("alive", "lecture"), ("base", "lecture")),
       "model_effect|retrieved": (("alive", "retrieved"), ("base", "retrieved")),
       "context_effect|base": (("base", "retrieved"), ("base", "lecture")),
       "context_effect|alive": (("alive", "retrieved"), ("alive", "lecture"))}
out["contrasts"] = {}
for name, (a, b) in con.items():
    out["contrasts"][name] = {}
    for mt in ["bertscore_f1", "rougeL", "nli_entail"]:
        d = per[a][mt] - per[b][mt]
        p = float(wilcoxon(per[a][mt], per[b][mt]).pvalue) if np.any(d != 0) else 1.0
        out["contrasts"][name][mt] = {"delta": round(float(d.mean()), 4), "ci95": boot(d), "p_wilcoxon": p}
inter = {mt: (per[("alive", "retrieved")][mt] - per[("base", "retrieved")][mt]) - (per[("alive", "lecture")][mt] - per[("base", "lecture")][mt])
         for mt in ["bertscore_f1", "rougeL", "nli_entail"]}
out["interaction"] = {mt: {"delta": round(float(d.mean()), 4), "ci95": boot(d)} for mt, d in inter.items()}
orig = {("base", "lecture"): "orig_base_lecture", ("alive", "lecture"): "orig_alive_lecture", ("base", "retrieved"): "orig_base_retrieved"}
out["reproduction_exact_match"] = {f"{m}|{k}": round(float(np.mean([p.strip() == c[o].strip() for p, c in zip(cells[(m, k)], C)])), 4)
                                  for (m, k), o in orig.items()}
out["prompt_tokens_match_original_rag"] = round(float(np.mean([g["prompt_tokens_retrieved"] == c["orig_rag_prompt_tokens"]
                                                              for g, c in zip(G["base"], C)])), 4)
for d in ["basic", "intermediate", "advanced"]:
    idx = np.array([c["difficulty"] == d for c in C])
    out.setdefault("by_difficulty_bertscore", {})[d] = {f"{m}|{k}": round(float(per[(m, k)]["bertscore_f1"][idx].mean()), 4) for (m, k) in per}
json.dump(out, open(out_f, "w"), indent=1)
if ex_f:
    with open(ex_f, "w") as f:
        for i in range(len(C)):
            f.write(json.dumps({"i": i, **{f"{m}|{k}|{mt}": float(per[(m, k)][mt][i]) for (m, k) in per for mt in per[(m, k)]}}) + "\n")
print(json.dumps(out["cells"], indent=1)); print(json.dumps(out["contrasts"], indent=1))
print("repro", out["reproduction_exact_match"], "rag prompt tokens match", out["prompt_tokens_match_original_rag"])
