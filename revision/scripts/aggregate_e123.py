"""Aggregate judge results for the additional conditions E1-E3 (and E4 when present).
Cells come from judge_e123_*.jsonl plus the original 2x2 cells (judge2x2_*.jsonl) used as comparators.
Contrasts (paired, per judge): E1 alive_none - base_none; E2 alive|lecture - base_brief, base_brief - base|lecture;
E3 alive_e2e - base_e2e; E4 alive - base per context. Wilcoxon + bootstrap CI on F/P/G/A; McNemar on any-error.
Holm within judge over all axis contrasts. Output: revision/results/judge_e123.json
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon, binomtest

R = Path(__file__).resolve().parents[1]; J = R / "results/judge"
AX = ["F", "P", "G", "A"]
CONTRASTS = [("E1", "alive_none|lecture", "base_none|lecture"), ("E2a", "alive|lecture", "base_brief|lecture"),
             ("E2b", "base_brief|lecture", "base|lecture"), ("E3", "alive_e2e|lecture", "base_e2e|lecture")]
E4 = [("E4_e2e", "alive|lecture", "base|lecture"), ("E4_sem", "alive|retrieved", "base|retrieved"),
      ("E4L_e2e", "alive|lecture", "base_brief|lecture"), ("E4L_sem", "alive|retrieved", "base_brief|retrieved")]


def err(r):
    return str(r.get("E")).lower() not in ("none", "null")


def summarize(d, contrasts):
    o = {"cells": {}, "contrasts": {}}
    for c in sorted({c for _, c in d}):
        rs = [r for (i, cc), r in d.items() if cc == c and r["F"] is not None]
        o["cells"][c] = {**{ax: round(float(np.mean([float(r[ax]) for r in rs])), 3) for ax in AX},
                         "err": round(float(np.mean([err(r) for r in rs])), 3), "n": len(rs)}
    pv = []
    for name, a_c, b_c in contrasts:
        ids = sorted(i for (i, c) in d if c == a_c and (i, b_c) in d and d[(i, a_c)]["F"] is not None and d[(i, b_c)]["F"] is not None)
        if not ids:
            continue
        for ax in AX:
            dd = np.array([float(d[(i, a_c)][ax]) - float(d[(i, b_c)][ax]) for i in ids])
            rng = np.random.default_rng(0); bs = [dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(2000)]
            p = float(wilcoxon(dd).pvalue) if np.any(dd != 0) else 1.0
            o["contrasts"][f"{name}:{ax}"] = {"n": len(ids), "delta": round(float(dd.mean()), 3),
                                              "ci95": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)], "p": p}
            pv.append((f"{name}:{ax}", p))
        b01 = sum(err(d[(i, b_c)]) and not err(d[(i, a_c)]) for i in ids); b10 = sum(err(d[(i, a_c)]) and not err(d[(i, b_c)]) for i in ids)
        o["contrasts"][f"{name}:err"] = {"n": len(ids), "a_err": sum(err(d[(i, a_c)]) for i in ids), "b_err": sum(err(d[(i, b_c)]) for i in ids),
                                         "mcnemar_p": float(binomtest(b10, b01 + b10).pvalue) if b01 + b10 else 1.0}
    pv.sort(key=lambda x: x[1]); run = 0
    for k, (name, p) in enumerate(pv):
        run = max(run, min(1.0, (len(pv) - k) * p)); o["contrasts"][name]["p_holm"] = run
    return o


res = {}
for j in ["gpt-4o", "claude-sonnet-4-6"]:
    d = {}
    for f in [J / f"judge2x2_{j}.jsonl", J / f"judge_e123_{j}.jsonl"]:
        for r in map(json.loads, open(f)):
            d[(r["i"], r["cell"])] = r
    res[j] = summarize(d, CONTRASTS)
    f4 = J / f"judge_e4_{j}.jsonl"
    if f4.exists():
        res[j + "|E4"] = summarize({(r["i"], r["cell"]): r for r in map(json.loads, open(f4))}, E4)
json.dump(res, open(R / "results/judge_e123.json", "w"), indent=1)
for j, o in res.items():
    print("==", j)
    for c, v in o["cells"].items():
        print(f"  {c:20s}", v)
    for k, v in o["contrasts"].items():
        print(f"  {k:12s}", {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()})
