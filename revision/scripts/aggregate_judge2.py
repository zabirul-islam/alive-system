"""Aggregate judge_v2 outputs: per-cell axis means, paired ALIVE-vs-Base contrasts within each context
(Wilcoxon + bootstrap CI, Holm within judge), error-type distribution, and inter-judge agreement
(exact agreement and quadratic-weighted kappa per axis) when both judges are present.
Usage: python aggregate_judge2.py out.json judge1.jsonl [judge2.jsonl]
"""
import json, sys
from collections import Counter
import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import cohen_kappa_score

out_f, files = sys.argv[1], sys.argv[2:]
AX = ["F", "P", "G", "A"]
res = {}
data = {}
for f in files:
    rows = [json.loads(l) for l in open(f)]
    j = rows[0]["judge"]; d = {(r["i"], r["cell"]): r for r in rows}; data[j] = d
    cells = sorted({r["cell"] for r in rows})
    o = {"n_rows": len(rows), "failures": sum(r["F"] is None for r in rows), "cells": {}, "contrasts": {}, "errors": {}}
    for c in cells:
        rs = [r for r in rows if r["cell"] == c and r["F"] is not None]
        o["cells"][c] = {ax: round(float(np.mean([float(r[ax]) for r in rs])), 3) for ax in AX}
        o["errors"][c] = dict(Counter(str(r.get("E")).lower() for r in rs))
        o["errors"][c]["any_error_rate"] = round(float(np.mean([str(r.get("E")).lower() not in ("none", "null") for r in rs])), 3)
    pv = []
    for ctx in ["lecture", "retrieved"]:
        for ax in AX:
            ids = [i for (i, c) in d if c == f"base|{ctx}" and (i, f"alive|{ctx}") in d
                   and d[(i, f"base|{ctx}")][ax] is not None and d[(i, f"alive|{ctx}")][ax] is not None]
            b = np.array([float(d[(i, f"base|{ctx}")][ax]) for i in ids]); a = np.array([float(d[(i, f"alive|{ctx}")][ax]) for i in ids])
            dd = a - b; rng = np.random.default_rng(0); bs = [dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(2000)]
            p = float(wilcoxon(a, b).pvalue) if np.any(dd != 0) else 1.0
            o["contrasts"][f"{ctx}:{ax}"] = {"n": len(ids), "delta": round(float(dd.mean()), 3),
                                             "ci95": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)],
                                             "win": round(float((dd > 0).mean()), 3), "tie": round(float((dd == 0).mean()), 3),
                                             "loss": round(float((dd < 0).mean()), 3), "p": p}
            pv.append((f"{ctx}:{ax}", p))
    pv.sort(key=lambda x: x[1]); run = 0
    for k, (name, p) in enumerate(pv):
        run = max(run, min(1.0, (len(pv) - k) * p)); o["contrasts"][name]["p_holm"] = run
    res[j] = o
if len(data) == 2:
    j1, j2 = list(data)
    common = sorted(set(data[j1]) & set(data[j2]))
    res["inter_judge"] = {}
    for ax in AX:
        pr = [(int(data[j1][k][ax]), int(data[j2][k][ax])) for k in common if data[j1][k][ax] is not None and data[j2][k][ax] is not None]
        x, y = zip(*pr)
        res["inter_judge"][ax] = {"n": len(pr), "exact": round(float(np.mean(np.array(x) == np.array(y))), 3),
                                  "kappa_qw": round(float(cohen_kappa_score(x, y, weights="quadratic")), 3)}
    pe = [(str(data[j1][k].get("E")).lower() != "none", str(data[j2][k].get("E")).lower() != "none") for k in common]
    x, y = zip(*pe)
    res["inter_judge"]["any_error"] = {"exact": round(float(np.mean(np.array(x) == np.array(y))), 3),
                                       "kappa": round(float(cohen_kappa_score(x, y)), 3)}
json.dump(res, open(out_f, "w"), indent=1)
for j, o in res.items():
    if j == "inter_judge":
        print("inter-judge", o); continue
    print("==", j, "rows", o["n_rows"], "fail", o["failures"])
    for c, v in o["cells"].items():
        print(f"  {c:16s}", v, "err", o["errors"][c]["any_error_rate"])
    for k, v in o["contrasts"].items():
        print(f"  {k:14s} d={v['delta']:+.3f} {v['ci95']} W/T/L {v['win']:.2f}/{v['tie']:.2f}/{v['loss']:.2f} pH={v['p_holm']:.1e}")
