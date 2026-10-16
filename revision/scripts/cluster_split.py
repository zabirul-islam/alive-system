"""R1-2 follow-up (external review): cluster-aware uncertainty for the split experiments.
Recomputes per-item BERTScore F1 (same scorer) for Base and the retrained adapter, then reports a cluster
bootstrap that resamples whole slides (slide-grouped) or whole lectures (lecture-held-out), and per-lecture deltas.
Usage: python cluster_split.py test_ctx.jsonl gen_base.jsonl gen_new.jsonl cluster_key(slide|lecture) out.json
"""
import json, sys
from collections import defaultdict
import numpy as np
from bert_score import score as bscore

ctx_f, gb_f, gn_f, key, out_f = sys.argv[1:6]
C = [json.loads(l) for l in open(ctx_f)]
refs = [c["reference"] for c in C]
f1 = {}
for m, f in [("base", gb_f), ("new", gn_f)]:
    preds = [r["pred_lecture"] for r in map(json.loads, open(f))]
    f1[m] = bscore(preds, refs, lang="en", model_type="microsoft/deberta-xlarge-mnli", verbose=False)[2].numpy()
d = f1["new"] - f1["base"]
groups = defaultdict(list)
for i, c in enumerate(C):
    groups[c[key]].append(i)
gk = list(groups); rng = np.random.default_rng(0); bs = []
for _ in range(5000):
    idx = np.concatenate([groups[g] for g in rng.choice(gk, len(gk), replace=True)])
    bs.append(d[idx].mean())
out = {"cluster_key": key, "n_items": len(C), "n_clusters": len(gk), "delta": round(float(d.mean()), 4),
       "cluster_ci95": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]}
if key == "lecture":
    out["per_lecture"] = {g: {"n": len(ix), "delta": round(float(d[ix].mean()), 4), "share_positive": round(float((d[ix] > 0).mean()), 3)}
                          for g, ix in groups.items()}
json.dump(out, open(out_f, "w"), indent=1); print(json.dumps(out, indent=1))
