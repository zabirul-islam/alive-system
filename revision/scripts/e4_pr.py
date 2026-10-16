"""E4 length context (post hoc, descriptive): BERTScore precision / recall / F1 and word counts for Base, ALIVE and the
length-instructed Base (E2 prompt) on the cross-segment set, for both contexts (pred_lecture = Eq. 1, pred_retrieved = semantic).
Paired ALIVE - X differences with bootstrap 95% CI. Same BERTScore model as score_2x2.py.
Usage: python e4_pr.py contexts_e4.jsonl gen_base.jsonl gen_alive.jsonl gen_base_brief.jsonl out.json
"""
import json, sys
import numpy as np
from bert_score import score as bscore

ctx_f, gb, ga, gbr, out_f = sys.argv[1:6]
refs = [json.loads(l)["reference"] for l in open(ctx_f)]
G = {k: [json.loads(l) for l in open(f)] for k, f in [("base", gb), ("alive", ga), ("base_brief", gbr)]}
out = {"ref_words": float(np.mean([len(r.split()) for r in refs]))}
for ctx in ["lecture", "retrieved"]:
    M = {}
    for k, rows in G.items():
        preds = [r[f"pred_{ctx}"] for r in rows]
        P, Rr, F = bscore(preds, refs, lang="en", model_type="microsoft/deberta-xlarge-mnli", verbose=False)
        M[k] = {"P": P.numpy(), "R": Rr.numpy(), "F": F.numpy(), "words": np.array([len(p.split()) for p in preds], float)}
    o = {k: {m: round(float(v.mean()), 4) for m, v in d.items()} for k, d in M.items()}
    for b in ["base", "base_brief"]:
        for m in ["P", "R", "F"]:
            dd = M["alive"][m] - M[b][m]; rng = np.random.default_rng(0)
            bs = [dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(2000)]
            o[f"alive-{b}:{m}"] = {"delta": round(float(dd.mean()), 4), "ci95": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]}
    out[ctx] = o
json.dump(out, open(out_f, "w"), indent=1); print(json.dumps(out, indent=1))
