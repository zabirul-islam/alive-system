"""R1-2: score Base vs a split-retrained adapter on that split's held-out test set (lecture-restricted context).
Metric code identical to score_2x2.py. Adds the near-duplicate-filtered subset (max cosine to the split's
own training questions < 0.90).
Usage: python score_split.py test_ctx.jsonl train.jsonl gen_base.jsonl gen_new.jsonl out.json
"""
import json, sys
import numpy as np
from bert_score import score as bscore
from rouge_score import rouge_scorer
from scipy.stats import wilcoxon
from sentence_transformers import SentenceTransformer
from transformers import pipeline

ctx_f, train_f, gb_f, gn_f, out_f = sys.argv[1:6]
C = [json.loads(l) for l in open(ctx_f)]
refs = [c["reference"] for c in C]
P = {m: [r["pred_lecture"] for r in map(json.loads, open(f))] for m, f in [("base", gb_f), ("new", gn_f)]}
nli = pipeline("text-classification", model="facebook/bart-large-mnli", device=0)
rs = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
M = {}
for m, preds in P.items():
    _, _, F = bscore(preds, refs, lang="en", model_type="microsoft/deberta-xlarge-mnli", verbose=False)
    M[m] = {"bertscore_f1": F.numpy(), "rougeL": np.array([rs.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs)]),
            "nli_entail": np.array([0.0 if not p.strip() else float("entail" in nli(f"{c['ctx_lecture'][:1024]} </s></s> {p}", truncation=True)[0]["label"].lower())
                                    for p, c in zip(preds, C)])}
st = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda")
tq = [json.loads(l)["instruction"] for l in open(train_f)]
mx = (st.encode([c["question"] for c in C], normalize_embeddings=True) @ st.encode(tq, normalize_embeddings=True).T).max(1)


def comp(idx):
    o = {"n": int(len(idx))}
    for mt in M["base"]:
        a, b = M["base"][mt][idx], M["new"][mt][idx]; d = b - a; rng = np.random.default_rng(0)
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        o[mt] = {"base": round(float(a.mean()), 4), "adapted": round(float(b.mean()), 4), "delta": round(float(d.mean()), 4),
                 "ci95": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
                 "p_wilcoxon": float(wilcoxon(b, a).pvalue) if np.any(d != 0) else 1.0}
    return o


out = {"all": comp(np.arange(len(C))), "max_cos_lt_0.90": comp(np.where(mx < .9)[0]),
       "words": {m: round(float(np.mean([len(p.split()) for p in P[m]])), 1) for m in P}}
json.dump(out, open(out_f, "w"), indent=1)
print(json.dumps(out, indent=1))
