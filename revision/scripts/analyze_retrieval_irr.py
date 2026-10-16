"""Second relevance annotation (free selection, same procedure as the first author): agreement and robustness.
Reads revision/annotation/retrieval_relevance_annotator2.xlsx (sheet Questions, column "Relevant slides ...", entries
"lecture-slide" separated by ';' or ','), and the first author's labels (alive-eval-bundle/raw/retrieval_gt.json).
Agreement per question: Jaccard and Dice (positive specific agreement) of the two relevant sets, plus Cohen's kappa
over all slides of every lecture either annotator used for that question (pooled over questions).
Robustness: re-runs retrieval_v2.py with relevance = annotator 2, intersection, and union; the pause anchor stays
the first author's first slide so only the relevance labels change.
Usage: python analyze_retrieval_irr.py [path.xlsx]     Output: revision/results/retrieval_irr.json
"""
import json, re, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

R = Path(__file__).resolve().parents[1]; ROOT = R.parent
GT = json.load(open(ROOT / "alive-eval-bundle/raw/retrieval_gt.json"))
chunks = json.load(open(ROOT / "backend/rag_store/chunks.json"))
valid = {c["slide_id"] for c in chunks}
by_lec = {}
for s in valid:
    by_lec.setdefault(s.split("__")[0], set()).add(s)
xlsx = Path(sys.argv[1]) if len(sys.argv) > 1 else R / "annotation/retrieval_relevance_annotator2.xlsx"
df = pd.read_excel(xlsx, sheet_name="Questions")
col = [c for c in df.columns if str(c).startswith("Relevant slides")][0]


def parse(cell):
    ids, bad = [], []
    for tok in re.split(r"[;,\n]+", str(cell) if pd.notna(cell) else ""):
        m = re.fullmatch(r"\s*(\d+)\s*[-–:/ ]\s*(\d+)\s*", tok)
        if not tok.strip():
            continue
        sid = f"Lecture_{int(m.group(1))}__Slide{int(m.group(2))}" if m else None
        (ids if sid in valid else bad).append(sid or tok.strip())
    return set(ids), bad


A1, A2, bad = [], [], {}
for q, row in zip(df["Q"], df[col]):
    s2, b = parse(row)
    A1.append(set(GT[int(q) - 1]["relevant_slides"])); A2.append(s2)
    if b:
        bad[int(q)] = b
jac = [len(a & b) / len(a | b) if a | b else 1.0 for a, b in zip(A1, A2)]
dice = [2 * len(a & b) / (len(a) + len(b)) if a or b else 1.0 for a, b in zip(A1, A2)]
x1, x2 = [], []
for a, b in zip(A1, A2):
    for s in sorted(set().union(*[by_lec[t.split("__")[0]] for t in a | b])):
        x1.append(s in a); x2.append(s in b)
out = {"n_questions": len(A1), "invalid_entries": bad, "empty_annotator2": int(sum(not b for b in A2)),
       "relevant_per_q": {"annotator1": round(float(np.mean([len(a) for a in A1])), 2), "annotator2": round(float(np.mean([len(b) for b in A2])), 2)},
       "mean_jaccard": round(float(np.mean(jac)), 3), "mean_dice": round(float(np.mean(dice)), 3),
       "share_any_overlap": round(float(np.mean([bool(a & b) for a, b in zip(A1, A2)])), 3),
       "kappa_lecture_universe": round(float(cohen_kappa_score(x1, x2)), 3), "kappa_items": len(x1)}
if "--no-retrieval" not in sys.argv:
    for name, sets in [("annotator2", A2), ("intersection", [a & b for a, b in zip(A1, A2)]), ("union", [a | b for a, b in zip(A1, A2)])]:
        g = [{**GT[i], "relevant_slides": sorted(s) or GT[i]["relevant_slides"][:1], "anchor_slide": GT[i]["relevant_slides"][0]} for i, s in enumerate(sets)]
        p = R / f"results/retrieval_gt_{name}.json"; json.dump(g, open(p, "w"), indent=1)
        subprocess.run([sys.executable, str(R / "scripts/retrieval_v2.py"), "--index", "deployed", "--gt", str(p), "--tag", name], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        rv = json.load(open(R / f"results/retrieval_v2_{name}.json"))
        out[f"ndcg1_{name}"] = {a: {s: round(rv["arms"][a][s]["ndcg1"]["mean"], 3) for s in rv["arms"][a]}
                                for a in ["semantic", "semantic_local", "timestamp_only", "hybrid_local_cv"]}
        out[f"cv_lambdas_{name}"] = rv["cv_chosen_lambdas"]
    out["empty_sets_filled_with_anchor"] = {n: int(sum(not s for s in sets)) for n, sets in
                                            [("annotator2", A2), ("intersection", [a & b for a, b in zip(A1, A2)])]}
json.dump(out, open(R / "results/retrieval_irr.json", "w"), indent=1); print(json.dumps(out, indent=1))
