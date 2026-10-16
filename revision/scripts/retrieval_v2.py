"""R1-5 retrieval re-evaluation, implementing revision/PROTOCOL.md (frozen 2026-10-14).

Step 0 reproduces the original (circular) protocol, then runs the target-conditioned
simulation with real SRT-aligned slide timings, all arms on one index, nested-CV lambda.
Output: revision/results/retrieval_v2.json, revision/results/retrieval_v2_lambda_curve.json
Usage:  python revision/scripts/retrieval_v2.py [--index deployed|textbook]
"""
import json, math, re, sys
from pathlib import Path
import numpy as np
import subprocess
from sentence_transformers import SentenceTransformer
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "revision/results"
which = sys.argv[sys.argv.index("--index") + 1] if "--index" in sys.argv else "deployed"
STORE = ROOT / ("backend/rag_store" if which == "deployed" else "backend/alive_scripts/rag_store")
GT = json.load(open(sys.argv[sys.argv.index("--gt") + 1] if "--gt" in sys.argv else ROOT / "alive-eval-bundle/raw/retrieval_gt.json"))
TAG = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else None
TIM = json.load(open(RES / "slide_timings.json"))
QUAL = json.load(open(RES / "alignment_quality.json"))
LAMBDAS = [0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
SCEN = {"during": 0, "lag1": 60, "lag3": 180, "lag5": 300}
STRESS = "random"  # added after external review: pause uniform over the whole current lecture (uninformative timing)
SEEDS = range(5)
K = 5

chunks = json.load(open(STORE / "chunks.json"))
XB = RES / f"xb_{which}.npy"
if not XB.exists():  # ponytail: faiss and torch segfault in one process on macOS -> dump vectors in a subprocess
    subprocess.run([sys.executable, "-c", "import faiss,numpy as np,sys;i=faiss.read_index(sys.argv[1]);"
                    "np.save(sys.argv[2],faiss.rev_swig_ptr(i.get_xb(),i.ntotal*i.d).reshape(i.ntotal,i.d).copy())",
                    str(STORE / "faiss.index"), str(XB)], check=True)
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")
Q = model.encode([q["question"] for q in GT], normalize_embeddings=True).astype("float32")
COS = Q @ np.load(XB).T  # flat inner-product index == exact cosine

sid = np.array([c.get("slide_id", "") for c in chunks])
lec = np.array([c.get("lecture_num", -1) if c.get("source", "lecture") != "book" and c.get("slide_id") else -1 for c in chunks])
boost = np.array([c.get("source_boost", 0.0) for c in chunks])
grid_mid = np.array([(c["start"] + c["end"]) / 2 if c.get("start", -1) >= 0 else np.nan for c in chunks])
real_mid = np.array([np.mean(TIM[s]) if s in TIM else np.nan for s in sid])
dur = {int(k.split("_")[1]): v["duration_s"] for k, v in QUAL.items()}


def ndcg(ret, rel, k, standard=True):
    """standard: IDCG from the full annotated relevance set (binary); standard=False reproduces the original
    eval_retrieval.py, which normalised by the ideal ordering of the retrieved list only."""
    r = [1 if s in rel else 0 for s in ret[:k]]
    dcg = lambda v: sum(x / math.log2(i + 2) for i, x in enumerate(v))
    i = dcg([1] * min(len(rel), k)) if standard else dcg(sorted(r, reverse=True))
    return dcg(r) / i if i > 0 else 0.0


def metrics(ret, rel, standard=True):
    return {"ndcg1": ndcg(ret, rel, 1, standard), "ndcg3": ndcg(ret, rel, 3, standard), "ndcg5": ndcg(ret, rel, 5, standard),
            "hit1": float(ret[0] in rel), "hit5": float(any(s in rel for s in ret[:5]))}


def top(scores, k=K):
    return list(sid[np.argsort(-scores, kind="stable")[:k]])


def rank_semantic(q, t):
    return top(COS[q])


def rank_timestamp_only(q, t, L):
    s = np.where(lec == L, -np.abs(real_mid - t) * 1e3 + COS[q], -1e9)  # distance, cosine tie-break
    return top(np.nan_to_num(s, nan=-1e9))


def rank_hybrid_deployed(q, t, mid, lam=0.4):  # exact app.py: semantic top-16 pool, no lecture filter
    pool = np.argsort(-COS[q])[:16]
    adj = COS[q, pool] + boost[pool] - np.where(np.isnan(mid[pool]), 0, lam * np.abs(mid[pool] - t) / 60)
    return list(sid[pool[np.argsort(-adj, kind="stable")][:K]])


def rank_hybrid_local(q, t, L, lam):
    d = np.where(lec == L, np.abs(real_mid - t), dur[L])
    d = np.nan_to_num(d, nan=dur[L])
    return top(COS[q] + boost - lam * d / 60)


def target(q):
    s = GT[q].get("anchor_slide", GT[q]["relevant_slides"][0])  # pause anchor fixed when relevance labels vary
    return s, int(re.search(r"Lecture_(\d+)", s).group(1))


def pause_times(q, scen):
    s, L = target(q)
    st, en = TIM[s]
    out = []
    for seed in SEEDS:
        rng = np.random.default_rng(1000 * q + seed)
        if scen == STRESS:
            out.append(rng.uniform(0, dur[L])); continue
        out.append(min(rng.uniform(st, max(en, st + 1e-6)) + SCEN[scen], dur[L]))
    return out


def rank_semantic_local(q, t, L):  # added after external review: semantic ranking restricted to the current lecture
    return top(np.where(lec == L, COS[q], -1e9))


def evaluate(rank_fn, scens=None):
    """per-question metrics averaged over seeds, per scenario"""
    res = {}
    for scen in (scens or list(SCEN) + [STRESS]):
        rows = []
        for q in range(len(GT)):
            rel, (_, L) = set(GT[q]["relevant_slides"]), target(q)
            ms = [metrics(rank_fn(q, t, L), rel) for t in pause_times(q, scen)]
            rows.append({k: float(np.mean([m[k] for m in ms])) for k in ms[0]})
        res[scen] = rows
    return res


def summarize(rows, key="ndcg1", B=2000):
    v = np.array([r[key] for r in rows]); rng = np.random.default_rng(0)
    bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(B)]
    return {"mean": round(float(v.mean()), 4), "ci": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]}


out = {"index": which, "n_chunks": len(chunks), "n_questions": len(GT)}

# Step 0: reproduce original circular protocol (synthetic grid, pause = gold-slide midpoint, top-15 pool)
orig = {"semantic": [], "timestamp": []}
for q in range(len(GT)):
    rel, t = set(GT[q]["relevant_slides"]), GT[q]["pause_time"]
    orig["semantic"].append(metrics(top(COS[q]), rel, standard=False))
    pool = np.argsort(-COS[q])[:15]
    adj = COS[q, pool] - 0.4 * np.abs(np.nan_to_num(grid_mid[pool]) - t) / 60
    orig["timestamp"].append(metrics(list(sid[pool[np.argsort(-adj, kind="stable")][:K]]), rel, standard=False))
out["original_protocol"] = {k: {m: summarize(v, m)["mean"] for m in v[0]} for k, v in orig.items()}
print("original protocol:", out["original_protocol"])

# Step 1: arms under the target-conditioned simulation
arms = {
    "semantic": evaluate(lambda q, t, L: rank_semantic(q, t)),
    "semantic_local": evaluate(rank_semantic_local),
    "timestamp_only": evaluate(rank_timestamp_only),
    "hybrid_deployed_gridtimes": evaluate(lambda q, t, L: rank_hybrid_deployed(q, t, grid_mid)),
    "hybrid_deployed_realtimes": evaluate(lambda q, t, L: rank_hybrid_deployed(q, t, real_mid)),
}
curve = {lam: evaluate(lambda q, t, L, lam=lam: rank_hybrid_local(q, t, L, lam)) for lam in LAMBDAS}

# Step 2: nested 5-fold CV for lambda (single lambda across scenarios)
folds = np.random.default_rng(0).permutation(len(GT)) % 5
cv_rows = {s: [None] * len(GT) for s in list(SCEN) + [STRESS]}; chosen = []
for f in range(5):
    tr = np.where(folds != f)[0]; te = np.where(folds == f)[0]
    best = max(LAMBDAS, key=lambda lam: np.mean([curve[lam][s][q]["ndcg1"] for s in SCEN for q in tr]))
    chosen.append(best)
    for s in list(SCEN) + [STRESS]:
        for q in te:
            cv_rows[s][q] = curve[best][s][q]
arms["hybrid_local_cv"] = cv_rows
out["cv_chosen_lambdas"] = chosen

out["arms"] = {a: {s: {m: summarize(r[s], m) for m in r[s][0]} for s in list(SCEN) + [STRESS]} for a, r in arms.items()}
tests = {}
for s in SCEN:
    for ref in ["semantic", "semantic_local", "timestamp_only"]:
        a = np.array([r["ndcg1"] for r in arms["hybrid_local_cv"][s]]); b = np.array([r["ndcg1"] for r in arms[ref][s]])
        p = float(wilcoxon(a, b).pvalue) if np.any(a != b) else 1.0
        tests[f"{s}:hybrid_local_cv_vs_{ref}"] = {"delta": round(float((a - b).mean()), 4), "p": p}
ps = sorted(tests.items(), key=lambda kv: kv[1]["p"]); m = len(ps); run = 0
for i, (k, v) in enumerate(ps):
    run = max(run, min(1.0, (m - i) * v["p"])); tests[k]["p_holm"] = run
out["tests_ndcg1"] = tests
out["targets_in_low_quality_lecture3"] = sum(target(q)[1] == 3 for q in range(len(GT)))

suffix = ("" if which == "deployed" else "_textbook") + (f"_{TAG}" if TAG else "")
json.dump(out, open(RES / f"retrieval_v2{suffix}.json", "w"), indent=1)
json.dump({str(l): {s: summarize(curve[l][s])["mean"] for s in list(SCEN) + [STRESS]} for l in LAMBDAS},
          open(RES / f"retrieval_v2_lambda_curve{suffix}.json", "w"), indent=1)
for a in out["arms"]:
    print(f"{a:28s}", "  ".join(f"{s}:{out['arms'][a][s]['ndcg1']['mean']:.3f}" for s in list(SCEN) + [STRESS]))
print("cv lambdas", chosen)
for k, v in tests.items():
    print(k, v)
