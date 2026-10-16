"""E3 end-to-end contexts: corrected re-ranker (Eq. 1, lambda=0.05, lecture-local, aligned timings) with a pause
drawn uniformly inside each test question's inferred source slide (seed 42); top-4 segments joined as context.
Grounding for the judges stays the 2x2 excerpts (lecture-restricted + semantic top-4) so scores are comparable.
Output: revision/results/contexts_e3.jsonl  (fields as contexts_2x2.jsonl; ctx_lecture = end-to-end context)
"""
import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

R = Path(__file__).resolve().parents[1]; ROOT = R.parent
C = [json.loads(l) for l in open(R / "results/contexts_2x2.jsonl")]
recs = [r for r in json.load(open(R / "results/splits/record_slides.json")) if r["orig"] == "test"]
slide_of = {r["question"]: r["slide"] for r in recs}
TIM = json.load(open(R / "results/slide_timings.json")); QUAL = json.load(open(R / "results/alignment_quality.json"))
dur = {int(k.split("_")[1]): v["duration_s"] for k, v in QUAL.items()}
chunks = json.load(open(ROOT / "backend/rag_store/chunks.json")); XB = np.load(R / "results/xb_deployed.npy")
sid = np.array([c["slide_id"] for c in chunks]); lec = np.array([c["lecture_num"] for c in chunks])
mid = np.array([np.mean(TIM[s]) if s in TIM else np.nan for s in sid])
m = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")
Q = m.encode([c["question"] for c in C], normalize_embeddings=True)
rng = np.random.default_rng(42); LAM = 0.05; out = []; hit = 0; missing = 0
for c, q in zip(C, Q):
    s = slide_of.get(c["question"].strip())
    if s is None or s not in TIM:
        missing += 1; s = None
    L = int(c["lecture"].split()[-1])
    st, en = TIM[s] if s else (0.0, dur[L])
    t = rng.uniform(st, max(en, st + 1e-6))
    d = np.nan_to_num(np.where(lec == L, np.abs(mid - t), dur[L]), nan=dur[L])
    top = np.argsort(-(XB @ q - LAM * d / 60))[:4]
    hit += int(s in set(sid[top])) if s else 0
    out.append({**{k: c[k] for k in ["i", "lecture", "difficulty", "question", "reference"]},
                "ctx_lecture": "\n\n".join(chunks[j]["text"] for j in top), "e2e_sources": list(sid[top]), "pause_time": round(float(t), 1),
                "inferred_slide": s, "grounding_lecture": c["ctx_lecture"], "grounding_retrieved": c["ctx_retrieved"]})
with open(R / "results/contexts_e3.jsonl", "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
print("items", len(out), "missing slide", missing, "inferred slide in top-4", hit)
