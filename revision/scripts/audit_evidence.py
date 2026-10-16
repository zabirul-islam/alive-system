"""Audit of evaluation-protocol defects found during the TLT revision (Phase 0).

Prints verifiable evidence for defects A-E from the released eval bundle.
Run from the project root:  python3 revision/scripts/audit_evidence.py
Writes revision/results/audit_evidence.json.
"""
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / "alive-eval-bundle"
AS = ROOT / "backend" / "alive_scripts"
out = {}


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


# A: judge prompts lacked transcript
for j in ["gpt-4o", "claude-sonnet-4-5"]:
    rows = jl(B / f"raw/judge_{j}.jsonl")
    pat = re.compile(r"missing|not provided|no lecture|cannot be verified|without (the )?(lecture|transcript)", re.I)
    hits = sum(bool(pat.search(str((r.get(k) or {}).get("reason", "")))) for r in rows for k in ("base_scores", "fine_scores"))
    out[f"A_judge_{j}"] = {"rows": len(rows), "reasons_flagging_missing_context": hits,
                           "md5": hashlib.md5(open(B / f"raw/judge_{j}.jsonl", "rb").read()).hexdigest()}

# B: pause time == midpoint of first relevant slide on 30 s grid
gt = json.load(open(B / "raw/retrieval_gt.json"))
ident = sum(q["pause_time"] == (int(re.search(r"Slide(\d+)", q["relevant_slides"][0]).group(1)) - 1) * 30 + 15 for q in gt)
multi = sum(len({r.split("__")[0] for r in q["relevant_slides"]}) > 1 for q in gt)
chunks = json.load(open(ROOT / "backend/rag_store/chunks.json"))
grid = sum(abs(c["start"] - (int(re.search(r"Slide(\d+)", c.get("slide_id", c.get("id", ""))).group(1)) - 1) * 30) < 1e-6
           for c in chunks if re.search(r"Slide(\d+)", c.get("slide_id", c.get("id", ""))))
out["B_retrieval"] = {"n_questions": len(gt), "pause_eq_first_gold_mid": ident,
                      "multi_lecture_gold": multi, "chunks": len(chunks), "chunks_on_30s_grid": grid}

# C: "gold" context provenance
enr = jl(B / "data/test_full_enriched.jsonl")
prov = [r.get("slide_text_provenance", "") for r in enr]
sc = [r.get("slide_text_nn_score") for r in enr if r.get("slide_text_nn_score") is not None]
tf = jl(B / "data/test_full.jsonl")
out["C_gold_context"] = {"n": len(enr), "nn_provenance": sum(p.startswith("st_nn") for p in prov),
                         "nn_score_lt_0.4": sum(s < 0.4 for s in sc), "nn_score_lt_0.5": sum(s < 0.5 for s in sc),
                         "empty_slide_text_in_test_full": sum(not r["slide_text"] for r in tf)}

# D: training inputs carry no slide context
train = jl(AS / "data/finetune/train.jsonl")
out["D_training_input"] = {"n_train": len(train), "distinct_inputs": len({r["input"] for r in train}),
                           "example_input": train[0]["input"]}

# E: train/test near-duplicates
test = jl(B / "data/test.jsonl")
exact_ans = len({r["output"].strip() for r in test} & {r["output"].strip() for r in train})
from sentence_transformers import SentenceTransformer  # noqa: E402
import numpy as np  # noqa: E402
m = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")
et = m.encode([r["instruction"] for r in test], normalize_embeddings=True, batch_size=64)
er = m.encode([r["instruction"] for r in train], normalize_embeddings=True, batch_size=64)
mx = (et @ er.T).max(1)
np.save(ROOT / "revision/results/test_max_train_cos.npy", mx)
out["E_leakage"] = {"n_test": len(test), "cos_ge_0.90": int((mx >= .90).sum()), "cos_ge_0.95": int((mx >= .95).sum()),
                    "exact_question": int((mx >= .9999).sum()), "exact_answer_overlap": exact_ans}

json.dump(out, open(ROOT / "revision/results/audit_evidence.json", "w"), indent=2)
print(json.dumps(out, indent=2))
