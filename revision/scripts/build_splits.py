"""R1-2: build slide-grouped and lecture-held-out splits of the 5,207 cleaned QA pairs.

Provenance trace: raw pair index -> generation chunk (per-chunk valid-pair counts in the final
successful run of qa_gen.log) -> slides contained in that chunk (deterministic re-chunking of the
generation-time slide texts, generate_qa.chunk_lecture_text) -> nearest slide within the chunk
(all-mpnet-base-v2, question+answer vs slide text). Pairs are grouped by that slide.
Training records keep the deployed recipe's format (instruction/input/output, context-free input).
Test records get the same "lecture-restricted retrieval" context as the original test set
(top-1 deployed-index chunk within the known lecture, question-only query).
Output: revision/results/splits/{slide_grouped,lecture_heldout}/{train,val}.jsonl, test_ctx.jsonl, stats.json
"""
import json, re
from collections import defaultdict
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
AS = ROOT / "backend/alive_scripts"
LEC = ROOT / "revision/data/as_lectures"
OUT = ROOT / "revision/results/splits"
SEED = 42


def lnum(s):
    return int(re.search(r"(\d+)", s).group(1))


# 1. per-lecture chunk counts from the final successful run
log = (AS / "data_pipeline/qa_gen.log").read_text().splitlines()
counts, cur = {}, None
for line in log:
    m = re.search(r"Processing: Lecture (\d+)", line)
    if m:
        cur, buf = int(m.group(1)), []
    m = re.search(r"Chunk (\d+): got (\d+) valid pairs", line)
    if m and cur is not None:
        buf.append((int(m.group(1)), int(m.group(2))))
    m = re.search(r"SAVED: Lecture (\d+)", line)
    if m:
        counts[int(m.group(1))] = list(buf)


# 2. re-chunk generation-time lecture text (verbatim logic from generate_qa.py)
def chunks_for(n):
    files = sorted((LEC / f"Lecture {n}/Texts").glob("*.txt"), key=lambda f: lnum(f.name))
    slides = [(f.stem, f.read_text(encoding="utf-8", errors="ignore").strip()) for f in files]
    slides = [(s, t) for s, t in slides if len(t.split()) >= 15]
    text = "\n\n".join(f"=== {s} ===\n{t}" for s, t in slides)
    out, start = [], 0
    while start < len(text):
        end = start + 5000
        b = text.rfind("=== Slide", start, end)
        if b > start + 1000:
            end = b
        out.append(text[start:end]); start = end
    res, prev = [], None
    for c in out:
        names = re.findall(r"=== (Slide\d+) ===", c)
        if not c.startswith("=== Slide") and prev:
            names = [prev] + names  # chunk opens mid-slide
        prev = names[-1] if names else prev
        res.append(names)
    return res, dict(slides)


raw_map, slide_text, n_trace_ok = {}, {}, 0
for n, cl in counts.items():
    raw = json.load(open(AS / f"data/qa_raw/Lecture_{n}.json"))["pairs"]
    chunk_slides, texts = chunks_for(n)
    for s, t in texts.items():
        slide_text[(n, s)] = t
    idx = 0
    for ci, got in cl:
        for _ in range(got):
            if idx < len(raw):
                raw_map[(n, raw[idx]["question"].strip())] = chunk_slides[ci - 1]
            idx += 1
    n_trace_ok += idx == len(raw)

# 3. load the cleaned dataset (train/val/test) with its original split label
recs = []
for split, f in [("train", "train.jsonl"), ("val", "val.jsonl")]:
    for l in open(AS / f"data/finetune/{f}"):
        r = json.loads(l)
        recs.append({"lecture": lnum(r["input"]), "question": r["instruction"].strip(), "answer": r["output"], "orig": split})
for l in open(AS / "data/finetune/test_full.jsonl"):
    r = json.loads(l)
    recs.append({"lecture": r["lecture_num"], "question": r["question"].strip(), "answer": r["answer"], "orig": "test",
                 "difficulty": r["difficulty"]})
diff = {}
for n in counts:
    for p in json.load(open(AS / f"data/qa_raw/Lecture_{n}.json"))["pairs"]:
        diff[(n, p["question"].strip())] = p.get("difficulty", "")

# 4. nearest slide within the traced chunk
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")
keys = sorted(slide_text)
S = model.encode([slide_text[k] for k in keys], normalize_embeddings=True, batch_size=32)
kidx = {k: i for i, k in enumerate(keys)}
E = model.encode([r["question"] + " " + r["answer"] for r in recs], normalize_embeddings=True, batch_size=64)
untraced = 0
for r, e in zip(recs, E):
    cand = raw_map.get((r["lecture"], r["question"]))
    if not cand:
        untraced += 1
        cand = [s for (n, s) in keys if n == r["lecture"]]
    cand = [c for c in cand if (r["lecture"], c) in kidx]
    sims = [float(S[kidx[(r["lecture"], c)]] @ e) for c in cand]
    r["slide"] = f"Lecture_{r['lecture']}__{cand[int(np.argmax(sims))]}"
    r["slide_sim"] = max(sims)
    r.setdefault("difficulty", diff.get((r["lecture"], r["question"]), ""))

# 5. lecture-restricted context for any record that may land in a test set
chunks = json.load(open(ROOT / "backend/rag_store/chunks.json"))
XB = np.load(ROOT / "revision/results/xb_deployed.npy")
Q = model.encode([r["question"] for r in recs], normalize_embeddings=True, batch_size=64)
by_lec = defaultdict(list)
for i, c in enumerate(chunks):
    by_lec[c["lecture_num"]].append(i)
for r, q in zip(recs, Q):
    ids = by_lec[r["lecture"]]; j = ids[int(np.argmax(XB[ids] @ q))]
    r["ctx_lecture"] = chunks[j]["text"]


def write(name, assign):
    d = OUT / name; d.mkdir(parents=True, exist_ok=True)
    for sp in ["train", "val"]:
        with open(d / f"{sp}.jsonl", "w") as f:
            for r, a in zip(recs, assign):
                if a == sp:
                    f.write(json.dumps({"instruction": r["question"], "input": f"LECTURE: Lecture {r['lecture']}\nCONTEXT:\nLecture {r['lecture']}",
                                        "output": r["answer"]}) + "\n")
    with open(d / "test_ctx.jsonl", "w") as f:
        for i, (r, a) in enumerate(zip(recs, assign)):
            if a == "test":
                f.write(json.dumps({"i": i, "lecture": f"Lecture {r['lecture']}", "difficulty": r["difficulty"], "question": r["question"],
                                    "reference": r["answer"], "ctx_lecture": r["ctx_lecture"], "ctx_retrieved": "", "slide": r["slide"]}) + "\n")
    te = np.array([a == "test" for a in assign]); tr = np.array([a == "train" for a in assign])
    mx = (Q[te] @ Q[tr].T).max(1)
    st = {"train": int(tr.sum()), "val": int(sum(a == "val" for a in assign)), "test": int(te.sum()),
          "test_lectures": sorted({r["lecture"] for r, a in zip(recs, assign) if a == "test"}),
          "test_slides": len({r["slide"] for r, a in zip(recs, assign) if a == "test"}),
          "slide_overlap_train_test": len({r["slide"] for r, a in zip(recs, assign) if a == "test"} & {r["slide"] for r, a in zip(recs, assign) if a == "train"}),
          "test_maxcos_ge_0.90": int((mx >= .9).sum()), "test_maxcos_ge_0.95": int((mx >= .95).sum())}
    json.dump(st, open(d / "stats.json", "w"), indent=1); return st


rng = np.random.default_rng(SEED)
# slide-grouped: whole slides to test/val, ~10% each, within every lecture
assign = []
grp = defaultdict(list)
for i, r in enumerate(recs):
    grp[(r["lecture"], r["slide"])].append(i)
lab = {}
for n in sorted({r["lecture"] for r in recs}):
    gs = [g for g in grp if g[0] == n]; rng.shuffle(gs)
    tot = sum(len(grp[g]) for g in gs); acc = 0
    for g in gs:
        lab[g] = "test" if acc < 0.1 * tot else ("val" if acc < 0.2 * tot else "train"); acc += len(grp[g])
sg = write("slide_grouped", [lab[(r["lecture"], r["slide"])] for r in recs])

# lecture-held-out: 5 lectures to test, 10% of remaining pairs (slide-grouped) to val
lecs = sorted({r["lecture"] for r in recs}); held = set(rng.choice(lecs, 5, replace=False).tolist())
lab2 = {}
for n in lecs:
    if n in held:
        continue
    gs = [g for g in grp if g[0] == n]; rng.shuffle(gs); tot = sum(len(grp[g]) for g in gs); acc = 0
    for g in gs:
        lab2[g] = "val" if acc < 0.1 * tot else "train"; acc += len(grp[g])
lh = write("lecture_heldout", ["test" if r["lecture"] in held else lab2[(r["lecture"], r["slide"])] for r in recs])

stats = {"n_records": len(recs), "lectures_traced_exactly": n_trace_ok, "lectures": len(counts), "untraced_records": untraced,
         "mean_within_chunk_slide_sim": round(float(np.mean([r["slide_sim"] for r in recs])), 3),
         "distinct_slides": len(grp), "slide_grouped": sg, "lecture_heldout": lh}
json.dump(stats, open(OUT / "stats.json", "w"), indent=1)
print(json.dumps(stats, indent=1))

# per-record inferred slide (used by the end-to-end experiment E3)
json.dump([{"orig": r["orig"], "lecture": r["lecture"], "question": r["question"], "slide": r["slide"], "slide_sim": round(r["slide_sim"], 4)}
           for r in recs], open(OUT / "record_slides.json", "w"))
