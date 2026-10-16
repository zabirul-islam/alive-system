"""E4: independently generated cross-segment questions (R2 multi-hop; AE self-distillation).
Generator: claude-opus-4-8 (not Llama; different model from both judges). Slides 1-2 (title/agenda) and slides <60 words excluded. 30 within-lecture pairs (slides >=10 min apart,
most similar such pair per sampled lecture) + 30 cross-lecture pairs (most similar slide pairs from different lectures,
one pair per lecture pair). Each question must need both slides; the reference answer uses only the two slide texts.
Contexts: (a) corrected re-ranker with the pause inside the later slide (lambda=0.05), (b) semantic top-4 over all lectures.
Output: revision/results/e4_items.jsonl, revision/results/contexts_e4.jsonl
"""
import json, re, subprocess, time
from collections import Counter
from itertools import combinations
from pathlib import Path
import numpy as np
import anthropic
from sentence_transformers import SentenceTransformer

R = Path(__file__).resolve().parents[1]; ROOT = R.parent
TIM = json.load(open(R / "results/slide_timings.json")); QUAL = json.load(open(R / "results/alignment_quality.json"))
dur = {int(k.split("_")[1]): v["duration_s"] for k, v in QUAL.items()}
chunks = json.load(open(ROOT / "backend/rag_store/chunks.json")); XB = np.load(R / "results/xb_deployed.npy")
sid = np.array([c["slide_id"] for c in chunks]); lec = np.array([c["lecture_num"] for c in chunks])
text = {c["slide_id"]: c["text"] for c in chunks}
mid = np.array([np.mean(TIM[s]) if s in TIM else np.nan for s in sid])
ok = np.array([s in TIM and len(text[s].split()) >= 60 and int(s.split("Slide")[-1]) > 2 for s in sid])  # no title/agenda slides
S = XB @ XB.T; rng = np.random.default_rng(42); pairs = []
# within-lecture
for L in rng.permutation(sorted(set(lec)))[:23].tolist() + rng.choice(sorted(set(lec)), 7, replace=False).tolist():
    idx = [i for i in np.where((lec == L) & ok)[0]]
    cand = [(S[a, b], a, b) for a, b in combinations(idx, 2) if abs(mid[a] - mid[b]) >= 600 and (a, b) not in [(p[1], p[2]) for p in pairs]]
    cand.sort(reverse=True); sim, a, b = cand[len([p for p in pairs if lec[p[1]] == L])]
    pairs.append(("within", a, b) if mid[a] < mid[b] else ("within", b, a))
# cross-lecture
cross = []
for la, lb in combinations(sorted(set(lec)), 2):
    ia = np.where((lec == la) & ok)[0]; ib = np.where((lec == lb) & ok)[0]
    sub = S[np.ix_(ia, ib)]; k = np.unravel_index(sub.argmax(), sub.shape)
    cross.append((float(sub[k]), int(ia[k[0]]), int(ib[k[1]])))
cross.sort(reverse=True); used = Counter()
for sim, a, b in cross:  # most similar pairs first; each lecture in at most 3 cross pairs
    if len([p for p in pairs if p[0] == "cross"]) == 30:
        break
    if used[lec[a]] >= 3 or used[lec[b]] >= 3:
        continue
    used.update([lec[a], lec[b]]); pairs.append(("cross", a, b))

key = subprocess.run(["security", "find-generic-password", "-s", "ANTHROPIC_API_KEY_1", "-w"], capture_output=True, text=True).stdout.strip()
client = anthropic.Anthropic(api_key=key)
PROMPT = """Below are the narration texts of two slides from an undergraduate medical imaging course.

[SLIDE A — {la}]
{ta}

[SLIDE B — {lb}]
{tb}

Write ONE question a student might ask that can only be answered well by combining information from BOTH slides
(for example comparing, relating, or applying an idea from one slide to the other). Phrase it the way a student would
ask while watching the lecture: do not mention "Slide A", "Slide B", "the slides", or lecture numbers. Then write a reference answer of
2-3 sentences that uses only information in these two slides. Return JSON only:
{{"question": "...", "reference": "..."}}"""
items = []
for n, (kind, a, b) in enumerate(pairs):
    p = PROMPT.format(la=sid[a].replace("__", " "), ta=text[sid[a]][:2500], lb=sid[b].replace("__", " "), tb=text[sid[b]][:2500])
    for t in range(6):
        try:
            r = client.messages.create(model="claude-opus-4-8", max_tokens=400, messages=[{"role": "user", "content": p}])
            j = json.loads(re.search(r"\{.*\}", r.content[0].text, re.S).group(0)); break
        except Exception as e:
            print("retry", n, t, str(e)[:120], flush=True); time.sleep(5 * 2 ** t); j = None
    if j:
        items.append({"i": len(items), "kind": kind, "slide_a": sid[a], "slide_b": sid[b], "question": j["question"], "reference": j["reference"],
                      "lecture": f"Lecture {int(lec[b])}", "difficulty": "cross-segment", "pause_slide": sid[b]})
with open(R / "results/e4_items.jsonl", "w") as f:
    for it in items:
        f.write(json.dumps(it) + "\n")
# contexts
m = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")
Q = m.encode([it["question"] for it in items], normalize_embeddings=True); rng = np.random.default_rng(42)
with open(R / "results/contexts_e4.jsonl", "w") as f:
    for it, q in zip(items, Q):
        L = int(it["lecture"].split()[-1]); st, en = TIM[it["pause_slide"]]; t = rng.uniform(st, max(en, st + 1e-6))
        d = np.nan_to_num(np.where(lec == L, np.abs(mid - t), dur[L]), nan=dur[L])
        e2e = np.argsort(-(XB @ q - 0.05 * d / 60))[:4]; sem = np.argsort(-(XB @ q))[:4]
        f.write(json.dumps({**it, "ctx_lecture": "\n\n".join(text[sid[j]] for j in e2e), "ctx_retrieved": "\n\n".join(text[sid[j]] for j in sem),
                            "e2e_sources": list(sid[e2e]), "sem_sources": list(sid[sem]),
                            "grounding": text[it["slide_a"]] + "\n\n" + text[it["slide_b"]]}) + "\n")
print("items", len(items), {k: sum(it["kind"] == k for it in items) for k in ["within", "cross"]})
