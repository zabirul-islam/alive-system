"""E6 (R2 portability): derive slide timings from raw Whisper ASR segments (no avatar-service subtitles) with the
same monotonic aligner, and compare them with the SRT-derived timings used in the paper.
Reports per lecture: mean token overlap, slide start-time differences (mean/median |delta|, share within 10 s / 30 s).
Output: revision/results/e6_portability.json
"""
import json, sys
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "scripts"))
from align_slides import align, LEC  # noqa: E402

TIM = json.load(open(R / "results/slide_timings.json")); out = {}
for n in [2, 13, 19]:
    asr = json.load(open(R / f"results/e6/asr_lecture{n:02d}.json"))
    cues = [(c["start"], c["end"], c["text"]) for c in asr]
    files = sorted((LEC / f"Lecture {n}/Texts").glob("Slide*.txt"), key=lambda p: int("".join(ch for ch in p.stem if ch.isdigit())))
    slides = [(int("".join(ch for ch in p.stem if ch.isdigit())), p.read_text(errors="ignore")) for p in files]
    assign, q = align(slides, cues)
    d = []; prev = 0.0
    for j, (k, _) in enumerate(slides):
        idx = [i for i, a in enumerate(assign) if a == j]
        st = cues[idx[0]][0] if idx else prev
        prev = cues[idx[-1]][1] if idx else prev
        ref = TIM.get(f"Lecture_{n}__Slide{k}")
        if ref and idx:
            d.append(abs(st - ref[0]))
    d = np.array(d)
    out[f"Lecture_{n}"] = {"asr_segments": len(cues), "slides": len(slides), "mean_token_overlap": round(q, 3),
                           "mean_abs_start_diff_s": round(float(d.mean()), 1), "median_abs_start_diff_s": round(float(np.median(d)), 1),
                           "share_within_10s": round(float((d <= 10).mean()), 3), "share_within_30s": round(float((d <= 30).mean()), 3)}
allv = {k: round(float(np.mean([v[k] for v in out.values()])), 3) for k in ["mean_token_overlap", "share_within_10s", "share_within_30s"]}
out["mean_over_lectures"] = allv
json.dump(out, open(R / "results/e6_portability.json", "w"), indent=1); print(json.dumps(out, indent=1))
