"""Align per-slide narration text to lecture SRT cues -> real slide start/end times.

Monotonic DP: each SRT cue is assigned to one slide, slide index non-decreasing,
maximising summed token overlap (content words, len>3). Slides with no cues get a
zero-length window at the previous slide's end.
Output: revision/results/slide_timings.json  {"Lecture_N__SlideK": [start, end], ...}
        revision/results/alignment_quality.json
"""
import json, re
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parents[1]
LEC, SRT = R / "data/lectures", R / "data/srt"
tok = lambda s: {w for w in re.findall(r"[a-z]+", s.lower()) if len(w) > 3}


def ts(s):
    h, m, rest = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(rest.replace(",", "."))


def read_srt(p):
    blocks = re.split(r"\n\s*\n", p.read_text(encoding="utf-8", errors="ignore").strip())
    cues = []
    for b in blocks:
        m = re.search(r"(\d+:\d+:\d+[,.]\d+)\s*-->\s*(\d+:\d+:\d+[,.]\d+)\s*\n(.*)", b, re.S)
        if m:
            cues.append((ts(m.group(1)), ts(m.group(2)), m.group(3).replace("\n", " ")))
    return cues


def align(slides, cues):
    S = [tok(t) for _, t in slides]
    C = [tok(c[2]) for c in cues]
    n, m = len(C), len(S)
    sim = np.array([[len(c & s) / max(len(c), 1) for s in S] for c in C])
    f = np.full((n, m), -1e9); back = np.zeros((n, m), int)
    f[0] = sim[0]
    for i in range(1, n):
        best, arg = -1e9, 0
        for j in range(m):  # prefix max over j' <= j (allows skipping empty slides)
            if f[i - 1, j] > best:
                best, arg = f[i - 1, j], j
            f[i, j] = sim[i, j] + best; back[i, j] = arg
    j = int(f[-1].argmax()); assign = [0] * n
    for i in range(n - 1, -1, -1):
        assign[i] = j; j = back[i, j]
    return assign, float(np.mean([sim[i, a] for i, a in enumerate(assign)]))


def main():
    timings, quality = {}, {}
    for d in sorted(LEC.iterdir(), key=lambda p: int(p.name.split()[-1])):
        n = int(d.name.split()[-1])
        srt = SRT / f"lecture{n:02d}.srt"
        files = sorted((d / "Texts").glob("Slide*.txt"), key=lambda p: int(re.search(r"\d+", p.stem).group()))
        slides = [(int(re.search(r"\d+", p.stem).group()), p.read_text(errors="ignore")) for p in files]
        cues = read_srt(srt)
        assign, q = align(slides, cues)
        prev_end = 0.0
        for j, (k, _) in enumerate(slides):
            idx = [i for i, a in enumerate(assign) if a == j]
            st, en = (cues[idx[0]][0], cues[idx[-1]][1]) if idx else (prev_end, prev_end)
            timings[f"Lecture_{n}__Slide{k}"] = [st, en]; prev_end = en
        quality[f"Lecture_{n}"] = {"slides": len(slides), "cues": len(cues), "mean_cue_overlap": round(q, 3),
                                   "slides_without_cues": sum(1 for j in range(len(slides)) if j not in set(assign)),
                                   "duration_s": cues[-1][1]}

    out = R / "results"
    json.dump(timings, open(out / "slide_timings.json", "w"), indent=1)
    json.dump(quality, open(out / "alignment_quality.json", "w"), indent=1)
    qs = [v["mean_cue_overlap"] for v in quality.values()]
    print(f"lectures={len(quality)} slides={len(timings)} mean_overlap={np.mean(qs):.3f} min={min(qs):.3f} "
          f"empty_slides={sum(v['slides_without_cues'] for v in quality.values())}")
    for k, v in quality.items():
        print(k, v)


if __name__ == "__main__":
    main()
