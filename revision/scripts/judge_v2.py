"""R1-3/R1-4: re-run the external LLM judges with the lecture excerpts actually present.

Fixes defect A (original judge prompts carried an empty transcript). Rubric is the original F/P/G/A
(scripts/eval_judge.py, MIMIC) plus E = primary error type. Every candidate is judged on its own,
with the SAME grounding excerpt for all four 2x2 cells: the lecture-restricted chunk followed by the
retrieved top-4 chunks (de-duplicated), so F is comparable across cells.
Usage: python judge_v2.py --judge gpt-4o|claude-sonnet-4-5 --contexts C --gen base=G1 --gen alive=G2 --out O [--max N]
Anthropic key is read from the macOS keychain item ANTHROPIC_API_KEY_1 when ANTHROPIC_API_KEY is unset.
"""
import argparse, json, os, re, subprocess, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

INSTR = """You are evaluating a teaching assistant's answer to a student question
in an undergraduate medical imaging course. Score the answer on four axes
using a 1-5 integer scale:

  F = Factuality vs. lecture transcript (5=fully grounded, 1=fabricated)
  P = Pedagogical clarity (5=clear scaffold, 1=confusing or wrong intuition)
  G = Conciseness & length fit (5=18-32 words, complete; 1=verbose hedging or too short)
  A = Agreement with reference answer (5=substantively equivalent, 1=contradicts)

Also classify the most serious error in the answer, judged against the lecture transcript
and the reference answer:
  E = one of "none", "conceptual" (wrong concept, definition or relationship),
      "mathematical" (wrong formula, derivation or quantitative relation),
      "causal" (wrong cause/effect or mechanism), "numerical" (wrong number or unit),
      "other" (e.g., off-topic, unanswered, contradicts itself)

Return ONLY a JSON object of the form:
  {"F": <int>, "P": <int>, "G": <int>, "A": <int>, "E": "<label>", "reason": "<one sentence>"}
No preamble, no markdown fences."""

PROMPT = """[LECTURE_TRANSCRIPT (excerpt for grounding)]
{transcript}

[QUESTION]
{question}

[REFERENCE_ANSWER]
{reference}

[CANDIDATE_ANSWER]
{candidate}

Score the CANDIDATE_ANSWER per the instructions. Return JSON only."""


def client_call(judge):
    if judge.startswith("gpt"):
        from openai import OpenAI
        c = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url="https://api.openai.com/v1")
        return lambda p: c.chat.completions.create(model=judge, temperature=0.0, max_tokens=250, messages=[
            {"role": "system", "content": INSTR}, {"role": "user", "content": p}]).choices[0].message.content
    import anthropic
    key = os.environ.get("ANTHROPIC_API_KEY") or subprocess.run(
        ["security", "find-generic-password", "-s", "ANTHROPIC_API_KEY_1", "-w"], capture_output=True, text=True).stdout.strip()
    c = anthropic.Anthropic(api_key=key)
    return lambda p: c.messages.create(model=judge, system=INSTR, temperature=0.0, max_tokens=250,
                                       messages=[{"role": "user", "content": p}]).content[0].text


def parse(s):
    s = re.sub(r"```(?:json)?", "", s).strip("` \n"); m = re.search(r"\{.*\}", s, re.S)
    try:
        return json.loads(m.group(0) if m else s)
    except Exception:
        return {"F": None, "P": None, "G": None, "A": None, "E": None, "reason": f"unparseable: {s[:120]}"}


ap = argparse.ArgumentParser()
ap.add_argument("--judge", required=True); ap.add_argument("--contexts", required=True)
ap.add_argument("--gen", action="append", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--max", type=int); ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--conds", default="lecture,retrieved")
a = ap.parse_args()
C = [json.loads(l) for l in open(a.contexts)][: a.max]
G = {k: {r["i"]: r for r in map(json.loads, open(v))} for k, v in (g.split("=", 1) for g in a.gen)}
call = client_call(a.judge)
done = set()
if Path(a.out).exists():
    done = {(r["i"], r["cell"]) for r in map(json.loads, open(a.out)) if r["F"] is not None}  # failed calls are retried
jobs = []
for c in C:
    parts = [c["ctx_lecture"]] + ([c["ctx_retrieved"]] if c.get("ctx_retrieved") and c["ctx_retrieved"] != c["ctx_lecture"] else [])
    transcript = "\n\n".join(parts)[:6000]
    for m in G:
        for k in a.conds.split(","):
            cell = f"{m}|{k}"
            if (c["i"], cell) not in done:
                jobs.append((c["i"], cell, PROMPT.format(transcript=transcript, question=c["question"], reference=c["reference"],
                                                         candidate=G[m][c["i"]][f"pred_{k}"])))


def run(job):
    i, cell, p = job
    for t in range(4):
        try:
            return {"i": i, "cell": cell, "judge": a.judge, **parse(call(p))}
        except Exception as e:  # rate limits / transient errors
            time.sleep(2 ** t * 3); err = str(e)[:200]
    return {"i": i, "cell": cell, "judge": a.judge, "F": None, "P": None, "G": None, "A": None, "E": None, "reason": f"API failure: {err}"}


with open(a.out, "a") as f, ThreadPoolExecutor(a.workers) as ex:
    for n, r in enumerate(ex.map(run, jobs), 1):
        f.write(json.dumps(r) + "\n"); f.flush()
        if n % 200 == 0:
            print(n, "/", len(jobs), flush=True)
print("done", len(jobs))
