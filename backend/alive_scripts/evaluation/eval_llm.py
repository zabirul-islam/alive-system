"""
eval_llm_patched.py
===================
Patched version of `evaluation/eval_llm.py` for the MIMIC paper revision.

Two additions vs. the original (everything else byte-identical so that
re-running on the same model/seed reproduces the existing
`results_base.json` and `results_finetuned.json` BERTScore to within
±0.001):

  1. Loads `data/finetune/test_full_enriched.jsonl` if present
     (produced by `scripts/enrich_test_set.py`); otherwise falls back
     to the original `test_full.jsonl`. Enrichment fills the previously
     empty `slide_text` field, which makes the NLI factual-consistency
     metric meaningful.
  2. Adds a regex-based slide-grounded factual-error metric (matches
     paper Eq. (10): an answer is flagged if it contains a numerical
     quantity with an imaging unit that does not appear in the source
     transcript).
  3. Persists raw model predictions and the slide_texts used into the
     output JSON so that downstream evaluations (RAG comparison, LLM-
     as-judge) can run without regenerating predictions.

DEPLOY
------
On H100:
    cp evaluation/eval_llm.py evaluation/eval_llm.py.bak  # keep original
    scp local:scripts/patches/eval_llm_patched.py \\
        h100:~/islamm11/alive/backend/alive_scripts/evaluation/eval_llm.py

Then run as before:
    python evaluation/eval_llm.py --model base
    python evaluation/eval_llm.py --model finetuned
"""

import json
import time
import argparse
import re
import requests
import statistics
from pathlib import Path
from collections import defaultdict

# ── CONFIG ────────────────────────────────────────────────────
VLLM_URL      = "http://127.0.0.1:8000/v1/completions"
BASE_MODEL    = "meta-llama/Meta-Llama-3.1-8B-Instruct"
FT_MODEL      = "checkpoints/alive-llama-lora/merged"

TEST_FILE_ENRICHED = Path("data/finetune/test_full_enriched.jsonl")
TEST_FILE_ORIGINAL = Path("data/finetune/test_full.jsonl")

EVAL_DIR      = Path("evaluation")
EVAL_DIR.mkdir(exist_ok=True)

MAX_EXAMPLES  = None   # None = all 508


# ── PROMPT ────────────────────────────────────────────────────
def build_prompt(question: str, slide_text: str, model_name: str) -> str:
    """Same prompt format as app.py's build_prompt() — DO NOT change."""
    system = (
        "You are a teaching assistant answering a student. "
        "Use ONLY the lecture context provided. "
        "Answer in 1-3 clear, precise sentences."
    )
    user = (
        f"LECTURE CONTEXT:\n{slide_text}\n\n"
        f"QUESTION:\n{question}\n\nANSWER:"
    )
    return (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n{system}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n{user}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
    )


def call_llm(question: str, slide_text: str, model_name: str) -> str:
    prompt = build_prompt(question, slide_text, model_name)
    r = requests.post(
        VLLM_URL,
        json={
            "model":       model_name,
            "prompt":      prompt,
            "temperature": 0.0,
            "max_tokens":  200,
            "stop":        ["<|eot_id|>", "<|end_of_text|>", "\n\n"],
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["text"].strip()


# ── LOAD TEST SET ─────────────────────────────────────────────
def load_test_set():
    test_file = TEST_FILE_ENRICHED if TEST_FILE_ENRICHED.exists() else TEST_FILE_ORIGINAL
    if not test_file.exists():
        raise FileNotFoundError(
            f"Neither {TEST_FILE_ENRICHED} nor {TEST_FILE_ORIGINAL} found. "
            f"Run scripts/enrich_test_set.py to produce the enriched file."
        )
    print(f"Test file: {test_file}")
    if test_file == TEST_FILE_ORIGINAL:
        print("[WARN] Using original test_full.jsonl — slide_text is empty for "
              "every record. NLI factual metric will be uninformative. "
              "Run scripts/enrich_test_set.py first to produce the enriched file.")
    records = [
        json.loads(line)
        for line in test_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if MAX_EXAMPLES:
        records = records[:MAX_EXAMPLES]
    print(f"Loaded {len(records)} test examples")
    return records, str(test_file)


# ── GENERATE PREDICTIONS ─────────────────────────────────────
def generate_predictions(records: list, model_name: str) -> tuple:
    preds, refs = [], []
    errors = 0
    print(f"\nGenerating predictions with: {model_name}")
    print(f"(This takes ~{len(records) * 2 // 60} minutes)")
    for i, r in enumerate(records):
        try:
            pred = call_llm(r["question"], r["slide_text"], model_name)
        except Exception as e:
            print(f"  Error on example {i}: {e}")
            pred = ""
            errors += 1
        preds.append(pred)
        refs.append(r["answer"])
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(records)} examples done")
        time.sleep(0.05)
    if errors:
        print(f"  ⚠  {errors} errors encountered")
    return preds, refs


# ── METRICS ───────────────────────────────────────────────────
def compute_bertscore(preds, refs):
    from bert_score import score as bs
    print("Computing BERTScore (this takes 1-2 minutes)...")
    P, R, F1 = bs(
        preds, refs, lang="en",
        model_type="microsoft/deberta-xlarge-mnli", verbose=False,
    )
    return {
        "precision": float(P.mean()),
        "recall":    float(R.mean()),
        "f1":        float(F1.mean()),
        "f1_std":    float(F1.std()),
        "f1_values": F1.tolist(),
    }


def compute_rouge(preds, refs):
    from rouge_score import rouge_scorer as rs
    scorer = rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = defaultdict(list)
    for p, r in zip(preds, refs):
        result = scorer.score(r, p)
        for k, v in result.items():
            scores[k].append(v.fmeasure)
    return {k: {"mean": statistics.mean(v), "std": statistics.stdev(v) if len(v) > 1 else 0}
            for k, v in scores.items()}


def compute_factual_consistency(preds, slide_texts):
    """NLI entailment vs slide context (now non-empty when enriched)."""
    print("Computing NLI factual consistency...")
    try:
        from transformers import pipeline
        nli = pipeline("text-classification",
                       model="facebook/bart-large-mnli",
                       device=0)
        results = []
        for pred, ctx in zip(preds, slide_texts):
            if not pred.strip() or not ctx.strip():
                results.append("contradiction" if not pred.strip() else "neutral")
                continue
            premise = ctx[:1024]
            r = nli(f"{premise} </s></s> {pred}", truncation=True)[0]
            results.append(r["label"].lower())
        entailment_rate = sum(1 for r in results if "entail" in r) / len(results)
        contradiction_rate = sum(1 for r in results if "contra" in r) / len(results)
        return {
            "entailment_rate":     round(entailment_rate, 4),
            "contradiction_rate":  round(contradiction_rate, 4),
            "n_with_context":      sum(1 for c in slide_texts if c.strip()),
            "labels":              results[:30],
        }
    except Exception as e:
        print(f"  NLI failed ({e}) — skipping")
        return {"entailment_rate": None, "error": str(e)}


# Slide-grounded regex factual-error check (paper Eq. 10).
# A predicted numerical quantity is flagged if it does not appear
# (as substring) in the source slide transcript.
NUM_UNIT_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(HU|ms|MHz|kHz|GHz|MeV|keV|mm|cm|T1|T2|Tesla|T)\b",
    re.IGNORECASE,
)


def slide_grounded_factual_error(preds, slide_texts):
    """Returns dict with rate + per-example flags."""
    n_err = 0
    n_total = 0
    flags = []
    for pred, ctx in zip(preds, slide_texts):
        if not pred.strip():
            flags.append(False)
            continue
        n_total += 1
        flagged = False
        for m in NUM_UNIT_RE.finditer(pred):
            token = f"{m.group(1)} {m.group(2)}"
            if (m.group(1) not in (ctx or "")
                and token not in (ctx or "")
                and m.group(0) not in (ctx or "")):
                flagged = True
                break
        flags.append(flagged)
        if flagged:
            n_err += 1
    return {
        "factual_error_rate": round(n_err / max(1, n_total), 4),
        "n_errors":           n_err,
        "n_evaluated":        n_total,
        "flag_per_example":   flags,
    }


def per_difficulty_scores(records, f1_values):
    by_diff = defaultdict(list)
    for r, f1 in zip(records, f1_values):
        by_diff[r.get("difficulty", "unknown")].append(f1)
    return {
        diff: {"n": len(v), "mean_f1": round(statistics.mean(v), 4)}
        for diff, v in by_diff.items()
    }


def select_qualitative_examples(records, preds, f1_values, n=5):
    pairs = list(zip(records, preds, f1_values))
    pairs.sort(key=lambda x: x[2])
    step = max(1, len(pairs) // n)
    selected = [pairs[i] for i in range(0, len(pairs), step)][:n]
    return [
        {
            "lecture":      r["lecture_name"],
            "slide":        r["slide_name"],
            "difficulty":   r.get("difficulty", "?"),
            "question":     r["question"],
            "reference":    r["answer"],
            "prediction":   pred,
            "bertscore_f1": round(f1, 4),
        }
        for r, pred, f1 in selected
    ]


# ── MAIN EVAL ─────────────────────────────────────────────────
def run_eval(model_label: str, model_name: str, out_path: Path):
    print(f"\n{'='*65}")
    print(f"  EVALUATING: {model_label}   model={model_name}")
    print(f"{'='*65}")

    records, test_file_used = load_test_set()
    preds, refs = generate_predictions(records, model_name)
    slide_texts = [r.get("slide_text", "") for r in records]

    bert  = compute_bertscore(preds, refs)
    rouge = compute_rouge(preds, refs)
    nli   = compute_factual_consistency(preds, slide_texts)
    fact  = slide_grounded_factual_error(preds, slide_texts)
    diff  = per_difficulty_scores(records, bert["f1_values"])
    qual  = select_qualitative_examples(records, preds, bert["f1_values"])

    n_with_ctx = sum(1 for ctx in slide_texts if ctx.strip())

    result = {
        "model_label":          model_label,
        "model_name":           model_name,
        "n_examples":           len(records),
        "test_file":            test_file_used,
        "n_with_context":       n_with_ctx,
        "bertscore":            bert,
        "rouge":                rouge,
        "factual_consistency":  nli,
        "factual_error_regex":  fact,
        "per_difficulty":       diff,
        "qualitative_examples": qual,
        # raw artefacts (NEW) — needed for downstream RAG/judge eval
        "predictions":          preds,
        "references":           refs,
        "slide_texts_used":     slide_texts,
        "lecture_ids":          [r.get("lecture_name", "") for r in records],
        "slide_nums":           [r.get("slide_num") for r in records],
        "questions":            [r["question"] for r in records],
        "difficulties":         [r.get("difficulty", "") for r in records],
    }

    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nResults saved → {out_path}")

    print(f"\n{'='*65}")
    print(f"  RESULTS — {model_label}")
    print(f"{'='*65}")
    print(f"  BERTScore F1:        {bert['f1']:.4f} ± {bert['f1_std']:.4f}")
    print(f"  ROUGE-1/2/L:         {rouge['rouge1']['mean']:.4f} / "
          f"{rouge['rouge2']['mean']:.4f} / {rouge['rougeL']['mean']:.4f}")
    if nli.get("entailment_rate") is not None:
        print(f"  NLI entailment rate: {nli['entailment_rate']*100:.1f}% "
              f"(n_with_ctx={nli.get('n_with_context')})")
    print(f"  Slide-grounded factual error: "
          f"{fact['factual_error_rate']*100:.2f}% "
          f"({fact['n_errors']}/{fact['n_evaluated']})")
    print(f"\n  Per-difficulty BERTScore F1:")
    for d, v in sorted(diff.items()):
        print(f"    {d:<15}: {v['mean_f1']:.4f}  (n={v['n']})")


# ── COMPARISON TABLE ──────────────────────────────────────────
def print_comparison():
    base_path = EVAL_DIR / "results_base.json"
    ft_path   = EVAL_DIR / "results_finetuned.json"
    if not base_path.exists() or not ft_path.exists():
        print("ERROR: Run both evaluations first.")
        return
    base = json.loads(base_path.read_text())
    ft   = json.loads(ft_path.read_text())

    def delta(a, b):
        return f"+{b-a:.4f}" if b >= a else f"{b-a:.4f}"

    print(f"\n  {'Metric':<28} {'Base':>10} {'MIMIC-LM':>12} {'Delta':>10}")
    print(f"  {'-'*60}")
    for name, b_val, ft_val in [
        ("BERTScore F1",  base["bertscore"]["f1"],          ft["bertscore"]["f1"]),
        ("BERTScore Prec",base["bertscore"]["precision"],   ft["bertscore"]["precision"]),
        ("BERTScore Rec", base["bertscore"]["recall"],      ft["bertscore"]["recall"]),
        ("ROUGE-1",       base["rouge"]["rouge1"]["mean"],  ft["rouge"]["rouge1"]["mean"]),
        ("ROUGE-L",       base["rouge"]["rougeL"]["mean"],  ft["rouge"]["rougeL"]["mean"]),
    ]:
        sig = " *" if abs(ft_val - b_val) > 0.02 else ""
        print(f"  {name:<28} {b_val:>10.4f} {ft_val:>12.4f} "
              f"{delta(b_val, ft_val):>10}{sig}")

    bf = base.get("factual_error_regex", {}).get("factual_error_rate")
    ff = ft.get("factual_error_regex", {}).get("factual_error_rate")
    if bf is not None and ff is not None:
        print(f"  {'Slide-grounded factual err':<28} {bf*100:>9.2f}% "
              f"{ff*100:>11.2f}%  {(ff-bf)*100:+.2f}pp")

    bn = base.get("factual_consistency", {}).get("entailment_rate")
    fn = ft.get("factual_consistency", {}).get("entailment_rate")
    if bn is not None and fn is not None:
        print(f"  {'NLI entailment':<28} {bn*100:>9.2f}% "
              f"{fn*100:>11.2f}%  {(fn-bn)*100:+.2f}pp")


# ── CLI ───────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--model", choices=["base", "finetuned"])
    g.add_argument("--compare", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    if args.compare:
        print_comparison()
    else:
        model_name = BASE_MODEL if args.model == "base" else FT_MODEL
        out_path   = args.out or (EVAL_DIR / f"results_{args.model}.json")
        run_eval(args.model, model_name, out_path)
