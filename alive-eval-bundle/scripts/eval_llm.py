"""
eval_llm.py
===========
Evaluates LLM answer quality on the held-out test set.

Computes:
  - BERTScore F1 (semantic similarity to reference answer)
  - ROUGE-L      (lexical overlap)
  - Factual consistency (NLI entailment from retrieved context)
  - Per-difficulty breakdown
  - Qualitative side-by-side examples for the paper

Run TWICE — once for each model:
    cd ~/islamm11/alive/backend
    conda activate sadtalker

    # Step 1: evaluate BASE model (make sure vLLM is serving base model)
    python evaluation/eval_llm.py --model base --out evaluation/results_base.json

    # Step 2: switch vLLM to fine-tuned model (see instructions in output)
    # Then:
    python evaluation/eval_llm.py --model finetuned --out evaluation/results_finetuned.json

    # Step 3: print comparison table
    python evaluation/eval_llm.py --compare

Install deps:
    pip install bert-score rouge-score transformers
"""

import json
import time
import argparse
import requests
import statistics
from pathlib import Path
from collections import defaultdict

# ── CONFIG ────────────────────────────────────────────────────
VLLM_URL      = "http://127.0.0.1:8000/v1/completions"
BASE_MODEL    = "meta-llama/Meta-Llama-3.1-8B-Instruct"
FT_MODEL      = "checkpoints/alive-llama-lora/merged"   # after merge-lora
TEST_FILE     = Path("data/finetune/test_full.jsonl")
EVAL_DIR      = Path("evaluation")
EVAL_DIR.mkdir(exist_ok=True)

# How many test examples to use  (None = all)
# For quick iteration during development, set to 50
MAX_EXAMPLES  = None

# ── PROMPT ────────────────────────────────────────────────────
def build_prompt(question: str, slide_text: str, model_name: str) -> str:
    """
    Same prompt format as app.py's build_prompt() so eval matches real usage.
    Uses Llama-3 chat format for instruction-following models.
    """
    system = (
        "You are a teaching assistant answering a student. "
        "Use ONLY the lecture context provided. "
        "Answer in 1-3 clear, precise sentences."
    )
    user = (
        f"LECTURE CONTEXT:\n{slide_text}\n\n"
        f"QUESTION:\n{question}\n\nANSWER:"
    )
    # Llama-3.1 chat format
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
            "temperature": 0.0,        # greedy for reproducibility
            "max_tokens":  200,
            "stop":        ["<|eot_id|>", "<|end_of_text|>", "\n\n"],
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["text"].strip()


# ── LOAD TEST SET ─────────────────────────────────────────────
def load_test_set():
    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Test file not found: {TEST_FILE}\n"
            f"Run clean_and_split.py first."
        )
    records = [
        json.loads(line)
        for line in TEST_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if MAX_EXAMPLES:
        records = records[:MAX_EXAMPLES]
    print(f"Loaded {len(records)} test examples from {TEST_FILE}")
    return records


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
        time.sleep(0.05)   # avoid hammering vLLM

    if errors:
        print(f"  ⚠  {errors} errors encountered (empty predictions for those examples)")
    return preds, refs


# ── METRICS ───────────────────────────────────────────────────
def compute_bertscore(preds, refs):
    from bert_score import score as bs
    print("Computing BERTScore (this takes 1-2 minutes)...")
    # Use DeBERTa-xlarge-mnli for highest correlation with human judgment
    # If this model is too slow, use model_type='distilbert-base-uncased'
    P, R, F1 = bs(
        preds, refs,
        lang="en",
        model_type="microsoft/deberta-xlarge-mnli",
        verbose=False,
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
    return {k: {"mean": statistics.mean(v), "std": statistics.stdev(v) if len(v)>1 else 0}
            for k, v in scores.items()}


def compute_factual_consistency(preds, slide_texts):
    """
    Uses NLI to check if prediction is entailed by slide context.
    Score = fraction of predictions labeled as 'entailment'.
    """
    print("Computing factual consistency (NLI)...")
    try:
        from transformers import pipeline
        nli = pipeline(
            "text-classification",
            model="facebook/bart-large-mnli",
            device=0,    # use GPU
        )
        results = []
        for pred, ctx in zip(preds, slide_texts):
            if not pred.strip():
                results.append("contradiction")
                continue
            # NLI: premise=context, hypothesis=prediction
            premise = ctx[:1024]   # truncate long contexts
            r = nli(f"{premise} </s></s> {pred}", truncation=True)[0]
            results.append(r["label"].lower())

        entailment_rate = sum(1 for r in results if "entail" in r) / len(results)
        return {
            "entailment_rate": round(entailment_rate, 4),
            "labels": results[:20],  # first 20 for inspection
        }
    except Exception as e:
        print(f"  NLI failed ({e}) — skipping factual consistency metric")
        return {"entailment_rate": None, "error": str(e)}


# ── PER-DIFFICULTY BREAKDOWN ──────────────────────────────────
def per_difficulty_scores(records, f1_values):
    by_diff = defaultdict(list)
    for r, f1 in zip(records, f1_values):
        by_diff[r.get("difficulty", "unknown")].append(f1)
    return {
        diff: {"n": len(v), "mean_f1": round(statistics.mean(v), 4)}
        for diff, v in by_diff.items()
    }


# ── QUALITATIVE EXAMPLES ─────────────────────────────────────
def select_qualitative_examples(records, preds, f1_values, n=5):
    """
    Pick n examples to show in the paper as Table X (qualitative comparison).
    Selects examples covering different difficulty levels and score ranges.
    """
    pairs = list(zip(records, preds, f1_values))
    # Sort by F1 to get high/medium/low examples
    pairs.sort(key=lambda x: x[2])
    step = max(1, len(pairs) // n)
    selected = [pairs[i] for i in range(0, len(pairs), step)][:n]
    return [
        {
            "lecture":    r["lecture_name"],
            "slide":      r["slide_name"],
            "difficulty": r.get("difficulty", "?"),
            "question":   r["question"],
            "reference":  r["answer"],
            "prediction": pred,
            "bertscore_f1": round(f1, 4),
        }
        for r, pred, f1 in selected
    ]


# ── MAIN EVAL ─────────────────────────────────────────────────
def run_eval(model_label: str, model_name: str, out_path: Path):
    print(f"\n{'='*65}")
    print(f"  EVALUATING: {model_label}")
    print(f"  Model:      {model_name}")
    print(f"{'='*65}")

    records = load_test_set()
    preds, refs = generate_predictions(records, model_name)

    slide_texts = [r["slide_text"] for r in records]

    # Metrics
    bert  = compute_bertscore(preds, refs)
    rouge = compute_rouge(preds, refs)
    nli   = compute_factual_consistency(preds, slide_texts)
    diff  = per_difficulty_scores(records, bert["f1_values"])
    qual  = select_qualitative_examples(records, preds, bert["f1_values"])

    result = {
        "model_label":  model_label,
        "model_name":   model_name,
        "n_examples":   len(records),
        "bertscore":    bert,
        "rouge":        rouge,
        "factual_consistency": nli,
        "per_difficulty": diff,
        "qualitative_examples": qual,
    }

    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nResults saved → {out_path}")

    # Print table
    print(f"\n{'='*65}")
    print(f"  RESULTS — {model_label}")
    print(f"{'='*65}")
    print(f"  BERTScore F1:      {bert['f1']:.4f} ± {bert['f1_std']:.4f}")
    print(f"  ROUGE-1:           {rouge['rouge1']['mean']:.4f}")
    print(f"  ROUGE-2:           {rouge['rouge2']['mean']:.4f}")
    print(f"  ROUGE-L:           {rouge['rougeL']['mean']:.4f}")
    if nli["entailment_rate"] is not None:
        print(f"  Factual consist.:  {nli['entailment_rate']*100:.1f}%")
    print(f"\n  Per-difficulty BERTScore F1:")
    for d, v in sorted(diff.items()):
        print(f"    {d:<15}: {v['mean_f1']:.4f}  (n={v['n']})")


# ── COMPARISON TABLE ──────────────────────────────────────────
def print_comparison():
    base_path = EVAL_DIR / "results_base.json"
    ft_path   = EVAL_DIR / "results_finetuned.json"

    if not base_path.exists() or not ft_path.exists():
        print("ERROR: Run both evaluations first:")
        print("  python evaluation/eval_llm.py --model base")
        print("  python evaluation/eval_llm.py --model finetuned")
        return

    base = json.loads(base_path.read_text())
    ft   = json.loads(ft_path.read_text())

    def delta(a, b):
        return f"+{b-a:.4f}" if b >= a else f"{b-a:.4f}"

    print(f"\n{'='*65}")
    print(f"  COMPARISON TABLE  (copy to paper Table 2)")
    print(f"{'='*65}")
    print(f"  {'Metric':<25} {'Base LLM':>12} {'ALIVE-ED':>12} {'Delta':>10}")
    print(f"  {'-'*59}")

    metrics = [
        ("BERTScore F1",  base["bertscore"]["f1"],             ft["bertscore"]["f1"]),
        ("BERTScore Prec",base["bertscore"]["precision"],       ft["bertscore"]["precision"]),
        ("BERTScore Rec", base["bertscore"]["recall"],          ft["bertscore"]["recall"]),
        ("ROUGE-1",       base["rouge"]["rouge1"]["mean"],      ft["rouge"]["rouge1"]["mean"]),
        ("ROUGE-2",       base["rouge"]["rouge2"]["mean"],      ft["rouge"]["rouge2"]["mean"]),
        ("ROUGE-L",       base["rouge"]["rougeL"]["mean"],      ft["rouge"]["rougeL"]["mean"]),
    ]

    for name, b_val, ft_val in metrics:
        d = delta(b_val, ft_val)
        sig = " *" if abs(ft_val - b_val) > 0.02 else ""
        print(f"  {name:<25} {b_val:>12.4f} {ft_val:>12.4f} {d:>10}{sig}")

    base_fc = base["factual_consistency"].get("entailment_rate")
    ft_fc   = ft["factual_consistency"].get("entailment_rate")
    if base_fc and ft_fc:
        print(f"  {'Factual Consistency':<25} {base_fc*100:>11.1f}% {ft_fc*100:>11.1f}% {delta(base_fc, ft_fc):>10}")

    print(f"\n  * indicates improvement > 0.02 (likely significant)")
    print(f"\n  Per-difficulty comparison (BERTScore F1):")
    all_diffs = set(base["per_difficulty"]) | set(ft["per_difficulty"])
    for diff in sorted(all_diffs):
        b = base["per_difficulty"].get(diff, {}).get("mean_f1", 0)
        f = ft["per_difficulty"].get(diff, {}).get("mean_f1", 0)
        n = ft["per_difficulty"].get(diff, {}).get("n", 0)
        print(f"    {diff:<15}: base={b:.4f}  ft={f:.4f}  delta={delta(b,f)}  (n={n})")

    # Print qualitative examples from fine-tuned model
    print(f"\n{'='*65}")
    print(f"  QUALITATIVE EXAMPLES (for paper Table X)")
    print(f"{'='*65}")
    for i, ex in enumerate(ft["qualitative_examples"][:3]):
        print(f"\n  Example {i+1} [{ex['difficulty']}] — {ex['lecture']}/{ex['slide']}")
        print(f"  Q:   {ex['question']}")
        print(f"  Ref: {ex['reference'][:150]}")
        print(f"  FT:  {ex['prediction'][:150]}")
        print(f"  BERTScore F1: {ex['bertscore_f1']}")


# ── CLI ───────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLM for ALIVE-ED Paper 1")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--model", choices=["base", "finetuned"],
        help="Which model to evaluate"
    )
    group.add_argument(
        "--compare", action="store_true",
        help="Print comparison table (requires both evaluations done)"
    )
    parser.add_argument(
        "--out", type=Path,
        help="Output JSON path (default: evaluation/results_{model}.json)"
    )
    args = parser.parse_args()

    if args.compare:
        print_comparison()
    else:
        model_name = BASE_MODEL if args.model == "base" else FT_MODEL
        out_path   = args.out or (EVAL_DIR / f"results_{args.model}.json")
        run_eval(args.model, model_name, out_path)

        if args.model == "base":
            print(f"\n{'='*65}")
            print(f"  NEXT STEP:")
            print(f"  1. Stop current vLLM (Ctrl+C in Terminal A)")
            print(f"  2. Merge LoRA:  axolotl merge-lora configs/alive_lora.yml")
            print(f"  3. Serve FT model:")
            print(f"     CUDA_VISIBLE_DEVICES=0 vllm serve {FT_MODEL} \\")
            print(f"       --host 127.0.0.1 --port 8000 --max-model-len 4096")
            print(f"  4. Update llm_client.py: VLLM_MODEL = '{FT_MODEL}'")
            print(f"  5. Run: python evaluation/eval_llm.py --model finetuned")
            print(f"{'='*65}")
