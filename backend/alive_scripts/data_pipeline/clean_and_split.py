"""
clean_and_split.py
==================
Loads all raw QA JSON files from data/qa_raw/,
cleans them, deduplicates, and produces:

    data/finetune/train.jsonl   (~80%)
    data/finetune/val.jsonl     (~10%)
    data/finetune/test.jsonl    (~10%)
    data/finetune/test_full.jsonl   (test set WITH metadata — used by eval_llm.py)
    data/finetune/stats.json        (dataset statistics for the paper)

Alpaca format used for LoRA training:
    {"instruction": "<question>", "input": "<lecture context>", "output": "<answer>"}

Run:
    cd ~/islamm11/alive/backend
    python data_pipeline/clean_and_split.py
"""

import re
import json
import random
import hashlib
import statistics
from pathlib import Path
from collections import defaultdict

# ── CONFIG ────────────────────────────────────────────────────
RAW_DIR  = Path("data/qa_raw")
OUT_DIR  = Path("data/finetune")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED            = 42
TRAIN_RATIO     = 0.80
VAL_RATIO       = 0.10
TEST_RATIO      = 0.10
MIN_ANS_WORDS   = 10
MIN_Q_WORDS     = 5
DEDUP_THRESHOLD = 10   # first N words used as fingerprint for dedup


# ── LOAD ──────────────────────────────────────────────────────
def load_all():
    records = []
    fail_count = 0
    for f in sorted(RAW_DIR.glob("*.json")):
        if "_FAIL" in f.name:
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            fail_count += 1
            continue

        for i, p in enumerate(data.get("pairs", [])):
            records.append({
                "lecture_name": data["lecture_name"],
                "lecture_num":  data["lecture_num"],
                "slide_name":   data.get("slide_name", f"pair_{i}"),
                "slide_num":    data.get("slide_num", i),
                "slide_text":   data.get("slide_text", ""),
                "question":     p["question"].strip(),
                "answer":       p["answer"].strip(),
                "difficulty":   p.get("difficulty", "intermediate"),
            })

    print(f"Loaded {len(records)} raw pairs from {RAW_DIR}")
    if fail_count:
        print(f"  (skipped {fail_count} malformed JSON files)")
    return records


# ── FILTER ────────────────────────────────────────────────────
def filter_records(records):
    before = len(records)
    filtered = []
    reasons = defaultdict(int)

    for r in records:
        q = r["question"]
        a = r["answer"]

        if len(q.split()) < MIN_Q_WORDS:
            reasons["q_too_short"] += 1
            continue
        if len(a.split()) < MIN_ANS_WORDS:
            reasons["a_too_short"] += 1
            continue
        if "?" not in q:
            reasons["no_question_mark"] += 1
            continue
        # Reject meta-questions about slide structure
        bad_phrases = ["slide number", "slide title", "this slide", "the title of",
                       "what does this slide", "what is shown on slide"]
        if any(bp in q.lower() for bp in bad_phrases):
            reasons["meta_question"] += 1
            continue
        # Reject answers that are just echoing the question
        q_words = set(q.lower().split())
        a_words = set(a.lower().split())
        if len(a_words) > 0 and len(q_words & a_words) / len(a_words) > 0.7:
            reasons["answer_echoes_question"] += 1
            continue

        filtered.append(r)

    print(f"After filtering: {len(filtered)} / {before} records")
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"  Removed {count:>5} — {reason}")
    return filtered


# ── DEDUPLICATE ───────────────────────────────────────────────
def deduplicate(records):
    """Use first N words of question as fingerprint."""
    seen   = {}
    unique = []
    dupes  = 0

    for r in records:
        words = r["question"].lower().split()[:DEDUP_THRESHOLD]
        key   = hashlib.md5(" ".join(words).encode()).hexdigest()

        if key in seen:
            dupes += 1
        else:
            seen[key] = True
            unique.append(r)

    print(f"After dedup: {len(unique)} records (removed {dupes} duplicates)")
    return unique


# ── CONVERT TO ALPACA FORMAT ──────────────────────────────────
def to_alpaca(r: dict) -> dict:
    """
    Alpaca format expected by axolotl for instruction-tuning.
    The 'input' field provides the grounding context so the model
    learns to answer ONLY from lecture content.
    """
    context = r["slide_text"] if r["slide_text"] else r["lecture_name"]
    return {
        "instruction": r["question"],
        "input": (
            f"LECTURE: {r['lecture_name']}\n"
            f"CONTEXT:\n{context}"
        ),
        "output": r["answer"],
    }


# ── SPLIT ─────────────────────────────────────────────────────
def stratified_split(records):
    """
    Stratify by lecture number so each lecture appears in all splits.
    This prevents train/test contamination where a lecture only
    appears in one split.
    """
    # Group by lecture
    by_lecture = defaultdict(list)
    for r in records:
        by_lecture[r["lecture_num"]].append(r)

    train, val, test = [], [], []
    random.seed(SEED)

    for lec_num in sorted(by_lecture.keys()):
        lec_records = by_lecture[lec_num]
        random.shuffle(lec_records)
        n = len(lec_records)

        n_val   = max(1, int(n * VAL_RATIO))
        n_test  = max(1, int(n * TEST_RATIO))
        n_train = n - n_val - n_test

        train.extend(lec_records[:n_train])
        val.extend(lec_records[n_train:n_train + n_val])
        test.extend(lec_records[n_train + n_val:])

    # Final shuffle within each split
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test


# ── SAVE ──────────────────────────────────────────────────────
def save_jsonl(records, path, alpaca=True):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            item = to_alpaca(r) if alpaca else r
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Saved {len(records):>5} records → {path}")


# ── STATS ─────────────────────────────────────────────────────
def compute_stats(train, val, test, all_records):
    q_lens   = [len(r["question"].split()) for r in all_records]
    a_lens   = [len(r["answer"].split())   for r in all_records]
    txt_lens = [len(r["slide_text"].split()) for r in all_records]

    # Per-difficulty counts
    diff_counts = defaultdict(int)
    for r in all_records:
        diff_counts[r["difficulty"]] += 1

    # Per-lecture counts
    lec_counts = defaultdict(int)
    for r in all_records:
        lec_counts[r["lecture_name"]] += 1

    stats = {
        "total_pairs":    len(all_records),
        "train":          len(train),
        "val":            len(val),
        "test":           len(test),
        "difficulty": dict(diff_counts),
        "lectures":   dict(lec_counts),
        "question_len": {
            "min": min(q_lens), "max": max(q_lens),
            "mean": round(statistics.mean(q_lens), 1),
            "median": statistics.median(q_lens),
        },
        "answer_len": {
            "min": min(a_lens), "max": max(a_lens),
            "mean": round(statistics.mean(a_lens), 1),
            "median": statistics.median(a_lens),
        },
        "slide_text_len": {
            "min": min(txt_lens), "max": max(txt_lens),
            "mean": round(statistics.mean(txt_lens), 1),
        },
    }
    return stats


# ── MAIN ──────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  ALIVE-ED  clean_and_split.py")
    print("=" * 60)

    # Load
    records = load_all()
    if not records:
        print("ERROR: No QA records found in data/qa_raw/")
        print("Run generate_qa.py first.")
        return

    # Clean
    records = filter_records(records)
    records = deduplicate(records)

    # Split (stratified by lecture)
    train, val, test = stratified_split(records)

    print(f"\nFinal split:")
    print(f"  Train: {len(train)}")
    print(f"  Val:   {len(val)}")
    print(f"  Test:  {len(test)}")
    print(f"  Total: {len(train)+len(val)+len(test)}")

    # Save training files (Alpaca format for axolotl)
    print(f"\nSaving to {OUT_DIR}/")
    save_jsonl(train, OUT_DIR / "train.jsonl", alpaca=True)
    save_jsonl(val,   OUT_DIR / "val.jsonl",   alpaca=True)
    save_jsonl(test,  OUT_DIR / "test.jsonl",  alpaca=True)

    # Save test set WITH metadata (used by eval_llm.py)
    save_jsonl(test, OUT_DIR / "test_full.jsonl", alpaca=False)

    # Save stats
    stats = compute_stats(train, val, test, records)
    (OUT_DIR / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"  Saved stats → {OUT_DIR}/stats.json")

    # Print summary for paper Table 3
    print(f"\n{'='*60}")
    print(f"  DATASET STATS  (copy to paper Table 3)")
    print(f"{'='*60}")
    print(f"  Total QA pairs:          {stats['total_pairs']:>6,}")
    print(f"  Train / Val / Test:      {stats['train']:>6,} / {stats['val']:>5,} / {stats['test']:>5,}")
    print(f"  Lectures covered:        {len(stats['lectures']):>6}")
    print(f"  Avg question length:     {stats['question_len']['mean']:>6.1f} words")
    print(f"  Avg answer length:       {stats['answer_len']['mean']:>6.1f} words")
    print(f"  Avg slide context len:   {stats['slide_text_len']['mean']:>6.1f} words")
    print(f"  Difficulty distribution:")
    for diff, count in stats["difficulty"].items():
        pct = 100 * count / stats["total_pairs"]
        print(f"    {diff:<15}: {count:>5,}  ({pct:.1f}%)")
    print(f"{'='*60}\n")
    print("Next step: run  axolotl train configs/alive_lora.yml")


if __name__ == "__main__":
    main()
