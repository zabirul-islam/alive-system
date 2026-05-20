"""
generate_qa.py
==============
Strategy:
  - Per lecture: merge ALL slide texts in order → one full lecture document
  - Generate 5 × N QA pairs for the whole lecture at once (N = slide count)
  - Save one JSON per lecture
  - Resume: skip already-processed lectures
"""

import re
import json
import time
import logging
import requests
from pathlib import Path
from math import ceil

# ── CONFIG ────────────────────────────────────────────────────
LECTURES_DIR = Path("Lectures")
OUT_DIR      = Path("data/qa_raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)

VLLM_URL     = "http://127.0.0.1:8000/v1/completions"
VLLM_MODEL   = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MIN_WORDS    = 15        # minimum words for a slide to count
QA_PER_SLIDE = 5         # multiplier: total pairs = QA_PER_SLIDE × slide_count
MAX_TOKENS   = 4000      # max tokens for LLM response
SLEEP_SEC    = 1.0       # pause between lecture API calls
BATCH_SIZE        = 20         # QA pairs per API call
MAX_LECTURE_CHARS = 5000       # max chars of lecture text per API call (~1200 tokens)

# ── LOGGING ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler("data_pipeline/qa_gen.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── PROMPT ────────────────────────────────────────────────────
SYSTEM = (
    "You are a medical imaging professor creating comprehensive study materials. "
    "You generate precise, educational question-answer pairs from full lecture content. "
    "You only output valid JSON arrays — nothing else."
)

USER_TEMPLATE = """\
Below is the complete text of a medical imaging lecture, merged from all slides in order.
Read and understand the full lecture, then generate exactly {n_pairs} question-answer pairs
that a student studying this lecture might ask.

REQUIREMENTS:
- Cover the breadth of the lecture — do not focus only on early slides
- Vary difficulty evenly: roughly 1/3 basic (definition/recall), 1/3 intermediate (apply/explain), 1/3 advanced (analyze/compare/derive)
- Each answer: 2-4 sentences, precise, uses correct medical/technical terminology
- Questions should require understanding the lecture content, not just keyword matching
- Do NOT generate questions about slide numbers, lecture titles, or meta-information
- Output ONLY a valid JSON array — no markdown fences, no explanation, no preamble

[FULL LECTURE TEXT]
{lecture_text}

JSON output (exactly {n_pairs} items):
[
  {{"question": "...", "answer": "...", "difficulty": "basic"}},
  ...
]"""


# ── HELPERS ───────────────────────────────────────────────────
def slide_number(filename: str) -> int:
    m = re.search(r"(\d+)", filename)
    return int(m.group(1)) if m else -1


def lecture_number(dirname: str) -> int:
    m = re.search(r"(\d+)", dirname)
    return int(m.group(1)) if m else 0


def load_lecture_slides(ldir: Path) -> list[dict]:
    txt_dir = ldir / "Texts"
    if not txt_dir.exists():
        return []
    slides = []
    for txt_file in sorted(txt_dir.glob("*.txt"), key=lambda f: slide_number(f.name)):
        try:
            text = txt_file.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception:
            text = ""
        if len(text.split()) >= MIN_WORDS:
            slides.append({
                "slide_name": txt_file.stem,
                "slide_num":  slide_number(txt_file.name),
                "text":       text,
            })
    return slides


def build_lecture_text(slides: list[dict]) -> str:
    """Merge all slide texts into one ordered lecture document."""
    parts = []
    for s in slides:
        parts.append(f"=== {s['slide_name']} ===\n{s['text']}")
    return "\n\n".join(parts)


def chunk_lecture_text(lecture_text: str, max_chars: int = 5000) -> list[str]:
    """Split lecture text into chunks that fit within context window."""
    if len(lecture_text) <= max_chars:
        return [lecture_text]
    chunks = []
    start  = 0
    while start < len(lecture_text):
        end = start + max_chars
        # Try to break at a slide boundary
        boundary = lecture_text.rfind("=== Slide", start, end)
        if boundary > start + 1000:
            end = boundary
        chunks.append(lecture_text[start:end])
        start = end
    return chunks


def call_vllm(lecture_text: str, n_pairs: int) -> str:
    user_msg = USER_TEMPLATE.format(
        lecture_text=lecture_text,
        n_pairs=n_pairs,
    )
    prompt = (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n{SYSTEM}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n{user_msg}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
    )
    r = requests.post(
        VLLM_URL,
        json={
            "model":       VLLM_MODEL,
            "prompt":      prompt,
            "temperature": 0.7,
            "max_tokens":  MAX_TOKENS,
            "stop":        ["<|eot_id|>", "<|end_of_text|>"],
        },
        timeout=180,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["text"].strip()


def parse_json_safe(raw: str):
    raw = raw.strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and parsed:
            return parsed
    except Exception:
        pass
    try:
        cleaned = re.sub(r"```json|```", "", raw).strip()
        parsed = json.loads(cleaned)
        if isinstance(parsed, list) and parsed:
            return parsed
    except Exception:
        pass
    try:
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if m:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, list) and parsed:
                return parsed
    except Exception:
        pass
    try:
        objects = re.findall(r"\{[^{}]+\}", raw, re.DOTALL)
        joined  = "[" + ",".join(objects) + "]"
        parsed  = json.loads(joined)
        if isinstance(parsed, list) and parsed:
            return parsed
    except Exception:
        pass
    return None


def validate_pairs(pairs: list) -> list:
    valid = []
    seen  = set()
    for p in pairs:
        if not isinstance(p, dict):
            continue
        q = str(p.get("question", "")).strip()
        a = str(p.get("answer",   "")).strip()
        if "?" not in q:
            continue
        if len(q.split()) < 5 or len(a.split()) < 8:
            continue
        q_lower = q.lower()
        if q_lower in seen:
            continue
        seen.add(q_lower)
        valid.append({
            "question":   q,
            "answer":     a,
            "difficulty": p.get("difficulty", "intermediate"),
        })
    return valid


def generate_in_batches(lecture_text: str, total_pairs: int) -> list:
    """
    Chunk lecture text to fit context window.
    For each chunk, request BATCH_SIZE pairs.
    Total calls = ceil(total_pairs / BATCH_SIZE), distributed across chunks.
    """
    all_pairs  = []
    chunks     = chunk_lecture_text(lecture_text, max_chars=MAX_LECTURE_CHARS)
    pairs_done = 0
    pairs_per_chunk = max(BATCH_SIZE, total_pairs // len(chunks))

    log.info(f"  Lecture split into {len(chunks)} chunk(s) of ~{MAX_LECTURE_CHARS} chars each")

    for ci, chunk in enumerate(chunks):
        if pairs_done >= total_pairs:
            break
        n = min(pairs_per_chunk, total_pairs - pairs_done)
        log.info(f"  Chunk {ci+1}/{len(chunks)}: requesting {n} pairs ({pairs_done} collected so far)")
        try:
            raw   = call_vllm(chunk, n)
            pairs = parse_json_safe(raw)
            if pairs:
                valid = validate_pairs(pairs)
                all_pairs.extend(valid)
                pairs_done += len(valid)
                log.info(f"  Chunk {ci+1}: got {len(valid)} valid pairs")
            else:
                log.warning(f"  Chunk {ci+1}: parse failed | raw: {raw[:120]}")
        except requests.RequestException as e:
            log.error(f"  Chunk {ci+1}: API error: {e}")
            time.sleep(5)
        time.sleep(SLEEP_SEC)

    return all_pairs


# ── MAIN ──────────────────────────────────────────────────────
def main():
    lecture_dirs = sorted(
        [d for d in LECTURES_DIR.iterdir() if d.is_dir()],
        key=lambda d: lecture_number(d.name),
    )
    log.info(f"Found {len(lecture_dirs)} lectures")

    stats = {"ok": 0, "skipped": 0, "failed": 0}

    for ldir in lecture_dirs:
        out_file = OUT_DIR / f"{ldir.name.replace(' ', '_')}.json"

        # Resume: skip already processed lectures
        if out_file.exists():
            log.info(f"SKIP (exists): {ldir.name}")
            stats["skipped"] += 1
            continue

        slides = load_lecture_slides(ldir)
        if not slides:
            log.warning(f"No usable slides in {ldir.name} — skipping")
            stats["failed"] += 1
            continue

        n_slides    = len(slides)
        total_pairs = n_slides * QA_PER_SLIDE
        log.info(f"Processing: {ldir.name} | {n_slides} slides → {total_pairs} QA pairs")

        lecture_text = build_lecture_text(slides)
        log.info(f"  Lecture text: {len(lecture_text)} chars, {len(lecture_text.split())} words")

        all_pairs = generate_in_batches(lecture_text, total_pairs)

        if not all_pairs:
            stats["failed"] += 1
            log.error(f"FAILED (no valid pairs): {ldir.name}")
            continue

        out_file.write_text(json.dumps({
            "lecture_name":  ldir.name,
            "lecture_num":   lecture_number(ldir.name),
            "slide_count":   n_slides,
            "pairs_target":  total_pairs,
            "pairs_actual":  len(all_pairs),
            "pairs":         all_pairs,
        }, indent=2, ensure_ascii=False))

        stats["ok"] += 1
        log.info(
            f"SAVED: {ldir.name} | "
            f"{len(all_pairs)}/{total_pairs} pairs → {out_file.name}"
        )

    log.info(f"\nFINISHED. {stats}")
    log.info("Run clean_and_split.py next.")


if __name__ == "__main__":
    main()
