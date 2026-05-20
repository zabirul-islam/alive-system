"""
rag_index_v2.py
===============
Rebuilds the ALIVE RAG store from the full Lectures/ directory.

Key improvements over the original rag_index.py:
  1. Reads ALL lectures (not just lecture01.srt)
  2. Uses a stronger embedding model: all-mpnet-base-v2 (768-dim vs 384-dim)
  3. Preserves lecture + slide metadata (not just start/end timestamps)
  4. Computes realistic timestamps from slide order within each lecture
  5. Keeps backward compatibility — app.py loads the same faiss.index + chunks.json

Directory structure it expects:
    Lectures/
        Lecture 1/Texts/Slide1.txt  ...  Slide50.txt
        Lecture 2/Texts/...
        ...
        Lecture 23/Texts/...

Run:
    cd ~/islamm11/alive/backend
    conda activate sadtalker            # same env as app.py
    pip install sentence-transformers faiss-gpu   # if not already installed
    python rag_index_v2.py

After this completes, RESTART the FastAPI server (Terminal B).
The new index is larger and richer — expect retrieval quality to improve.
"""

import re
import json
import logging
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ── CONFIG ────────────────────────────────────────────────────
LECTURES_DIR = Path("Lectures")
STORE_DIR    = Path("rag_store")
STORE_DIR.mkdir(parents=True, exist_ok=True)

# Upgrade from all-MiniLM-L6-v2 (384-dim) to all-mpnet-base-v2 (768-dim)
# Benchmark on BEIR: mpnet-base = 57.6  vs  MiniLM-L6 = 52.5 (NDCG@10)
EMB_MODEL = "sentence-transformers/all-mpnet-base-v2"

# How many seconds we assume each slide takes in a lecture video.
# This drives the timestamp-aware retrieval in app.py.
# 30s/slide is conservative — adjust to your actual lecture pacing.
# You can also set per-lecture if pacing varies significantly.
SECONDS_PER_SLIDE = 30

# ── LOGGING ───────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


# ── HELPERS ───────────────────────────────────────────────────
def lecture_number(name: str) -> int:
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else 0


def slide_number(name: str) -> int:
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else 0


# ── SLIDE LOADING ─────────────────────────────────────────────
def load_all_slides() -> list:
    """
    Returns a list of chunk dicts, sorted by (lecture_num, slide_num).
    Each chunk has the fields that app.py's retrieve_context() expects:
        start, end, text
    Plus extra metadata for Paper 1 evaluation:
        lecture_name, lecture_num, slide_name, slide_num
    """
    chunks = []

    lecture_dirs = sorted(
        [d for d in LECTURES_DIR.iterdir() if d.is_dir()],
        key=lambda d: lecture_number(d.name)
    )

    if not lecture_dirs:
        raise RuntimeError(f"No directories found in {LECTURES_DIR.resolve()}")

    for ldir in lecture_dirs:
        txt_dir = ldir / "Texts"
        if not txt_dir.exists():
            log.warning(f"No Texts/ in {ldir.name} — skipping")
            continue

        txt_files = sorted(
            txt_dir.glob("*.txt"),
            key=lambda f: slide_number(f.name)
        )

        if not txt_files:
            log.warning(f"No .txt files in {txt_dir} — skipping")
            continue

        lec_num = lecture_number(ldir.name)

        for txt_file in txt_files:
            sld_num  = slide_number(txt_file.name)
            text     = txt_file.read_text(encoding="utf-8", errors="ignore").strip()

            if not text or len(text.split()) < 5:
                log.debug(f"Skipping near-empty slide: {ldir.name}/{txt_file.name}")
                continue

            # Synthetic timestamp:
            # Slide N within Lecture L starts at (N-1)*30s
            # (relative to the start of THAT lecture's video)
            start = float((sld_num - 1) * SECONDS_PER_SLIDE)
            end   = float(sld_num * SECONDS_PER_SLIDE)

            chunks.append({
                # --- Fields used by app.py (backward compatible) ---
                "start": start,
                "end":   end,
                "text":  text,
                # --- Extra metadata for evaluation ---
                "lecture_name": ldir.name,
                "lecture_num":  lec_num,
                "slide_name":   txt_file.stem,
                "slide_num":    sld_num,
                "slide_id":     f"{ldir.name.replace(' ','_')}__Slide{sld_num}",
            })

        log.info(f"  {ldir.name:<15}: {len(txt_files)} slides loaded")

    log.info(f"Total chunks: {len(chunks)}")
    return chunks


# ── INDEX BUILDING ────────────────────────────────────────────
def build_faiss_index(chunks: list, model: SentenceTransformer):
    texts = [c["text"] for c in chunks]
    log.info(f"Embedding {len(texts)} chunks with {EMB_MODEL} ...")

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,    # required for inner-product similarity
        show_progress_bar=True,
        batch_size=64,                # H100 can handle large batches
    )

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)    # inner product = cosine on normalized vectors
    index.add(embeddings.astype(np.float32))

    log.info(f"FAISS index built: {index.ntotal} vectors × {dim} dims")
    return index


# ── SAVE ─────────────────────────────────────────────────────
def save_store(index, chunks):
    faiss.write_index(index, str(STORE_DIR / "faiss.index"))
    log.info(f"Saved faiss.index → {STORE_DIR}/faiss.index")

    (STORE_DIR / "chunks.json").write_text(
        json.dumps(chunks, indent=2, ensure_ascii=False)
    )
    log.info(f"Saved chunks.json  → {STORE_DIR}/chunks.json  ({len(chunks)} entries)")

    (STORE_DIR / "meta.json").write_text(
        json.dumps({
            "embedding_model":    EMB_MODEL,
            "n_chunks":           len(chunks),
            "index_dim":          index.d,
            "seconds_per_slide":  SECONDS_PER_SLIDE,
            "lectures_indexed":   sorted(set(c["lecture_name"] for c in chunks)),
        }, indent=2)
    )
    log.info(f"Saved meta.json    → {STORE_DIR}/meta.json")


# ── VERIFY ────────────────────────────────────────────────────
def verify_retrieval(index, chunks, model):
    """Quick sanity check — retrieve top-3 for a sample query."""
    log.info("\n--- Retrieval sanity check ---")
    test_queries = [
        "What is filtered backprojection in CT imaging?",
        "How does MRI use magnetic fields to create images?",
        "What are common CT artifacts and how to reduce them?",
    ]
    for q in test_queries:
        q_emb = model.encode([q], convert_to_numpy=True, normalize_embeddings=True)
        D, I  = index.search(q_emb.astype(np.float32), 3)
        log.info(f"\nQuery: {q}")
        for rank, (idx, score) in enumerate(zip(I[0], D[0])):
            c = chunks[idx]
            log.info(f"  [{rank+1}] score={score:.4f}  {c['lecture_name']} / {c['slide_name']}")
            log.info(f"       {c['text'][:100]}...")


# ── MAIN ──────────────────────────────────────────────────────
def main():
    log.info("=" * 60)
    log.info("  ALIVE-ED  rag_index_v2.py")
    log.info("=" * 60)

    # Load slides
    chunks = load_all_slides()
    if not chunks:
        log.error("No slides loaded — check LECTURES_DIR path")
        return

    # Load embedding model on GPU
    log.info(f"Loading embedding model: {EMB_MODEL}")
    model = SentenceTransformer(EMB_MODEL, device="cuda")

    # Build FAISS index
    index = build_faiss_index(chunks, model)

    # Save
    save_store(index, chunks)

    # Verify
    verify_retrieval(index, chunks, model)

    log.info("\n" + "=" * 60)
    log.info("  Done. RESTART FastAPI server (Terminal B) to load new index.")
    log.info("  Command: conda activate sadtalker && python -m uvicorn app:app \\")
    log.info("           --host 0.0.0.0 --port 9000")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
