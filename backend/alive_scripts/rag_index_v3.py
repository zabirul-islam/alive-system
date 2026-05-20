"""
rag_index_v3.py
===============
Rebuilds the ALIVE RAG store from BOTH lecture slides AND medical imaging books.

Key improvements over rag_index_v2.py:
  1. Adds book PDF ingestion (PyMuPDF / pdfplumber fallback)
  2. Overlapping 300-word chunks for books (captures cross-paragraph reasoning)
  3. Each chunk tagged with source="slide" or source="book" for transparency
  4. Merges both into a single FAISS index — one retrieval call covers everything
  5. Slide chunks are weighted slightly higher at retrieval time via a source_boost
     field stored in metadata (used in app.py retrieve_context)
  6. Fully backward compatible — app.py loads the same faiss.index + chunks.json

Directory structure expected:
    alive_scripts/
        Lectures/
            Lecture 1/Texts/Slide1.txt ... Slide50.txt
            Lecture 2/Texts/...
            ...
        Books/
            prince_medical_imaging.pdf
            bushberg_essential_physics.pdf
            <any other .pdf or .txt book files>

Run:
    cd ~/islamm11/alive/backend/alive_scripts
    conda activate sadtalker
    pip install pymupdf --break-system-packages   # or: pip install pdfplumber
    python rag_index_v3.py

After completion, RESTART the FastAPI server (Terminal D).
"""

import re
import json
import logging
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ── Try PDF libraries in order of preference ──────────────────────────────────
try:
    import fitz  # PyMuPDF — best quality
    PDF_BACKEND = "pymupdf"
except ImportError:
    try:
        import pdfplumber
        PDF_BACKEND = "pdfplumber"
    except ImportError:
        PDF_BACKEND = None

# ── CONFIG ────────────────────────────────────────────────────────────────────
LECTURES_DIR = Path("Lectures")
BOOKS_DIR    = Path("Books")          # put your PDF/TXT books here
STORE_DIR    = Path("rag_store")
STORE_DIR.mkdir(parents=True, exist_ok=True)

EMB_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Slide timing: how many seconds per slide (for timestamp-aware retrieval)
SECONDS_PER_SLIDE = 30

# Book chunking: target chunk size and overlap (in words)
BOOK_CHUNK_WORDS   = 300
BOOK_OVERLAP_WORDS = 60

# Source boost: added to retrieval score for slide chunks (they are primary source)
# Book chunks are secondary grounding — useful but slides take priority
SLIDE_BOOST = 0.05   # small — retrieval is still primarily semantic

# ── LOGGING ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# SLIDE LOADING  (unchanged from v2, adds source field)
# ════════════════════════════════════════════════════════════════════════════
def lecture_number(name):
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else 0

def slide_number(name):
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else 0

def load_slides():
    chunks = []
    lecture_dirs = sorted(
        [d for d in LECTURES_DIR.iterdir() if d.is_dir()],
        key=lambda d: lecture_number(d.name)
    )
    if not lecture_dirs:
        log.warning(f"No lecture dirs found in {LECTURES_DIR.resolve()}")
        return chunks

    for ldir in lecture_dirs:
        txt_dir = ldir / "Texts"
        if not txt_dir.exists():
            log.warning(f"No Texts/ in {ldir.name} — skipping")
            continue
        txt_files = sorted(txt_dir.glob("*.txt"), key=lambda f: slide_number(f.name))
        lec_num   = lecture_number(ldir.name)

        for txt_file in txt_files:
            sld_num = slide_number(txt_file.name)
            text    = txt_file.read_text(encoding="utf-8", errors="ignore").strip()
            if not text or len(text.split()) < 5:
                continue
            chunks.append({
                "start"       : float((sld_num - 1) * SECONDS_PER_SLIDE),
                "end"         : float(sld_num * SECONDS_PER_SLIDE),
                "text"        : text,
                "source"      : "slide",
                "source_boost": SLIDE_BOOST,
                "lecture_name": ldir.name,
                "lecture_num" : lec_num,
                "slide_name"  : txt_file.stem,
                "slide_num"   : sld_num,
                "slide_id"    : f"{ldir.name.replace(' ','_')}__Slide{sld_num}",
            })

        log.info(f"  {ldir.name:<15}: {len(txt_files)} slides")

    log.info(f"Slides total: {len(chunks)} chunks")
    return chunks


# ════════════════════════════════════════════════════════════════════════════
# BOOK LOADING
# ════════════════════════════════════════════════════════════════════════════
def extract_text_pymupdf(pdf_path):
    """Extract raw text from PDF using PyMuPDF (best quality)."""
    doc   = fitz.open(str(pdf_path))
    pages = []
    for page in doc:
        pages.append(page.get_text("text"))
    doc.close()
    return "\n".join(pages)

def extract_text_pdfplumber(pdf_path):
    """Extract raw text using pdfplumber (fallback)."""
    import pdfplumber
    pages = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                pages.append(t)
    return "\n".join(pages)

def extract_text_from_file(file_path):
    """Dispatch to correct extractor based on file type."""
    if file_path.suffix.lower() == ".pdf":
        if PDF_BACKEND == "pymupdf":
            return extract_text_pymupdf(file_path)
        elif PDF_BACKEND == "pdfplumber":
            return extract_text_pdfplumber(file_path)
        else:
            log.error("No PDF library installed. Run: pip install pymupdf")
            return ""
    elif file_path.suffix.lower() == ".txt":
        return file_path.read_text(encoding="utf-8", errors="ignore")
    return ""

def clean_book_text(raw):
    """
    Basic cleaning for PDF-extracted text:
      - Remove page numbers (isolated digits on a line)
      - Collapse multiple blank lines
      - Remove header/footer repetition heuristic (lines < 5 words appearing > 5x)
    """
    lines = raw.split("\n")

    # Count short repeated lines (likely headers/footers)
    from collections import Counter
    short_line_counts = Counter(
        l.strip() for l in lines
        if 0 < len(l.strip().split()) < 6
    )
    repeated = {l for l, c in short_line_counts.items() if c > 5}

    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned.append("")
            continue
        if stripped in repeated:       # skip repeated header/footer lines
            continue
        if re.match(r"^\d{1,4}$", stripped):  # skip isolated page numbers
            continue
        cleaned.append(stripped)

    # Collapse runs of blank lines to single blank
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned))
    return text.strip()

def chunk_text(text, chunk_words=BOOK_CHUNK_WORDS, overlap_words=BOOK_OVERLAP_WORDS):
    """
    Split text into overlapping word-count chunks.
    Tries to break at sentence boundaries within the target window.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start  = 0
    while start < len(words):
        end   = min(start + chunk_words, len(words))
        chunk = " ".join(words[start:end])

        # Try to end at a sentence boundary (last . ? ! in the chunk)
        last_sent = max(
            chunk.rfind(". "),
            chunk.rfind("? "),
            chunk.rfind("! "),
        )
        if last_sent > chunk_words * 0.5 * 5:   # only snap if boundary is in latter half
            chunk = chunk[:last_sent + 1]

        chunks.append(chunk.strip())
        start += chunk_words - overlap_words

    return [c for c in chunks if len(c.split()) >= 20]   # drop tiny trailing chunks

def load_books():
    chunks = []

    if not BOOKS_DIR.exists():
        log.warning(f"Books/ directory not found at {BOOKS_DIR.resolve()} — skipping book indexing")
        log.warning("Create the directory and add your PDF/TXT textbooks, then re-run.")
        return chunks

    book_files = list(BOOKS_DIR.glob("*.pdf")) + list(BOOKS_DIR.glob("*.txt"))
    if not book_files:
        log.warning(f"No .pdf or .txt files found in {BOOKS_DIR.resolve()}")
        return chunks

    for book_path in sorted(book_files):
        log.info(f"  Loading book: {book_path.name}")
        raw  = extract_text_from_file(book_path)
        if not raw.strip():
            log.warning(f"  No text extracted from {book_path.name} — skipping")
            continue

        cleaned    = clean_book_text(raw)
        word_count = len(cleaned.split())
        log.info(f"  {book_path.name}: {word_count:,} words after cleaning")

        book_chunks = chunk_text(cleaned)
        log.info(f"  {book_path.name}: {len(book_chunks)} chunks")

        for i, chunk_text_str in enumerate(book_chunks):
            chunks.append({
                # app.py backward-compatible fields
                # Books don't have timestamps — set to -1 so temporal penalty is neutral
                "start"       : -1.0,
                "end"         : -1.0,
                "text"        : chunk_text_str,
                # Metadata
                "source"      : "book",
                "source_boost": 0.0,    # no boost — slides take priority
                "book_name"   : book_path.stem,
                "chunk_index" : i,
                "slide_id"    : f"book__{book_path.stem}__chunk{i}",
            })

    log.info(f"Books total: {len(chunks)} chunks")
    return chunks


# ════════════════════════════════════════════════════════════════════════════
# INDEX BUILDING
# ════════════════════════════════════════════════════════════════════════════
def build_faiss_index(chunks, model):
    texts = [c["text"] for c in chunks]
    log.info(f"Embedding {len(texts)} chunks with {EMB_MODEL} ...")

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=64,
    )

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    log.info(f"FAISS index: {index.ntotal} vectors × {dim} dims")
    return index


# ════════════════════════════════════════════════════════════════════════════
# SAVE
# ════════════════════════════════════════════════════════════════════════════
def save_store(index, chunks):
    faiss.write_index(index, str(STORE_DIR / "faiss.index"))
    log.info(f"Saved faiss.index ({index.ntotal} vectors)")

    (STORE_DIR / "chunks.json").write_text(
        json.dumps(chunks, indent=2, ensure_ascii=False)
    )
    log.info(f"Saved chunks.json ({len(chunks)} entries)")

    slide_chunks = [c for c in chunks if c["source"] == "slide"]
    book_chunks  = [c for c in chunks if c["source"] == "book"]

    (STORE_DIR / "meta.json").write_text(
        json.dumps({
            "embedding_model"  : EMB_MODEL,
            "n_chunks"         : len(chunks),
            "n_slide_chunks"   : len(slide_chunks),
            "n_book_chunks"    : len(book_chunks),
            "index_dim"        : index.d,
            "seconds_per_slide": SECONDS_PER_SLIDE,
            "book_chunk_words" : BOOK_CHUNK_WORDS,
            "book_overlap_words": BOOK_OVERLAP_WORDS,
            "slide_boost"      : SLIDE_BOOST,
            "lectures_indexed" : sorted(set(
                c.get("lecture_name","") for c in slide_chunks
            )),
            "books_indexed"    : sorted(set(
                c.get("book_name","") for c in book_chunks
            )),
        }, indent=2)
    )
    log.info("Saved meta.json")


# ════════════════════════════════════════════════════════════════════════════
# VERIFY
# ════════════════════════════════════════════════════════════════════════════
def verify(index, chunks, model):
    log.info("\n--- Retrieval sanity check ---")
    queries = [
        "What is filtered backprojection in CT?",
        "How does MRI use magnetic fields?",
        "What is the Fourier transform in imaging?",
    ]
    for q in queries:
        q_emb = model.encode([q], convert_to_numpy=True, normalize_embeddings=True)
        D, I  = index.search(q_emb.astype(np.float32), 4)
        log.info(f"\nQuery: {q}")
        for rank, (idx, score) in enumerate(zip(I[0], D[0])):
            c      = chunks[idx]
            source = c["source"].upper()
            label  = c.get("lecture_name", c.get("book_name", "?"))
            log.info(f"  [{rank+1}] {source:<6} score={score:.4f}  {label}")
            log.info(f"       {c['text'][:100]}...")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
def main():
    log.info("=" * 60)
    log.info("  ALIVE-ED  rag_index_v3.py  (slides + books)")
    log.info("=" * 60)

    if PDF_BACKEND is None:
        log.warning("No PDF library detected. Book PDFs will be skipped.")
        log.warning("Install with: pip install pymupdf --break-system-packages")

    # Load slides
    slide_chunks = load_slides()

    # Load books
    book_chunks = load_books()

    # Merge — slides first so their indices come first in the FAISS store
    all_chunks = slide_chunks + book_chunks

    if not all_chunks:
        log.error("No chunks loaded — check LECTURES_DIR and BOOKS_DIR paths")
        return

    log.info(f"\nTotal: {len(all_chunks)} chunks "
             f"({len(slide_chunks)} slides + {len(book_chunks)} book)")

    # Load embedding model
    log.info(f"\nLoading embedding model: {EMB_MODEL}")
    model = SentenceTransformer(EMB_MODEL, device="cuda")

    # Build index
    index = build_faiss_index(all_chunks, model)

    # Save
    save_store(index, all_chunks)

    # Verify
    verify(index, all_chunks, model)

    log.info("\n" + "=" * 60)
    log.info("  Done. RESTART FastAPI server (Terminal D) to load new index.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
