import re, os, json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

DATA_DIR = Path(__file__).parent / "data"
STORE_DIR = Path(__file__).parent / "rag_store"
STORE_DIR.mkdir(exist_ok=True, parents=True)

SRT_FILE = DATA_DIR / "lecture01.srt"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def parse_srt(srt_text):
    blocks = re.split(r"\n\s*\n", srt_text.strip(), flags=re.M)
    items = []
    for b in blocks:
        lines = [l.strip() for l in b.splitlines() if l.strip()]
        if len(lines) >= 3:
            m = re.match(r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})", lines[1])
            if not m: 
                continue
            def to_sec(ts):
                h, m_, s_ms = ts.split(":")
                s, ms = s_ms.split(",")
                return int(h)*3600 + int(m_)*60 + int(s) + int(ms)/1000.0
            start = to_sec(m.group(1))
            end = to_sec(m.group(2))
            text = " ".join(lines[2:])
            items.append({"start": start, "end": end, "text": text})
    return items

def main():
    srt_text = SRT_FILE.read_text(encoding="utf-8")
    items = parse_srt(srt_text)

    merged = []
    buf, buf_start, buf_end = [], None, None
    max_span = 20.0
    for it in items:
        if not buf:
            buf = [it["text"]]
            buf_start, buf_end = it["start"], it["end"]
        else:
            if it["end"] - buf_start <= max_span:
                buf.append(it["text"])
                buf_end = it["end"]
            else:
                merged.append({"start": buf_start, "end": buf_end, "text": " ".join(buf)})
                buf = [it["text"]]
                buf_start, buf_end = it["start"], it["end"]
    if buf:
        merged.append({"start": buf_start, "end": buf_end, "text": " ".join(buf)})

    model = SentenceTransformer(EMB_MODEL)
    texts = [m["text"] for m in merged]
    emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb.astype(np.float32))

    faiss.write_index(index, str(STORE_DIR / "faiss.index"))
    (STORE_DIR / "chunks.json").write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    (STORE_DIR / "meta.json").write_text(json.dumps({"embedding_model": EMB_MODEL}, indent=2), encoding="utf-8")
    print(f"Indexed {len(merged)} chunks.")

if __name__ == "__main__":
    main()
