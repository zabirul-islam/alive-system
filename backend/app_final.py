# app.py
# ALIVE Backend — FastAPI + vLLM + Celery
#
# TTS strategy:
#   PRIMARY  → ElevenLabs API  (voice-cloned, best quality)
#   FALLBACK → gTTS            (Google free TTS, no API key, no credits)
#              Activates automatically on any ElevenLabs failure.
#              Install once: pip install gTTS
#
# All other features:
#   - UUID temp file IDs (no race condition under concurrent load)
#   - Dedicated tmp/ subdirectory
#   - /ask  → 3-5 sentence detailed answer (chat panel)
#   - /ask_avatar → 1-2 sentence spoken answer (avatar)
#   - Conversation history passed to LLM for multi-turn context
#   - Query expansion for vague follow-ups ("explain more?")
#   - Async Whisper + ElevenLabs (ThreadPoolExecutor, non-blocking)
#   - Embedding model on CPU (no GPU 0 race with vLLM)
#   - Redis startup retry loop
#   - Celery concurrency=2 (safe with MuseTalk threading.Lock)

import os
import json
import time
import uuid
import shutil
import logging
import asyncio
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, List, Dict

import requests
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from faster_whisper import WhisperModel

from fastapi import FastAPI, UploadFile, File, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from dotenv import load_dotenv
from celery.result import AsyncResult

from llm_client import call_vllm
from celery_app import celery_app
from tasks import dummy_job, sadtalker_job, gpu1_diagnostic_test, musetalk_job

load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

# ==================== PATHS ====================
APP_DIR     = Path(__file__).parent
STORE_DIR   = APP_DIR / "rag_store"
DATA_DIR    = APP_DIR / "data"
RESULTS_DIR = APP_DIR / "results"
TMP_DIR     = APP_DIR / "tmp"

RESULTS_DIR.mkdir(exist_ok=True)
TMP_DIR.mkdir(exist_ok=True)

REDIS_PATH = "/home/islamm11/anaconda3/envs/sadtalker/bin/redis-server"

# ==================== FASTAPI SETUP ====================
app = FastAPI(title="ALIVE Backend (FastAPI + vLLM + Celery)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/media",   StaticFiles(directory=str(DATA_DIR)),    name="media")
app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")

_executor = ThreadPoolExecutor(max_workers=8)


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse(DATA_DIR / "favicon.ico")


# ==================== REDIS STARTUP ====================
def ensure_redis_running():
    import socket
    for attempt in range(5):
        try:
            s = socket.socket()
            s.settimeout(0.5)
            s.connect(("127.0.0.1", 6379))
            s.close()
            print("Redis is running.")
            return
        except Exception:
            if attempt == 0:
                print("Redis not detected — starting it...")
                subprocess.Popen([REDIS_PATH])
            time.sleep(0.6)
    print("WARNING: Redis may not be up yet.")


@app.on_event("startup")
def startup_event():
    ensure_redis_running()


# ==================== TTS: ElevenLabs (primary) + gTTS (fallback) ===========

ELEVENLABS_API_KEY  = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = "X4ryQ1a2NClP7lLYw0as"
ELEVENLABS_MODEL_ID = "eleven_multilingual_v2"


def elevenlabs_tts_to_wav(text: str, wav_path: Path) -> None:
    if not ELEVENLABS_API_KEY:
        raise RuntimeError("ELEVENLABS_API_KEY not set")
    url     = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
    headers = {
        "xi-api-key"  : ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
        "Accept"      : "audio/wav",
    }
    payload = {
        "text"          : text,
        "model_id"      : ELEVENLABS_MODEL_ID,
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
    }
    r = requests.post(url, json=payload, headers=headers, timeout=60)
    r.raise_for_status()
    wav_path.write_bytes(r.content)


def gtts_tts_to_wav(text: str, wav_path: Path) -> None:
    """
    Fallback: Google TTS (free, no API key, no credits).
    gTTS outputs MP3 — convert to WAV with ffmpeg.
    """
    try:
        from gtts import gTTS
    except ImportError:
        raise RuntimeError("gTTS not installed. Run: pip install gTTS")

    tmp_mp3 = wav_path.with_suffix(".mp3")
    try:
        gTTS(text=text, lang="en", slow=False).save(str(tmp_mp3))
        rc, out = _run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(tmp_mp3),
            "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le",
            str(wav_path),
        ])
        if rc != 0 or not wav_path.exists():
            raise RuntimeError(f"ffmpeg MP3→WAV failed: {out}")
    finally:
        tmp_mp3.unlink(missing_ok=True)


def tts_to_wav(text: str, wav_path: Path) -> str:
    """
    Unified TTS with automatic fallback.
    Returns "elevenlabs" or "gtts" — whichever engine succeeded.
    """
    # Primary: ElevenLabs
    if ELEVENLABS_API_KEY:
        try:
            elevenlabs_tts_to_wav(text, wav_path)
            return "elevenlabs"
        except Exception as e:
            logging.warning(
                f"[TTS] ElevenLabs failed ({type(e).__name__}: {e}). "
                f"Switching to gTTS fallback."
            )

    # Fallback: gTTS
    try:
        gtts_tts_to_wav(text, wav_path)
        logging.info("[TTS] gTTS fallback succeeded.")
        return "gtts"
    except Exception as e:
        raise RuntimeError(f"Both TTS engines failed. Last error: {e}")


# ==================== RAG / EMBEDDINGS ====================
EMB_MODEL = json.loads((STORE_DIR / "meta.json").read_text())["embedding_model"]
index     = faiss.read_index(str(STORE_DIR / "faiss.index"))
chunks    = json.loads((STORE_DIR / "chunks.json").read_text(encoding="utf-8"))
emb_model = SentenceTransformer(EMB_MODEL, device="cpu")

# ==================== WHISPER (ASR) ====================
whisper = WhisperModel("small", device="cpu", compute_type="int8")


# ==================== STARTUP CLEANUP ====================
def cleanup_orphan_tmp_files() -> None:
    for pattern in ("*.webm", "*.wav", "*.mp3"):
        for p in TMP_DIR.glob(pattern):
            try:
                p.unlink()
            except Exception:
                pass
    for pattern in ("tmp_*.webm", "tmp_*.wav", "fixed_*.webm", "tmp_tts_*.wav"):
        for p in APP_DIR.glob(pattern):
            try:
                p.unlink()
            except Exception:
                pass


@app.on_event("startup")
def startup_cleanup():
    cleanup_orphan_tmp_files()


# ==================== MODELS ====================
class AskPayload(BaseModel):
    question:             str
    current_time:         Optional[float] = None
    conversation_history: Optional[List[Dict[str, str]]] = None


# ==================== RAG HELPERS ====================
def retrieve_context(
    query: str,
    current_time: Optional[float],
    k: int = 4,
    history: Optional[List[Dict[str, str]]] = None,
) -> List[dict]:
    # Query expansion for short follow-ups
    effective_query = query
    if history and len(query.split()) < 6:
        last_a = next(
            (m["content"] for m in reversed(history) if m["role"] == "assistant"),
            None
        )
        if last_a:
            effective_query = " ".join(last_a.split()[:60]) + " " + query

    q_emb = emb_model.encode(
        [effective_query], convert_to_numpy=True, normalize_embeddings=True
    ).astype(np.float32)

    D, I = index.search(q_emb, k * 4)
    candidates = []
    for idx, score in zip(I[0], D[0]):
        chunk    = chunks[idx]
        adjusted = score + chunk.get("source_boost", 0.0)
        if current_time is not None and chunk.get("start", -1) >= 0:
            c_mid     = 0.5 * (chunk["start"] + chunk["end"])
            adjusted -= 0.4 * abs(c_mid - current_time) / 60.0
        candidates.append((adjusted, chunk))

    candidates.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in candidates[:k]]


def build_prompt(question: str, ctx_list: List[dict]) -> str:
    ctx_text = "\n\n".join([c["text"] for c in ctx_list])
    return f"LECTURE CONTEXT:\n{ctx_text}\n\nSTUDENT QUESTION:\n{question}"


# ==================== /ask ====================
@app.post("/ask")
def ask(payload: AskPayload):
    ctx    = retrieve_context(payload.question, payload.current_time, k=4,
                              history=payload.conversation_history)
    prompt = build_prompt(payload.question, ctx)
    answer = call_vllm(prompt, mode="chat", history=payload.conversation_history)
    return {"answer": answer, "context": ctx}


# ==================== /ask_avatar ====================
@app.post("/ask_avatar")
def ask_avatar(payload: AskPayload):
    ctx    = retrieve_context(payload.question, payload.current_time, k=4,
                              history=payload.conversation_history)
    prompt = build_prompt(payload.question, ctx)
    answer = call_vllm(prompt, mode="avatar", history=payload.conversation_history)
    return {"answer": answer, "context": ctx}


# ==================== FFMPEG HELPERS ====================
def _run(cmd: list) -> tuple:
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        return p.returncode, (p.stdout or "")
    except Exception as e:
        return 999, f"subprocess_failed: {e}"


def webm_to_wav_safely(tmp_webm: Path, tmp_wav: Path, tmp_id: str) -> tuple:
    fixed_webm = TMP_DIR / f"fixed_{tmp_id}.webm"
    rc1, out1  = _run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                        "-i", str(tmp_webm), "-c", "copy", str(fixed_webm)])
    if rc1 == 0 and fixed_webm.exists() and fixed_webm.stat().st_size > 0:
        rc2, out2 = _run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                           "-i", str(fixed_webm), "-vn", "-ac", "1", "-ar", "16000",
                           "-acodec", "pcm_s16le", str(tmp_wav)])
        fixed_webm.unlink(missing_ok=True)
        if rc2 == 0 and tmp_wav.exists() and tmp_wav.stat().st_size > 2000:
            return True, out1 + "\n" + out2
        return False, out1 + "\n" + out2
    fixed_webm.unlink(missing_ok=True)
    rc3, out3 = _run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                       "-i", str(tmp_webm), "-vn", "-ac", "1", "-ar", "16000",
                       "-acodec", "pcm_s16le", str(tmp_wav)])
    if rc3 == 0 and tmp_wav.exists() and tmp_wav.stat().st_size > 2000:
        return True, out1 + "\n" + out3
    return False, out1 + "\n" + out3


def _transcribe_wav(wav_path: Path) -> str:
    try:
        segments, _ = whisper.transcribe(
            str(wav_path), beam_size=5, vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 600, "speech_pad_ms": 200},
            language="en", condition_on_previous_text=False, temperature=0.0,
        )
        return " ".join(seg.text.strip() for seg in segments if seg.text).strip()
    except Exception as e:
        logging.exception(f"Whisper failed: {e}")
        return ""


# ==================== /transcribe ====================
@app.post("/transcribe")
async def transcribe(audio_file: UploadFile = File(...)):
    start_t  = time.time()
    tmp_id   = uuid.uuid4().hex
    tmp_webm = TMP_DIR / f"tmp_{tmp_id}.webm"
    tmp_wav  = TMP_DIR / f"tmp_{tmp_id}.wav"
    try:
        tmp_webm.write_bytes(await audio_file.read())
        ok, log = webm_to_wav_safely(tmp_webm, tmp_wav, tmp_id)
        if not ok:
            logging.warning(f"ffmpeg decode failed:\n{log}")
            return {"text": "", "elapsed_s": round(time.time() - start_t, 3),
                    "debug": "ffmpeg_decode_failed"}
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(_executor, _transcribe_wav, tmp_wav)
        return {"text": text, "elapsed_s": round(time.time() - start_t, 3)}
    finally:
        tmp_webm.unlink(missing_ok=True)
        tmp_wav.unlink(missing_ok=True)


# ==================== /avatar_tts ====================
@app.post("/avatar_tts")
async def avatar_tts(payload: dict = Body(...)):
    text = (payload.get("text") or "").strip()
    if not text:
        return {"error": "empty text"}

    job_id   = uuid.uuid4().hex
    wav_path = TMP_DIR / f"tts_{job_id}.wav"

    loop = asyncio.get_event_loop()
    try:
        engine = await loop.run_in_executor(_executor, tts_to_wav, text, wav_path)
        logging.info(f"[avatar_tts] engine={engine}")
    except Exception as e:
        return {"error": "tts_failed", "detail": str(e)}

    task = musetalk_job.delay(str(wav_path), job_id)
    return {
        "task_id"   : task.id,
        "status"    : "queued",
        "job_id"    : job_id,
        "audio_path": str(wav_path),
        "tts_engine": engine,
    }


# ==================== JOB STATUS ====================
@app.get("/job_status/{task_id}")
def job_status(task_id: str):
    res  = AsyncResult(task_id, app=celery_app)
    data = {"task_id": task_id, "state": res.state, "result": None}
    if res.ready():
        data["result"] = res.result
    return data


# ==================== HEALTH ====================
@app.get("/health")
def health():
    try:
        from gtts import gTTS  # noqa
        gtts_status = "available"
    except ImportError:
        gtts_status = "not installed — run: pip install gTTS"

    return {
        "status"         : "ok",
        "tts_primary"    : "elevenlabs" if ELEVENLABS_API_KEY else "no key set",
        "tts_fallback"   : f"gtts ({gtts_status})",
    }


@app.get("/")
def home():
    return {
        "status" : "ok",
        "message": "ALIVE Backend (FastAPI + vLLM + Celery)",
        "routes" : ["/ask", "/ask_avatar", "/transcribe", "/avatar_tts",
                    "/job_status/{task_id}", "/delete_video", "/health"],
    }


# ==================== VIDEO CLEANUP ====================
@app.post("/delete_video")
def delete_video(payload: dict = Body(...)):
    deleted   = []
    video_url = payload.get("video_url")
    job_id    = None

    if video_url:
        try:
            video_path = APP_DIR / video_url.lstrip("/")
            if video_path.exists():
                llm_dir = video_path.parent
                if llm_dir.exists() and llm_dir.is_dir():
                    shutil.rmtree(llm_dir, ignore_errors=True)
                    deleted.append(str(llm_dir))
                if llm_dir.name.startswith("llm_"):
                    job_id = llm_dir.name.replace("llm_", "")
        except Exception as e:
            logging.warning(f"Video cleanup failed: {e}")

    wav_candidates = []
    if payload.get("audio_path"):
        ap = Path(payload["audio_path"])
        wav_candidates.append(ap if ap.is_absolute() else APP_DIR / ap)
    if job_id:
        for d in (TMP_DIR, APP_DIR):
            wav_candidates.append(d / f"tts_{job_id}.wav")
            wav_candidates.extend(d.glob(f"tts_{job_id}*.wav"))
            wav_candidates.extend(d.glob(f"tmp_tts_{job_id}*.wav"))

    for ap in set(wav_candidates):
        try:
            for _ in range(3):
                if ap.exists() and ap.is_file():
                    ap.unlink()
                    deleted.append(str(ap))
                    break
                time.sleep(0.3)
        except Exception as e:
            logging.warning(f"Failed to delete {ap}: {e}")

    return {"status": "ok", "deleted": deleted}


# ==================== TEST / DIAGNOSTIC ====================
@app.post("/enqueue_test")
def enqueue_test(payload: dict = Body(...)):
    return {"task_id": dummy_job.delay(payload.get("text", "test")).id,
            "status": "queued"}

@app.post("/gpu_test")
def gpu_test():
    task = gpu1_diagnostic_test.delay(20)
    return {"task_id": task.id, "status": "queued",
            "note": "GPU1 diagnostic running for ~20s"}
