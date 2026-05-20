import io, os, json, time, asyncio, binascii, subprocess, wave
from typing import Optional, List
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import requests
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from faster_whisper import WhisperModel
import edge_tts
import pyttsx3



#  -----ElevenLab------------
import logging
import sys

from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
import os
from pathlib import Path
from llm_client import call_vllm


load_dotenv()




# ==================== ELEVENLABS CONFIG ====================

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# Your cloned Wang voice ID
ELEVENLABS_VOICE_ID = "X4ryQ1a2NClP7lLYw0as"

ELEVENLABS_MODEL_ID = "eleven_multilingual_v2"


# ==================== CONFIG ====================

VOICE_MAP = {
    "male": "en-US-GuyNeural",
    "female": "en-US-JennyNeural",
}

APP_DIR = Path(__file__).parent
STORE_DIR = APP_DIR / "rag_store"
DATA_DIR = APP_DIR / "data"
RESULTS_DIR = APP_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

#OLLAMA_MODEL = "llama3.2:3b"
#OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

EMB_MODEL = json.loads((STORE_DIR / "meta.json").read_text())["embedding_model"]
SYSTEM_PROMPT = ( "You are a helpful teaching assistant. Answer based on the provided lecture context. " "If the question is off-topic or context is insufficient, say what you can and be concise." )


index = faiss.read_index(str(STORE_DIR / "faiss.index"))
chunks = json.loads((STORE_DIR / "chunks.json").read_text(encoding="utf-8"))
emb_model = SentenceTransformer(EMB_MODEL)

WHISPER_MODEL_SIZE = "small"
whisper = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")


# ==================== APP SETUP ====================

app = FastAPI(title="Local Interactive Lecture Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== CONTEXT RETRIEVAL ====================

class AskPayload(BaseModel):
    question: str
    current_time: Optional[float] = None


def retrieve_context(query: str, current_time: Optional[float], k: int = 4) -> List[dict]:
    q_emb = emb_model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    D, I = index.search(q_emb, k * 4)
    candidates = [chunks[i] for i in I[0]]

    if current_time is not None:
        adjusted = []
        for c, s in zip(candidates, D[0]):
            c_mid = 0.5 * (c["start"] + c["end"])
            time_penalty = abs(c_mid - current_time) / 60.0
            adjusted.append((s - 0.4 * time_penalty, c))
        adjusted.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in adjusted[:k]]

    return candidates[:k]

def build_prompt(question: str, ctx_list: List[dict]) -> str:
    ctx_text = "\n\n".join([f"[{round(c['start'])}-{round(c['end'])}s] {c['text']}" for c in ctx_list])
    return f"<<SYSTEM>> {SYSTEM_PROMPT}\n<<CONTEXT>>\n{ctx_text}\n\n<<QUESTION>> {question}\n<<INSTRUCTIONS>>"


@app.post("/ask")
def ask(payload: AskPayload):
    ctx = retrieve_context(payload.question, payload.current_time, k=4)
    prompt = build_prompt(payload.question, ctx)
    #answer = ollama_chat(prompt)
    answer = call_vllm(prompt)
    return {"answer": answer, "context": ctx}


# ==================== SPEECH RECOGNITION ====================

@app.post("/transcribe")
async def transcribe(audio_file: UploadFile = File(...)):
    start_t = time.time()
    audio_bytes = await audio_file.read()

    tmp_id = str(int(time.time() * 1000))
    tmp_webm = APP_DIR / f"tmp_{tmp_id}.webm"
    tmp_wav = APP_DIR / f"tmp_{tmp_id}.wav"

    with open(tmp_webm, "wb") as f:
        f.write(audio_bytes)

    os.system(f'ffmpeg -hide_banner -loglevel warning -y -i "{tmp_webm}" -t 10 -ac 1 -ar 16000 -f wav "{tmp_wav}"')

    text = ""
    try:
        segments, info = whisper.transcribe(str(tmp_wav), beam_size=5, vad_filter=False, language="en")
        text = " ".join(seg.text for seg in segments).strip()
    except:
        pass

    try:
        os.remove(tmp_webm)
        os.remove(tmp_wav)
    except:
        pass

    return {"text": text, "elapsed_s": round(time.time() - start_t, 3)}


# ==================== OFFLINE TTS ====================

# @app.post("/tts")
# def tts(payload: dict = Body(...)):
#     text = payload.get("text", "").strip()
#     if not text:
#         return {"error": "empty text"}

#     engine = pyttsx3.init()
#     engine.setProperty("rate", 170)
#     voices = engine.getProperty("voices")
#     voice_req = (payload.get("voice") or "male").lower()

#     if "female" in voice_req and len(voices) > 1:
#         engine.setProperty("voice", voices[1].id)
#     else:
#         engine.setProperty("voice", voices[0].id)

#     wav_path = APP_DIR / "tmp_tts.wav"
#     engine.save_to_file(text, str(wav_path))
#     engine.runAndWait()

#     with open(wav_path, "rb") as f:
#         hex_str = binascii.hexlify(f.read()).decode("ascii")

#     return {"audio_base64": hex_str}


@app.post("/tts")
def tts(payload: dict = Body(...)):
    text = payload.get("text", "").strip()
    if not text:
        return {"error": "empty text"}

    uid = int(time.time() * 1000)
    wav_path = APP_DIR / f"tmp_tts_{uid}.wav"

    elevenlabs_tts_to_wav(text, wav_path)

    with open(wav_path, "rb") as f:
        hex_str = binascii.hexlify(f.read()).decode("ascii")

    wav_path.unlink(missing_ok=True)

    return {"audio_base64": hex_str}



# ==================== SADTALKER AVATAR GENERATION ====================

app.mount("/media", StaticFiles(directory=str(DATA_DIR)), name="media")

SADTALKER_RESULTS = APP_DIR / "results"
SADTALKER_RESULTS.mkdir(exist_ok=True)
app.mount("/results", StaticFiles(directory=str(SADTALKER_RESULTS)), name="results")


# ========= KEEP ONLY SADTALKER LOGGING =========

# import subprocess, time, logging
# from datetime import datetime

# log_file = APP_DIR / "sadtalker_backend_debug.log"

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     handlers=[
#         logging.FileHandler(log_file, mode='a', encoding='utf-8'),
#         logging.StreamHandler(sys.stdout),
#     ]
# )


def elevenlabs_tts_to_wav(text: str, wav_path: Path):
    """
    Generate speech using ElevenLabs cloned voice and save as WAV.
    """
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"

    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/wav",
    }

    payload = {
        "text": text,
        "model_id": ELEVENLABS_MODEL_ID,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75,
        },
    }

    r = requests.post(url, json=payload, headers=headers, timeout=60)
    r.raise_for_status()

    wav_path.write_bytes(r.content)


@app.post("/avatar_tts")
def avatar_tts(payload: dict = Body(...)):
    return {
        "status": "disabled",
        "message": "Avatar generation temporarily disabled (will be re-enabled with job queue)"
    }



# ==================== VIDEO CLEANUP ====================

import shutil

@app.post("/delete_video")
def delete_video(payload: dict = Body(...)):
    """
    Delete generated media (entire llm_xxx directory + audio wav).
    """
    deleted = []

    video_url = payload.get("video_url")
    audio_path = payload.get("audio_path")

    # ----------------------
    # 1. Delete llm_xxx directory
    # ----------------------
    if video_url:
        try:
            rel = video_url.lstrip("/")   # results/llm_xxx/file.mp4
            video_path = APP_DIR / rel

            if video_path.exists():
                llm_dir = video_path.parent  # results/llm_xxx

                if llm_dir.exists() and llm_dir.is_dir():
                    shutil.rmtree(llm_dir, ignore_errors=True)
                    deleted.append(str(llm_dir))

        except Exception as e:
            logging.warning(f"Video cleanup failed: {e}")

    # ----------------------
    # 2. Delete temporary WAV
    # ----------------------
    if audio_path:
        ap = Path(audio_path)
        if not ap.is_absolute():
            ap = APP_DIR / ap

        if ap.exists() and ap.is_file():
            try:
                ap.unlink()
                deleted.append(str(ap))
            except Exception as e:
                logging.warning(f"Failed to delete audio {ap}: {e}")

    return {"deleted": deleted}




# ==================== HEALTH CHECK ====================

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def home():
    return {
        "status": "ok",
        "message": "Backend running successfully",
        "available_routes": [
            "/ask",
            "/transcribe",
            "/tts",
            "/avatar_tts",
            "/health"
        ],
    }

