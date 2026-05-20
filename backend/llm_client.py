# llm_client.py
# Updated: conversation history support + book-aware system prompts
import requests
import re
from typing import Optional, List, Dict

# ── vLLM config ───────────────────────────────────────────────────────────────
VLLM_URL   = "http://127.0.0.1:8000/v1/chat/completions"
VLLM_MODEL = "checkpoints/alive-llama-lora/merged"

# Max history turns to include (each turn = 1 user + 1 assistant message)
# Keep low to stay within max_model_len=8192
MAX_HISTORY_TURNS = 3


def _force_one_sentence(text: str, max_words: int = 60) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    if "." in text:
        text = text.split(".", 1)[0] + "."
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words]) + "."
    return text


# ── System prompts ────────────────────────────────────────────────────────────
_SYSTEM_CHAT = (
    "You are a helpful teaching assistant for a medical imaging course at RPI. "
    "Answer using ONLY the lecture context and conversation history provided. "
    "Give a thorough but focused explanation in 3-5 sentences. "
    "Use precise technical terminology from the course material. "
    "If the student asks a follow-up like 'explain more' or 'why?', "
    "elaborate on your previous answer — do not change the topic. "
    "Do NOT repeat the question. Do NOT include timestamps or slide references."
)

_SYSTEM_AVATAR = (
    "You are a teaching assistant answering a student verbally in a medical imaging course. "
    "Use ONLY the lecture context and conversation history provided. "
    "Answer in exactly 1-2 short sentences. Be direct and natural when spoken aloud. "
    "If the student asks a follow-up like 'explain more' or 'why?', "
    "give one concrete elaboration on the previous answer. "
    "Do NOT repeat the question. Do NOT include timestamps or slide references."
)


def call_vllm(
    prompt: str,
    mode: str = "chat",
    max_tokens: int = 512,
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    system_prompt = _SYSTEM_AVATAR if mode == "avatar" else _SYSTEM_CHAT
    if mode == "avatar":
        max_tokens = min(max_tokens, 120)

    # ── Build message list ────────────────────────────────────────────────────
    messages = [{"role": "system", "content": system_prompt}]

    # Add trimmed conversation history before the current turn
    if history:
        trimmed = history[-(MAX_HISTORY_TURNS * 2):]
        for msg in trimmed:
            if msg.get("role") in ("user", "assistant") and msg.get("content"):
                messages.append({
                    "role"   : msg["role"],
                    "content": str(msg["content"])[:800],
                })

    # Current user turn (retrieved context + question)
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model"      : VLLM_MODEL,
        "messages"   : messages,
        "temperature": 0.2,
        "max_tokens" : max_tokens,
        "stop"       : ["<|eot_id|>"],
    }

    r = requests.post(VLLM_URL, json=payload, timeout=30)
    r.raise_for_status()

    raw = r.json()["choices"][0]["message"]["content"].strip()

    if mode == "avatar":
        raw = _force_one_sentence(raw, max_words=60)

    return raw
