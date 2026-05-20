import requests

VLLM_URL = "http://127.0.0.1:8000/v1/completions"
VLLM_MODEL = "meta-llama/Llama-3.2-3B"

def call_vllm(prompt: str, max_tokens: int = 128) -> str:
    full_prompt = (
        "You are a helpful teaching assistant.\n\n"
        "Use the context below to answer the question.\n"
        "Answer in ONE clear sentence.\n\n"
        f"{prompt}\n\n"
        "### Answer:\n"
    )

    payload = {
        "model": VLLM_MODEL,
        "prompt": full_prompt,
        "temperature": 0.3,
        "max_tokens": max_tokens,
        "stop": ["\n\n"],
    }

    r = requests.post(VLLM_URL, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["text"].strip()

