"""E5 (R2): text-path latency under concurrent load. Each request = question embedding (all-mpnet-base-v2, CPU) +
top-4 retrieval over the 1,117-segment index + generation by the ALIVE model served with vLLM (OpenAI-compatible
completions endpoint, deployment prompt, temperature 0, max 200 tokens). Concurrency levels 1/4/8/16/32,
64 requests per level (questions from the test set). Reports per-request latency p50/p95/max and throughput.
Usage: python latency_load.py http://127.0.0.1:8013/v1/completions merged_alive contexts_2x2.jsonl xb_deployed.npy chunks.json out.json
"""
import asyncio, json, sys, time
import aiohttp
import numpy as np
from sentence_transformers import SentenceTransformer

url, model_name, ctx_f, xb_f, chunks_f, out_f = sys.argv[1:7]
C = [json.loads(l) for l in open(ctx_f)]; XB = np.load(xb_f); chunks = json.load(open(chunks_f))
emb = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")
SYSTEM = ("You are a teaching assistant answering a student. Use ONLY the lecture context provided. "
          "Answer in 1-3 clear, precise sentences.")


def prompt(q, ctx):
    return ("<|begin_of_text|>" f"<|start_header_id|>system<|end_header_id|>\n{SYSTEM}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\nLECTURE CONTEXT:\n{ctx}\n\nQUESTION:\n{q}\n\nANSWER:<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n")


async def one(session, q):
    t0 = time.perf_counter()
    v = await asyncio.to_thread(emb.encode, [q], normalize_embeddings=True)
    ctx = "\n\n".join(chunks[j]["text"] for j in np.argsort(-(XB @ v[0]))[:4])
    t1 = time.perf_counter()
    async with session.post(url, json={"model": model_name, "prompt": prompt(q, ctx), "temperature": 0.0, "max_tokens": 200,
                                       "stop": ["<|eot_id|>", "<|end_of_text|>", "\n\n"]}) as r:
        await r.json()
    t2 = time.perf_counter()
    return t1 - t0, t2 - t0


async def level(c, qs):
    sem = asyncio.Semaphore(c)
    async with aiohttp.ClientSession() as s:
        async def run(q):
            async with sem:
                return await one(s, q)
        t0 = time.perf_counter(); res = await asyncio.gather(*[run(q) for q in qs]); wall = time.perf_counter() - t0
    tot = np.array([r[1] for r in res]); ret = np.array([r[0] for r in res])
    return {"concurrency": c, "n": len(qs), "p50_s": round(float(np.percentile(tot, 50)), 3), "p95_s": round(float(np.percentile(tot, 95)), 3),
            "max_s": round(float(tot.max()), 3), "retrieval_p50_s": round(float(np.percentile(ret, 50)), 3),
            "throughput_req_per_s": round(len(qs) / wall, 2)}


qs = [c["question"] for c in C[:64]]
asyncio.run(level(1, qs[:4]))  # warm-up
out = [asyncio.run(level(c, qs)) for c in [1, 4, 8, 16, 32]]
json.dump(out, open(out_f, "w"), indent=1); print(json.dumps(out, indent=1))
