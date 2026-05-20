# ALIVE — Five-terminal launch sequence

The deployed system is a small fleet of cooperating processes. The
order matters: avatar and language-model servers must be ready
before the FastAPI layer accepts traffic, and the Celery worker
needs Redis to be up.

```bash
# Terminal A — MuseTalk avatar server on GPU 1
# wait for "Ready! 1348 frames cached" before starting Terminal D
conda activate musetalk && cd ~/islamm11/alive/backend/MuseTalk
CUDA_VISIBLE_DEVICES=1 python3 musetalk_server.py

# Terminal B — vLLM serving the merged ALIVE LoRA checkpoint on GPU 0
conda activate vllm_deploy && cd ~/islamm11/alive/backend/alive_scripts
CUDA_VISIBLE_DEVICES=0 vllm serve checkpoints/alive-llama-lora/merged \
  --host 127.0.0.1 --port 8000 --max-model-len 8192 \
  --gpu-memory-utilization 0.85

# Terminal C — Celery worker on GPU 1, concurrency 2
conda activate sadtalker_rt && cd ~/islamm11/alive/backend
CUDA_VISIBLE_DEVICES=1 celery -A celery_app.celery_app worker \
  --loglevel=info --concurrency=2

# Terminal D — FastAPI host on port 9000
conda activate sadtalker && cd /home/islamm11/islamm11/alive/backend
python -m uvicorn app:app --host 0.0.0.0 --port 9000

# Terminal E — Cloudflare tunnel for classroom access
cd /home/islamm11/islamm11/alive/backend
cloudflared tunnel --url http://127.0.0.1:9000
```

Two GPUs total: GPU 0 runs vLLM only, GPU 1 hosts both the MuseTalk
server and the Celery worker (the worker contends only when an
avatar job is dispatched). Redis is started automatically by the
FastAPI process if it is not already running.
