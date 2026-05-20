# tasks.py
import os
import time
import subprocess
from pathlib import Path

from celery_app import celery_app

APP_DIR = Path(__file__).parent
RESULTS_DIR = APP_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ---- SadTalker paths ----
SADTALKER_DIR = Path("/home/islamm11/islamm11/alive/backend/SadTalker")
SADTALKER_PY = "/home/islamm11/anaconda3/envs/sadtalker_rt/bin/python"

AVATAR_IMAGE = SADTALKER_DIR / "examples/source_image/avatar1.png"
CHECKPOINT_DIR = SADTALKER_DIR / "checkpoints"


def _find_video(output_dir: Path, timeout_s: int = 240) -> Path:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        mp4s = list(output_dir.glob("*.mp4"))
        if mp4s:
            web = [p for p in mp4s if p.stem.endswith("_web")]
            return web[0] if web else mp4s[0]
        time.sleep(1)
    raise RuntimeError("SadTalker timeout: no mp4 produced")


def _ffmpeg_web_compatible(in_path: Path) -> Path:
    if in_path.stem.endswith("_web"):
        return in_path

    out_path = in_path.with_name(in_path.stem + "_web.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(in_path),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        "-crf", "20",
        "-c:a", "aac",
        "-movflags", "+faststart",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    return out_path


@celery_app.task(bind=True)
def dummy_job(self, text: str):
    time.sleep(2)
    return f"Processed: {text}"


@celery_app.task(bind=True)
def sadtalker_job(self, wav_path: str, job_id: str):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    wav_path = Path(wav_path)
    if not wav_path.exists():
        raise FileNotFoundError(wav_path)

    output_dir = RESULTS_DIR / f"llm_{job_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        SADTALKER_PY,
        str(SADTALKER_DIR / "inference.py"),
        "--driven_audio", str(wav_path),
        "--source_image", str(AVATAR_IMAGE),
        "--checkpoint_dir", str(CHECKPOINT_DIR),
        "--result_dir", str(output_dir),
        "--still",
        "--preprocess", "full",
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "1"

    subprocess.run(
        cmd,
        cwd=str(SADTALKER_DIR),
        env=env,
        check=True,
    )

    raw_video = _find_video(output_dir)
    web_video = _ffmpeg_web_compatible(raw_video)

    rel = web_video.relative_to(APP_DIR)
    return {
        "status": "success",
        "video_url": f"/{rel.as_posix()}",
        "output_dir": str(output_dir),
        "wav_path": str(wav_path),
    }


# ============================
# GPU DIAGNOSTIC TEST TASK
# ============================
@celery_app.task(bind=True)
def gpu1_diagnostic_test(self, sleep_s: int = 15):
    """
    Purpose:
      - Verify that SadTalker runtime python sees CUDA
      - Verify that CUDA_VISIBLE_DEVICES=1 is respected
      - Force a real GPU kernel so nvidia-smi shows activity
    """

    import subprocess
    import os
    import time
    from datetime import datetime

    def log(msg):
        print(f"[GPU-TEST {datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "1"
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    # ---- Python-level CUDA proof ----
    proof_code = r"""
import torch, time
print("python:", torch.__file__)
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("cuda_device_name:", torch.cuda.get_device_name(0))

# force real GPU work
x = torch.randn(8192, 8192, device="cuda")
y = torch.matmul(x, x)
torch.cuda.synchronize()
print("matrix multiply done")

time.sleep(%d)
""" % sleep_s

    cmd = [
        SADTALKER_PY,
        "-c",
        proof_code
    ]

    log("Launching GPU diagnostic subprocess...")
    t0 = time.time()

    subprocess.run(
        cmd,
        env=env,
        cwd=str(SADTALKER_DIR),
        check=True,
    )

    log(f"GPU diagnostic finished in {time.time() - t0:.1f}s")

    return {
        "status": "ok",
        "message": "GPU1 diagnostic completed",
        "sleep_s": sleep_s,
    }


# ---- MuseTalk paths ----
MUSETALK_SERVER    = 'http://localhost:9100'
MUSETALK_VIDEO_DIR = Path('/home/islamm11/islamm11/alive/backend/MuseTalk/results/v15/avatars/avatar_wang_v15/vid_output')

@celery_app.task(bind=True)
def musetalk_job(self, wav_path: str, job_id: str):
    import shutil, requests as req
    t0 = time.time()

    wav_path = Path(wav_path)
    if not wav_path.exists():
        raise FileNotFoundError(f'WAV not found: {wav_path}')

    output_dir = RESULTS_DIR / f'llm_{job_id}'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Call persistent musetalk server (models already loaded)
    r = req.post(f'{MUSETALK_SERVER}/generate',
                 json={'wav_path': str(wav_path), 'job_id': job_id},
                 timeout=120)
    r.raise_for_status()
    data = r.json()
    if data.get('status') != 'success':
        raise RuntimeError(f'MuseTalk server error: {data}')

    # ── option A: skip re-encode, just move (faster, uncomment to use) ──
    # raw_video = Path(data['video_path'])
    # dest = output_dir / f'{job_id}.mp4'
    # shutil.move(str(raw_video), str(dest))

    # ── option B: re-encode for web compatibility (active) ────────────
    raw_video = Path(data['video_path'])
    web_video = _ffmpeg_web_compatible(raw_video)
    dest = output_dir / web_video.name
    shutil.copy2(str(web_video), str(dest))
    raw_video.unlink(missing_ok=True)
    if web_video != raw_video:
        web_video.unlink(missing_ok=True)

    rel = dest.relative_to(APP_DIR)
    return {
        'status': 'success',
        'video_url': f'/{rel.as_posix()}',
        'latency_s': round(time.time() - t0, 2),
        'engine': 'musetalk',
    }
