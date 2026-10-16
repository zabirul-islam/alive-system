"""E6 (R2 portability): transcribe lecture videos with Whisper (faster-whisper large-v3) to get time-stamped
segments directly from the audio, without the subtitle file produced by the avatar service.
Usage: python e6_asr.py videos/lectureNN.mp4 ... -> out2/asr_lectureNN.json  [{start, end, text}, ...]
"""
import json, sys
from pathlib import Path
from faster_whisper import WhisperModel

model = WhisperModel("large-v3", device="cuda", compute_type="float16")
for v in sys.argv[1:]:
    segs, _ = model.transcribe(v, language="en", beam_size=5, vad_filter=True)
    out = [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()} for s in segs]
    json.dump(out, open(f"out2/asr_{Path(v).stem}.json", "w"))
    print(v, len(out), out[-1]["end"] if out else 0, flush=True)
