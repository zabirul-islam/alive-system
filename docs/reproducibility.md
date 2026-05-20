# Reproducibility

Every quantitative claim in the ALIVE manuscript can be regenerated
from the artefacts in this repository. This document lists the
seeds, package versions, hardware, and expected outputs.

## Hardware

- 2 × NVIDIA H100 PCIe (1 for the language model, 1 for the
  MuseTalk avatar server and Celery worker)
- Host CPU for FastAPI, Whisper-small (int8), and the
  sentence-transformer encoder
- Redis on the host (started automatically by `app.py` if absent)

## Random seeds

- Train / val / test split (`data_pipeline/clean_and_split.py`):
  seed 42
- LoRA fine-tuning (`train_lora.py`): cosine schedule with warmup 50
- Evaluation (`evaluation/eval_llm.py`, `evaluation/eval_retrieval.py`,
  `analysis/stat_redo.py`): seed 42; bootstrap with 1,000 resamples
- Decoding: temperature 0.0 in evaluation, 0.2 in deployment

## Software environment

Python 3.12. Key packages:

| Package | Purpose |
| --- | --- |
| `vllm` | LLM serving |
| `transformers`, `peft`, `bitsandbytes` | LoRA fine-tuning |
| `sentence-transformers` (`all-mpnet-base-v2`) | retrieval embeddings |
| `faiss-cpu` | similarity search index |
| `faster-whisper` (model `small`, compute `int8`) | ASR |
| `fastapi`, `uvicorn`, `pydantic` | API layer |
| `celery`, `redis` | async avatar dispatch |
| `pymupdf` (or `pdfplumber`) | book PDF ingestion |
| `rouge_score`, `bert_score`, `scipy`, `pandas` | evaluation |

## Headline numbers

| Claim | Reproduce by running |
| --- | --- |
| BERTScore F1 0.737 → 0.769, ROUGE-L 0.422 → 0.476 | `python analysis/stat_redo.py` on `alive-eval-bundle/raw/results_*.json` |
| NDCG@1 0.300 → 0.640 | `python alive-eval-bundle/scripts/eval_retrieval.py` |
| LLM-as-judge scores (Table III) | `alive-eval-bundle/results/judge_*.summary.json` |
| Mean LLM latency 1.591 s → 0.595 s | `alive-eval-bundle/results/latency_*.json` |
| Survey composites and per-item Holm-Bonferroni p-values | `python analysis/survey_redo.py` (requires the IRB-restricted xlsx) |

## Output files written by the evaluation scripts

- `analysis/stat_redo_llm.json`
- `analysis/stat_redo_judge.json`
- `analysis/stat_redo_retrieval.json`
- `analysis/item_stats_v2.csv`
- `analysis/composites_v2.csv`
- `analysis/survey_redo.json`

## What is NOT redistributed

- Raw classroom-survey responses (covered by IRB-2325)
- Raw lecture videos (covered by course copyright)
- MuseTalk weights and the avatar reference frames
- Merged LoRA + base model weights (size; will be uploaded before
  publication)
