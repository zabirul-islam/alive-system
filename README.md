# ALIVE System

Source code, evaluation scripts, and audit artifacts for **ALIVE**
(Avatar-Lecture Interactive Video Engine), an interactive video
system that lets students pause a recorded lecture, ask a question,
and receive a concise, lecture-grounded answer optionally delivered
by an instructor-style talking-head avatar.

The dataset, retrieval index, and lecture transcripts that this
repository consumes are released as a companion HuggingFace
dataset (see the paper's Code and Data Availability section).

## Repository layout

```
github/
├── backend/                 deployed FastAPI + vLLM + Celery service
│   ├── app.py               retrieval + answer generation + audio/avatar routing
│   ├── llm_client.py        chat (3-5 sentence) and avatar (1 sentence) modes
│   ├── rag_index.py         legacy slide-only index builder
│   ├── celery_app.py        Celery queue config
│   ├── tasks.py             avatar synthesis worker
│   ├── alive_scripts/
│   │   ├── rag_index_v3.py  slide + book FAISS builder (current production)
│   │   ├── train_lora.py    LoRA fine-tuning entry point
│   │   ├── data_pipeline/   QA generation, filtering, splitting
│   │   ├── evaluation/      LLM, retrieval, latency benchmarks
│   │   ├── configs/         deployment config templates
│   │   └── figures/         figure-generation utilities
│   └── rag_store/           ready-to-load FAISS index (also on HF)
├── analysis/                paper Pass 2 statistical re-analysis
│   ├── stat_redo.py         paired Wilcoxon, bootstrap CI, Cohen's d, McNemar
│   ├── survey_redo.py       Cronbach alpha + Feldt CI + Holm-Bonferroni
│   └── *.json / *.csv       reproducible outputs
├── alive-eval-bundle/       per-example LLM, judge, and retrieval outputs
│   ├── data/                held-out test split + corpus stats
│   ├── raw/                 per-question Base, Base+RAG, ALIVE answers and scores
│   ├── results/             aggregated summaries used in paper tables
│   └── scripts/             portable rerun helpers
├── legacy-mimic-results/    prior aggregated results retained for provenance
└── docs/
    ├── startup_commands.md  five-terminal launch sequence
    └── reproducibility.md   seeds, package versions, expected outputs
```

## Revision materials (IEEE TLT, TLT-2026-05-0547)

The scripts, per-example outputs, human ratings, and relevance labels behind the
revised manuscript and its supplementary material are in [`revision/`](revision/README.md).

## Pipeline at a glance

1. `backend/alive_scripts/data_pipeline/generate_qa.py` walks the
   lecture corpus, merges all slide texts per lecture, and prompts a
   Llama-3.1-8B-Instruct teacher to produce roughly five QA pairs
   per slide with balanced difficulty (basic / intermediate /
   advanced).
2. `backend/alive_scripts/data_pipeline/clean_and_split.py` filters
   meta-questions, deduplicates, and produces the 4,191 / 508 / 508
   train / val / test split.
3. `backend/alive_scripts/train_lora.py` fine-tunes
   Llama-3.1-8B-Instruct with LoRA (rank 32, alpha 64, dropout 0.05,
   target modules `q_proj k_proj v_proj o_proj gate_proj up_proj
   down_proj`, learning rate 2e-4 cosine schedule, 3 epochs,
   effective batch 16, bf16+TF32, gradient checkpointing).
4. `backend/alive_scripts/rag_index_v3.py` rebuilds the FAISS index
   over both lecture slide texts and the medical-imaging textbook
   (`all-mpnet-base-v2` embeddings; book chunks of 300 words with 60
   words overlap; uniform 30-second slide timestamps; slide
   source-boost 0.05).
5. `backend/app.py` serves the system: pause timestamp + question →
   retrieval (top-4 after a 4k candidate re-rank with
   temporal-and-source-adjusted score, λ = 0.4) → vLLM call → text
   answer → optional avatar synthesis via Celery + Redis.

## Mapping to paper sections

| Paper section | Code |
| --- | --- |
| §IV Lecture Preparation | `backend/alive_scripts/data_pipeline/generate_qa.py` |
| §V Timestamp-Aware Retrieval | `backend/alive_scripts/rag_index_v3.py`, `backend/app.py::retrieve_context` |
| §VI Domain-Adapted Language Model | `backend/alive_scripts/train_lora.py`, `backend/llm_client.py` |
| §VII Interactive Question Answering | `backend/app.py` (`/ask`, `/ask_avatar`) |
| §VIII Real-Time Avatar Synthesis | `backend/tasks.py` (Celery worker), MuseTalk integration (external) |
| §IX Evaluation (Tables II–IV) | `backend/alive_scripts/evaluation/`, `alive-eval-bundle/`, `analysis/stat_redo.py` |
| §X Classroom Study | `analysis/survey_redo.py` |

## Reproducing the evaluation numbers

```bash
# 1. evaluation environment
conda create -n alive-eval python=3.12
conda activate alive-eval
pip install bert-score rouge-score numpy pandas matplotlib scipy faiss-cpu

# 2. paper numbers from the precomputed bundle
python analysis/stat_redo.py        # writes stat_redo_*.json
python analysis/survey_redo.py      # writes item_stats_v2.csv, composites_v2.csv
                                    # requires the IRB-restricted xlsx; see survey_redo.py

# 3. rerun the LLM scoring against new generations
python alive-eval-bundle/scripts/eval_llm.py \
  --predictions alive-eval-bundle/raw/results_finetuned.json \
  --references  alive-eval-bundle/data/test.jsonl \
  --output      results/rerun_finetuned.json
```

Random seed 42 across all evaluation scripts.

## Citation

```bibtex
@article{islam2026alive,
  title  = {ALIVE: An Avatar-Lecture Interactive Video Engine with
            Content-Aware Retrieval for Real-Time Interaction},
  author = {Islam, Md Zabirul and Manik, Md Motaleb Hossen and
            Marques, Oge and Mohamed, Hisham and Wang, Ge},
  journal= {IEEE Transactions on Learning Technologies},
  year   = {2026},
  note   = {Under review}
}
```


## License

The code in this repository is released under the MIT License. See
`LICENSE`. The companion dataset on HuggingFace is CC BY-NC-SA 4.0;
the medical-imaging textbook PDF retains its original license.
