# ALIVE evaluation bundle

Portable, self-contained snapshot of every evaluation script, raw model
output, aggregated summary, and test-split file used to produce the
quantitative results in Section~7 of the ALIVE paper (*ALIVE: An
Avatar-Lecture Interactive Video Engine with Content-Aware Retrieval
for Real-Time Interaction*, IEEE Transactions on Learning Technologies,
under review). Copy this folder to any system and the comparison
between the base Llama-3.1-8B-Instruct model, the Base+RAG baseline,
and the domain-adapted ALIVE model can be regenerated, inspected, or
reused as a starting point for follow-up ALIVE experiments.

Source machine: H100 PCIe server. Date of last refresh: see file mtimes.

---

## Layout

```
mimic-eval-bundle/
├── README.md                                  ← this file
├── scripts/
│   ├── eval_llm.py                            ← Base vs ALIVE-model lexical scorer
│   │                                            (BERTScore, ROUGE, slide-grounded
│   │                                            factual error rate; ALIVE paper §7.1)
│   ├── eval_retrieval.py                      ← Semantic vs timestamp-aware retrieval
│   │                                            (NDCG@k, Hit@5; ALIVE paper §7.3)
│   ├── benchmark_latency.py                   ← Single-request LLM latency benchmark
│   │                                            (ALIVE paper §7.4)
│   └── generate_figures.py                    ← Regenerates the per-axis judge and
│                                                difficulty-breakdown figures used
│                                                in the ALIVE supplementary material
├── results/                                   ← Aggregated summaries (ALIVE Sec 7)
│   ├── judge_gpt-4o.summary.json              ← Table 2 (tab:judge, §7.2):
│   │                                            GPT-4o judge means + W/T/L
│   ├── judge_claude-sonnet-4-5.summary.json   ← Table 2 (tab:judge, §7.2):
│   │                                            Claude Sonnet 4.5 judge means + W/T/L
│   ├── latency_base.json                      ← Table 4 (tab:latency, §7.4):
│   │                                            base LLM mean/std (20 iters)
│   ├── latency_finetuned.json                 ← Table 4 (tab:latency, §7.4):
│   │                                            ALIVE LLM mean/std (20 iters)
│   ├── latency_results.json                   ← Table 4 (tab:latency, §7.4):
│   │                                            combined + 2.67× speedup reference
│   ├── rag_results.scores.json                ← Table 1 Base+RAG row (§7.1)
│   │                                            and Table 3 text-only row (§7.3)
│   └── ndcg_results.json                      ← Table 3 (tab:retrieval, §7.3):
│                                                NDCG@1/@3/@5 + Hit@1/@5 for
│                                                semantic-only and timestamp-aware
├── raw/                                       ← Per-example outputs (large, regenerable)
│   ├── results_base.json                      ← 508 base-model answers + per-example scores
│   │                                            (Table 1 Base row, §7.1)
│   ├── results_finetuned.json                 ← 508 ALIVE-model answers + per-example scores
│   │                                            (Table 1 ALIVE row, §7.1)
│   ├── judge_gpt-4o.jsonl                     ← Per-pair GPT-4o judgments (Table 2)
│   ├── judge_claude-sonnet-4-5.jsonl          ← Per-pair Claude judgments (Table 2)
│   ├── rag_results.jsonl                      ← Per-question retrieved chunks + Base+RAG
│   │                                            generated answers (§7.1, §7.3)
│   └── retrieval_gt.json                      ← 50-question hand-labelled retrieval truth
│                                                used in §7.3
└── data/                                      ← Test split + dataset stats
    ├── test.jsonl                             ← 508 pairs, {instruction, input, output}
    │                                            (held-out test set used in §7.1, §7.2)
    ├── test_full.jsonl                        ← 508 pairs + slide metadata (8 cols)
    ├── test_full_enriched.jsonl               ← test_full + retrieval context
    │                                            (Base+RAG input file, §7.1)
    ├── test_full_enriched.stats.json          ← Enrichment statistics
    └── stats.json                             ← Full corpus stats: 5,207 lecture-derived
                                                  QA pairs across 23 lectures, used in
                                                  ALIVE §6.1 (lecture-derived QA dataset)
```

---

## Models compared

The three configurations evaluated in ALIVE Section~7 are:

- **Base:** `meta-llama/Meta-Llama-3.1-8B-Instruct`, served via vLLM on
  the ALIVE language-model GPU with no fine-tuning. This is the
  "Base model" column of Table~1 (`tab:llm_comparison`) and the
  "Base" rows of Table~2 (`tab:judge`) and Table~4 (`tab:latency`).
- **Base + RAG (FAISS top-$k=4$):** Base model with content-aware
  retrieval over the same 1,117-segment slide-aligned lecture corpus
  using the `all-mpnet-base-v2` sentence encoder (768-dim). Reported as
  the "Base + RAG" column of Table~1.
- **ALIVE model:** Base + LoRA adapter (rank 32, scaling factor 64,
  3 epochs, learning rate $2\!\times\!10^{-4}$ with cosine schedule)
  fine-tuned on the 4,191-pair ALIVE training split and merged into
  a single deployable checkpoint for vLLM serving (ALIVE §6.2). This
  is the "ALIVE model" column of Tables 1, 2, and the "ALIVE LLM" row
  of Table~4.

All three configurations were evaluated on the same 508-example open-book
held-out test set at temperature `0.0` with deterministic seed `42`. The
deployed system uses two NVIDIA H100~PCIe GPUs (one serving the LLM,
one serving the persistent MuseTalk avatar) as described in ALIVE §3
and §8.

---

## How each ALIVE artefact maps to files

| ALIVE paper artefact                                | File(s) in this bundle                                                        |
|-----------------------------------------------------|-------------------------------------------------------------------------------|
| Dataset description (ALIVE §6.1, `tab:dataset_stats`)| `data/stats.json`                                                             |
| Table 1 — Answer quality (`tab:llm_comparison`, §7.1)| `raw/results_base.json`, `raw/results_finetuned.json`, `results/rag_results.scores.json` (Base+RAG row) |
| Table 2 — LLM-as-judge (`tab:judge`, §7.2)          | `results/judge_gpt-4o.summary.json`, `results/judge_claude-sonnet-4-5.summary.json` (+ raw per-pair in `raw/judge_*.jsonl`) |
| Table 3 — Retrieval accuracy (`tab:retrieval`, §7.3)| `results/ndcg_results.json`, `results/rag_results.scores.json`, `raw/retrieval_gt.json` |
| Table 4 — System latency (`tab:latency`, §7.4)      | `results/latency_base.json`, `results/latency_finetuned.json`, `results/latency_results.json` |
| Table 5 — Ablation (`tab:ablation`, §7.5)           | Combines: `results/ndcg_results.json` (retrieval ablation row), `raw/results_base.json` + `raw/results_finetuned.json` (LLM ablation row), `results/latency_*.json` (avatar/LLM latency reference) |
| Supplementary per-axis judge plots, difficulty breakdown | Regenerated by `scripts/generate_figures.py`                                |

The headline ALIVE numbers also reproduce directly from this bundle:

| ALIVE headline claim                              | Source                                                                                  |
|---------------------------------------------------|-----------------------------------------------------------------------------------------|
| BERTScore F1 $0.737 \to 0.769$ ($+0.031$)        | `raw/results_base.json`, `raw/results_finetuned.json` aggregated by `scripts/eval_llm.py` |
| ROUGE-L $0.422 \to 0.476$ ($+0.054$)             | same                                                                                    |
| Slide-grounded factual error $0.20\% \to 0.00\%$ | same                                                                                    |
| NDCG@1 $0.300 \to 0.640$                          | `results/ndcg_results.json`, `raw/retrieval_gt.json`                                    |
| Mean LLM latency $1.591\,\text{s} \to 0.595\,\text{s}$ ($2.67\times$) | `results/latency_base.json`, `results/latency_finetuned.json`, `results/latency_results.json` |
| Both external judges score ALIVE above Base on every axis | `results/judge_gpt-4o.summary.json`, `results/judge_claude-sonnet-4-5.summary.json` |

---

## Reusing this bundle in a new system / follow-up ALIVE work

```bash
# 1. Copy the folder
scp -r mimic-eval-bundle/ user@newhost:~/

# 2. (On new host) Python environment
conda create -n alive-eval python=3.12
conda activate alive-eval
pip install bert-score rouge-score numpy pandas matplotlib

# 3. Inspect any aggregated summary
cat results/judge_gpt-4o.summary.json
cat results/latency_finetuned.json

# 4. Regenerate the per-axis / difficulty-breakdown figures
python scripts/generate_figures.py

# 5. Rerun lexical scoring against new generations
python scripts/eval_llm.py \
  --predictions raw/results_finetuned.json \
  --references  data/test.jsonl \
  --output      results/rerun_finetuned.json

# 6. Rerun the ALIVE latency benchmark against a new vLLM endpoint
python scripts/benchmark_latency.py \
  --endpoint http://localhost:8080/v1/chat/completions \
  --iterations 20 \
  --max-tokens 150 \
  --temperature 0.0

# 7. Rerun the retrieval comparison (semantic vs timestamp-aware)
python scripts/eval_retrieval.py \
  --ground-truth raw/retrieval_gt.json \
  --output       results/rerun_retrieval.json
```

---

## What is *not* in this bundle

- The training-set splits (`train.jsonl`, `val.jsonl`) — only the test
  split is needed to reproduce the ALIVE answer-quality and judge
  comparisons. Full 5,207-pair dataset on Hugging Face:
  <https://huggingface.co/datasets/zabir1996/mimic-medical-imaging-qa>.
- The merged ALIVE model weights (LoRA-adapted Llama-3.1-8B-Instruct)
  — too large for a portable folder. Available on request.
- The 23 lecture slide images / transcripts and the MuseTalk reference
  frames used by the avatar server — also too large; lecture materials
  on Hugging Face:
  <https://huggingface.co/datasets/zabir1996/mip-bench/tree/main/Lectures>.
- The classroom user-study survey responses (ALIVE Section~8) — held
  separately under the IRB-2325 data-use plan and not part of this
  technical-eval bundle.

---

## Provenance

All files in this bundle originate from the ALIVE source project at
`data/`. The
aggregated summaries in `results/` were also previously released
publicly at
<https://github.com/zabirul-islam/mimic/tree/main/results> as part of
a separate course-derivation-pipeline paper that shares the same
underlying QA corpus and evaluation scaffolding; this bundle is the
ALIVE-specific cut, with all file descriptions, table mappings, and
configuration labels rewritten against the ALIVE manuscript.

If reused in a new paper or follow-up ALIVE experiment, please cite:

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
