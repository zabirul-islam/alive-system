# Reproducing the MIMIC paper

This guide reproduces every numerical claim in the paper from a fresh
checkout. Total compute on the case-study course: ~2 hours on a single
NVIDIA H100 PCIe.

The same recipe applies, unchanged, to any other slide-and-transcript
course you want to adapt.

---

## 0. Hardware and software

| Component   | Version used |
|-------------|--------------|
| GPU         | NVIDIA H100 PCIe (80 GB) |
| OS          | Ubuntu 22.04 |
| CUDA        | 12.1 |
| PyTorch     | 2.3.0 |
| `transformers` | 4.43 |
| `peft`      | 0.12 |
| `trl`       | 0.9 |
| `datasets`  | 2.20 |
| `accelerate`| 0.33 |
| `bitsandbytes` | 0.43 (8-bit AdamW) |
| vLLM        | 0.5.4 |
| Sentence encoder | `all-mpnet-base-v2` (768-dim) |
| FAISS       | `faiss-cpu` |

A pinned `requirements.txt` is provided in the repository.

All runs use a fixed deterministic seed (`42`). All evaluation and judge
calls use temperature `0.0`.

---

## 1. Inputs

You need:

- Per-lecture slide images (JPEG/PNG) and per-slide transcripts (plain text).
- An H100-class GPU (80 GB) for LoRA fine-tuning and serving.
- API keys for the external judges (GPT-4o via OpenAI, Claude Sonnet 4.5 via
  Anthropic) **only** if you want to reproduce Table 7.

For the case-study course, all 23 lectures are released here:
<https://huggingface.co/datasets/zabir1996/mip-bench/tree/main/Lectures>

---

## 2. Pipeline

The pipeline has eight steps. Each step's exact CLI invocation lives in the
`System/scripts/` directory of the project.

| # | Step | Output |
|---|------|--------|
| 1 | Ingest lecture materials | per-slide structured records |
| 2 | Generate narration scripts | per-slide narration + spotlight events |
| 3 | Generate QA candidates | Bloom-stratified raw QA pairs |
| 4 | Filter and split | 80/10/10 train/val/test JSONL |
| 5 | LoRA fine-tune | merged MIMIC-LM checkpoint |
| 6 | Run inference | held-out generations for Base and MIMIC-LM |
| 7 | Score lexical metrics | BERTScore, ROUGE, factual-error rate |
| 8 | Run external LLM-as-judge | GPT-4o + Claude per-axis scores, win/tie/loss |

The released `dataset_stats.json` reproduces Table 1 of the paper directly.

---

## 3. Per-table reproduction

| Paper table / figure | Reproduced by | Released artefact |
|----------------------|---------------|-------------------|
| Table 1 — dataset stats | step 4 | `results/dataset_stats.json` |
| Table 2 — LoRA config | configuration only | (see paper Table 2) |
| **Table 3 — answer quality** | steps 6 + 7 | computed by scoring step over the released generation logs |
| Table 4 — qualitative examples | step 6 | (in paper) |
| **Table 5 — latency** | latency benchmark | `results/latency_base.json`, `results/latency_finetuned.json`, `results/latency_results.json` |
| **Table 6 — retrieval quality** | retrieval evaluation on the 50-question hand-labelled set | `results/ndcg_results.json`, `results/rag_results.scores.json` |
| **Table 7 — LLM-as-judge** | step 8 | `results/judge_gpt-4o.summary.json`, `results/judge_claude-sonnet-4-5.summary.json` |
| Figure 7, 8, 9, 10 | post-processing of the above | (regenerated from generation logs and judge JSONs) |

The full row-by-row mapping is in [`results/eval_summary.md`](results/eval_summary.md).

---

## 4. Determinism

Under the same seed and software stack, MIMIC-LM's open-book BERTScore F1
of **0.7685** reproduces to within ±0.001 across three independent re-runs
on the same H100.

---

## 5. Adapting to a new course

The pipeline is course-agnostic. To adapt:

1. Replace the input slides and transcripts with your own (step 1 input).
2. Re-run steps 1–8 unchanged.
3. The QA-generation prompt (Appendix A.1 of the paper) and the runtime
   classroom prompt (Appendix A.2) are domain-neutral and do not need
   modification beyond optional terminology hints.

If you adapt MIMIC to a new course, please cite the paper and consider
contributing your derived QA dataset back as an additional config of
the Hugging Face dataset.
