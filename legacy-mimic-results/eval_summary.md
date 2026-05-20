# Evaluation results — paper-table mapping

Each row of every numerical table and every reported headline number in the
MIMIC paper is reproduced from one of the JSON files in this directory. This
document is the canonical mapping a referee can use to verify the paper
without rerunning the experiments.

All evaluations are over the same 508-question held-out test split released
on the Hugging Face Hub:
<https://huggingface.co/datasets/zabir1996/mimic-medical-imaging-qa>.

---

## Paper Table 3 — LLM answer quality

Source files:

- `judge_gpt-4o.summary.json` (per-axis means, win/tie/loss vs base)
- `judge_claude-sonnet-4-5.summary.json` (independent judge, second model)
- (BERTScore / ROUGE / factual-error per-row values are computed by the
  released scoring script over the released generation logs; see
  `REPRODUCE.md` for the exact recipe.)

| Paper field         | Source key                                | Value     |
|---------------------|-------------------------------------------|-----------|
| Base BERTScore F1   | computed by scorer over Base outputs      | 0.7372 ± 0.0942 |
| RAG BERTScore F1    | computed by scorer over Base+RAG outputs  | 0.7443 ± 0.0951 |
| MIMIC-LM BERTScore F1 | computed by scorer over MIMIC-LM outputs | 0.7685 ± 0.0961 |
| Slide-grounded factual-error rate (Base / RAG / MIMIC-LM) | scorer over each system | 0.20% / 0.39% / 0.00% |

## Paper Table 5 — Response latency

Source files: `latency_base.json`, `latency_finetuned.json`,
`latency_results.json`.

| Paper field                | Source file              | Source key                | Value |
|----------------------------|--------------------------|---------------------------|------:|
| Base mean latency (s)      | `latency_base.json`      | `llm_base_mean`           | 1.591 |
| Base std (s)               | `latency_base.json`      | `llm_base_std`            | 0.208 |
| MIMIC-LM mean latency (s)  | `latency_finetuned.json` | `llm_finetuned_mean`      | 0.595 |
| MIMIC-LM std (s)           | `latency_finetuned.json` | `llm_finetuned_std`       | 0.102 |
| Speedup factor             | derived                  | base_mean / fine_mean     | 2.67× |
| Base+RAG per-question mean | from RAG eval logs       | mean over 508 test items  | 0.574 s |

## Paper Table 6 — Retrieval quality

Source file: `ndcg_results.json` and `rag_results.scores.json`.

The retrieval ground truth is the 50-question hand-labelled set described
in Section 7.3 of the paper. Two retrieval conditions are reported:
text-only cosine similarity, and a timestamp-aware variant.

## Paper Table 7 — External LLM-as-judge evaluation

Source files: `judge_gpt-4o.summary.json` and
`judge_claude-sonnet-4-5.summary.json`. Both are direct exports of the
aggregation step (per-axis means and pairwise win/tie/loss rates over
508 test questions).

| Paper cell                 | Source key (per file)               |
|----------------------------|-------------------------------------|
| F (Factuality), Base mean  | `F_base_mean`                       |
| F (Factuality), Fine mean  | `F_fine_mean`                       |
| F win / tie / loss rates   | `F_win_rate_fine` / `F_tie_rate` / `F_loss_rate` |
| P / G / A axes             | analogous keys with `P_` / `G_` / `A_` prefixes |

Direct values from the released summaries:

| Axis | GPT-4o Base | GPT-4o Fine | Δ (GPT-4o) | Claude Base | Claude Fine | Δ (Claude) |
|------|------------:|------------:|-----------:|------------:|------------:|-----------:|
| F    | 3.348       | 3.506       | +0.157     | 3.552       | 3.874       | +0.321     |
| P    | 3.892       | 4.138       | +0.246     | 3.781       | 4.118       | +0.337     |
| G    | 4.514       | 4.687       | +0.173     | 3.489       | 4.079       | +0.590     |
| A    | 3.528       | 3.807       | +0.280     | 3.655       | 4.091       | +0.436     |

## Paper dataset statistics (Table 1)

Source file: `dataset_stats.json`.

- Total pairs: 5,207 — 80/10/10 train/val/test split.
- Difficulty distribution: 1,245 basic / 2,013 intermediate / 1,949 advanced.
- 23 source lectures (per-lecture counts in `dataset_stats.json`).

---

## Notes for verification

- All judges were called at temperature 0.0 with deterministic seed.
- Per-question raw judge scores are not released in this directory because
  they may contain prompt verbatim and are large; rerunning the released
  scripts (`REPRODUCE.md`) regenerates them locally.
- Latency benchmarks use a fixed 150-token generation budget at temperature
  0.0 against a self-hosted vLLM endpoint on a single NVIDIA H100 PCIe.
