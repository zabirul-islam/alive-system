# ALIVE revision (IEEE TLT, TLT-2026-05-0547): scripts, results, and human ratings

This folder holds everything needed to check the numbers added in the major
revision of the ALIVE paper. The supplementary material refers to the
scripts here as `revision/scripts`. Lecture transcripts, slide texts, the
question-answer corpus, and the FAISS index are in the companion dataset
<https://huggingface.co/datasets/zabir1996/alive-medical-imaging>.

## Layout

| Folder | Contents |
|---|---|
| `scripts/` | Analysis and experiment scripts (vLLM 0.11.0, PyTorch 2.8.0, Transformers 4.57.1, PEFT 0.18.1, bert-score 0.3.13). |
| `results/` | All JSON/JSONL outputs, including per-example generations (`gen/`, `gen2/`, `e4/`), per-answer judge outputs (`judge/`), and logs. |
| `generated/` | LaTeX tables written by `scripts/make_supp_tables.py` from `results/`. |
| `figures/` | Regenerated supplementary latency figure (Fig. S2). |
| `annotation/` | Second annotator's slide-relevance labels for the 50 retrieval questions, with the instructions they followed. |
| `expert/` | The two external experts' rating workbooks (100 items each), the rater guide, and the A/B unblinding key. |
| `data/srt/` | Subtitle files used to align slide timestamps. |

## Where each result comes from

| Paper / supplement | Script(s) | Result file(s) |
|---|---|---|
| 2x2 answer quality (Table II; Table SVI) | `gen_2x2.py`, `score_2x2.py`, `nli_segments.py` | `score_2x2.json`, `score_2x2_per_example.jsonl`, `nli_seg_2x2.json`, `gen/` |
| Near-duplicates and grouped / held-out splits (Table I; Table SVIII) | `leakage_rescore.py`, `build_splits.py`, `train_lora_split.py`, `score_split.py`, `cluster_split.py` | `leakage_rescore.json`, `splits/` |
| External judges (Table III; Table SIX) | `judge_v2.py`, `aggregate_judge2.py` | `judge_2x2.json`, `judge/` |
| Expert review (Sec. IX-B; Table SX) | `make_expert_sheet.py`, `repair_rater_return.py`, `analyze_expert.py` | `expert_analysis.json`, `expert/` |
| Additional conditions E1-E3 (Table IV; Table SXI) | `gen_2x2.py`, `build_e3_contexts.py`, `aggregate_e123.py`, `nli_segments.py` | `e123/`, `judge_e123.json`, `contexts_e3.jsonl`, `gen2/` |
| Cross-segment questions E4 (Table SXII) | `build_e4.py`, `e4_pr.py` | `e4_items.jsonl`, `contexts_e4.jsonl`, `e4/` |
| Retrieval (Table V; Tables SXIII-SXIV; Fig. S6) | `align_slides.py`, `retrieval_v2.py`, `fig_lambda.py` | `slide_timings.json`, `alignment_quality.json`, `retrieval_v2*.json` |
| Second annotator (Sec. XI-C; Table SXV) | `make_free_annotation_sheet.py`, `analyze_retrieval_irr.py` | `retrieval_irr.json`, `retrieval_gt_*.json`, `retrieval_v2_{annotator2,intersection,union}*.json` |
| Slide timings from raw ASR, E6 (Table SI) | `e6_asr.py`, `e6_compare.py` | `e6/`, `e6_portability.json` |
| Latency under load, E5 (Table SIV) | `latency_load.py` | `e5/latency_load.json` |
| Survey composites, per-item tests, input-method counts (Q4) and the post hoc Q1+Q2 usability check (Table SXIX; Sec. X) | `survey_r17.py` (uses `analysis/survey_redo.py`) | `survey_r17.json` |
| Component latency (Table SIII; Fig. S2) | `fig_latency.py` | `backend/alive_scripts/evaluation/latency_{base,finetuned}.json`, `figures/fig_latency.*` |
| All supplementary tables | `make_supp_tables.py` | `generated/*.tex` |

`e4_items_v1_discarded.jsonl` and `e4_items_v2_discarded.jsonl` are the two
earlier E4 generations described in the supplement; they were discarded
before any answers were produced.

## Expert ratings

Answers were shown to the raters as A and B in an order randomized per item.
`expert/unblinding_key.json` maps each item to the system behind A and B;
`scripts/analyze_expert.py` uses it to unblind the workbooks. The raters are
anonymous. The Notes sheet of each workbook records how the file was handled:
rater 1 returned a CSV export that had split some long text cells, and the
ratings were transcribed into the template unchanged.

## Not included

- **Raw survey responses.** The survey form collected names, student IDs, and
  emails to award bonus points. Under the approved IRB protocol (IRB-2325)
  these responses are not public. `survey_r17.py` therefore runs only with the
  private file; its outputs are in `results/survey_r17.json` and
  `analysis/`.
- **Adapter and merged weights, and the avatar server scripts.** These will
  be released on acceptance, as stated in the paper.
- **Lecture audio.** The E6 check transcribed the audio of Lectures 2, 13,
  and 19; the ASR outputs are in `results/e6/`.

## Rerunning

Scripts resolve paths relative to this folder. Lecture texts and the index
come from the Hugging Face dataset. Generation and scoring need one GPU
(we used an NVIDIA H100 80 GB); judging needs `OPENAI_API_KEY` and
`ANTHROPIC_API_KEY` in the environment. To rebuild the supplementary tables
from the released results:

```bash
python revision/scripts/make_supp_tables.py
```
