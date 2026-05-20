"""
Pass 2 statistical re-analysis for the ALIVE paper.

Reads only existing bundle artifacts under
  ../alive-eval-bundle/{raw,results}
and emits paired Wilcoxon, bootstrap 95% CIs, Cohen's d, Cliff's delta,
Wilson CIs, and McNemar tests for the headline tables.

Outputs:
  stat_redo_llm.json
  stat_redo_judge.json
  stat_redo_retrieval.json

No new model runs. No survey data. CPU-only.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
from rouge_score import rouge_scorer
from scipy import stats

ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT.parent / "alive-eval-bundle"
RAW = BUNDLE / "raw"
RES = BUNDLE / "results"
OUT = ROOT

RNG = np.random.default_rng(seed=42)
N_BOOT = 1000


# ----------------------------- helpers -----------------------------

def bootstrap_mean_ci(x: Sequence[float], n_boot: int = N_BOOT, alpha: float = 0.05) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    n = len(x)
    idx = RNG.integers(0, n, size=(n_boot, n))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(x.mean()), float(lo), float(hi)


def bootstrap_diff_ci(a: Sequence[float], b: Sequence[float], n_boot: int = N_BOOT, alpha: float = 0.05) -> tuple[float, float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert len(a) == len(b)
    n = len(a)
    idx = RNG.integers(0, n, size=(n_boot, n))
    diffs = (b[idx] - a[idx]).mean(axis=1)
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float((b - a).mean()), float(lo), float(hi)


def cohens_d_paired(a: Sequence[float], b: Sequence[float]) -> float:
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else float("nan")


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    gt = (b[:, None] > a[None, :]).sum()
    lt = (b[:, None] < a[None, :]).sum()
    return float((gt - lt) / (len(a) * len(b)))


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float, float]:
    if n == 0:
        return (0.0, 0.0, 0.0)
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    halfw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (float(p), float(centre - halfw), float(centre + halfw))


def paired_wilcoxon(a: Sequence[float], b: Sequence[float]) -> dict:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = b - a
    nz = diff[diff != 0]
    if len(nz) < 1:
        return {"W": float("nan"), "p": 1.0, "n_pairs": int(len(a)), "n_nonzero": 0}
    res = stats.wilcoxon(a, b, alternative="two-sided", zero_method="wilcox", correction=False, method="auto")
    return {"W": float(res.statistic), "p": float(res.pvalue), "n_pairs": int(len(a)), "n_nonzero": int(len(nz))}


def mcnemar_exact(b_flags: Sequence[bool], f_flags: Sequence[bool]) -> dict:
    # 2x2: rows = Base flag (T/F), cols = Fine flag (T/F).
    b = np.asarray(b_flags, dtype=bool)
    f = np.asarray(f_flags, dtype=bool)
    n10 = int(((b) & (~f)).sum())   # Base error, Fine ok
    n01 = int(((~b) & (f)).sum())   # Base ok,    Fine error
    n11 = int((b & f).sum())
    n00 = int(((~b) & (~f)).sum())
    n_disc = n10 + n01
    if n_disc == 0:
        p = 1.0
    else:
        # exact binomial: p( X <= min(n10,n01) under p=0.5, two-sided)
        k = min(n10, n01)
        p = 2 * stats.binom.cdf(k, n_disc, 0.5)
        p = min(p, 1.0)
    return {"n10_base_err_fine_ok": n10, "n01_base_ok_fine_err": n01, "n11": n11, "n00": n00, "p_exact": float(p)}


# ----------------------------- LLM eval -----------------------------

def llm_eval() -> dict:
    base = json.load(open(RAW / "results_base.json"))
    fine = json.load(open(RAW / "results_finetuned.json"))
    rag = json.load(open(RES / "rag_results.scores.json"))
    rag_lines = [json.loads(l) for l in open(RAW / "rag_results.jsonl")]

    assert base["questions"] == fine["questions"]
    n = base["n_examples"]
    assert n == 508

    out: dict = {"n": n}

    # ---- BERTScore F1: per-example available for Base + Fine
    b_f1 = base["bertscore"]["f1_values"]
    f_f1 = fine["bertscore"]["f1_values"]
    diff_mean, lo, hi = bootstrap_diff_ci(b_f1, f_f1)
    out["bertscore_f1"] = {
        "base_mean": float(np.mean(b_f1)),
        "base_std":  float(np.std(b_f1, ddof=1)),
        "fine_mean": float(np.mean(f_f1)),
        "fine_std":  float(np.std(f_f1, ddof=1)),
        "delta_mean": diff_mean,
        "delta_ci95": [lo, hi],
        "wilcoxon": paired_wilcoxon(b_f1, f_f1),
        "cohens_d_paired": cohens_d_paired(b_f1, f_f1),
        "cliffs_delta": cliffs_delta(b_f1, f_f1),
    }

    # Aggregate-only metrics
    out["bertscore_precision"] = {
        "base": base["bertscore"]["precision"],
        "fine": fine["bertscore"]["precision"],
        "delta": float(fine["bertscore"]["precision"] - base["bertscore"]["precision"]),
        "note": "per-example precision not stored; no paired test available",
    }
    out["bertscore_recall"] = {
        "base": base["bertscore"]["recall"],
        "fine": fine["bertscore"]["recall"],
        "delta": float(fine["bertscore"]["recall"] - base["bertscore"]["recall"]),
        "note": "per-example recall not stored; no paired test available",
    }

    # ---- ROUGE per-example: recompute from stored predictions/references
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    refs = base["references"]
    assert refs == fine["references"]
    b_preds = base["predictions"]
    f_preds = fine["predictions"]
    rag_preds = [r["base_rag_answer"] for r in rag_lines]

    def rouge_vec(preds, refs):
        r1, r2, rl = [], [], []
        for p, r in zip(preds, refs):
            s = scorer.score(r, p)
            r1.append(s["rouge1"].fmeasure)
            r2.append(s["rouge2"].fmeasure)
            rl.append(s["rougeL"].fmeasure)
        return np.array(r1), np.array(r2), np.array(rl)

    b_r1, b_r2, b_rl = rouge_vec(b_preds, refs)
    f_r1, f_r2, f_rl = rouge_vec(f_preds, refs)
    rag_r1, rag_r2, rag_rl = rouge_vec(rag_preds, refs)

    for name, b_arr, f_arr, rag_arr in [
        ("rouge1", b_r1, f_r1, rag_r1),
        ("rouge2", b_r2, f_r2, rag_r2),
        ("rougeL", b_rl, f_rl, rag_rl),
    ]:
        diff_mean, lo, hi = bootstrap_diff_ci(b_arr, f_arr)
        rb_mean, rb_lo, rb_hi = bootstrap_mean_ci(rag_arr)
        out[name] = {
            "base_mean": float(b_arr.mean()),
            "fine_mean": float(f_arr.mean()),
            "rag_mean":  float(rb_mean),
            "rag_ci95":  [rb_lo, rb_hi],
            "delta_mean_fine_vs_base": diff_mean,
            "delta_ci95_fine_vs_base": [lo, hi],
            "wilcoxon_fine_vs_base": paired_wilcoxon(b_arr, f_arr),
            "cohens_d_paired_fine_vs_base": cohens_d_paired(b_arr, f_arr),
        }

    # ---- Slide-grounded factual error: exact counts, Wilson CI, McNemar
    b_flags = np.asarray(base["factual_error_regex"]["flag_per_example"], dtype=bool)
    f_flags = np.asarray(fine["factual_error_regex"]["flag_per_example"], dtype=bool)
    rag_err_count = rag["factual_errors"]
    n_rag = 508  # RAG eval ran on same 508 items

    b_k = int(b_flags.sum())
    f_k = int(f_flags.sum())
    p_b, lo_b, hi_b = wilson_ci(b_k, n)
    p_f, lo_f, hi_f = wilson_ci(f_k, n)
    p_r, lo_r, hi_r = wilson_ci(rag_err_count, n_rag)

    out["factual_error"] = {
        "base":  {"k": b_k, "n": n, "rate": p_b, "ci95": [lo_b, hi_b]},
        "fine":  {"k": f_k, "n": n, "rate": p_f, "ci95": [lo_f, hi_f]},
        "rag":   {"k": rag_err_count, "n": n_rag, "rate": p_r, "ci95": [lo_r, hi_r]},
        "mcnemar_base_vs_fine": mcnemar_exact(b_flags, f_flags),
        "note": "Metric is a strict substring check for numeric quantities; near floor on this corpus.",
    }

    return out


# ----------------------------- Judge eval -----------------------------

def judge_eval() -> dict:
    out: dict = {}
    for judge in ["gpt-4o", "claude-sonnet-4-5"]:
        recs_all = [json.loads(l) for l in open(RAW / f"judge_{judge}.jsonl")]
        # Drop pairs where any of the 4 axes is missing or non-numeric
        def ok(r):
            for ax in "FPGA":
                for k in ("base_scores", "fine_scores"):
                    v = r.get(k, {}).get(ax)
                    if not isinstance(v, (int, float)):
                        return False
            return True
        dropped = sum(1 for r in recs_all if not ok(r))
        recs = [r for r in recs_all if ok(r)]
        n = len(recs)
        per_axis: dict = {"_n_dropped_for_missing_scores": int(dropped)}
        for axis in ["F", "P", "G", "A"]:
            b_scores = np.array([r["base_scores"][axis] for r in recs], dtype=float)
            f_scores = np.array([r["fine_scores"][axis] for r in recs], dtype=float)
            diff_mean, lo, hi = bootstrap_diff_ci(b_scores, f_scores)
            wins = int((f_scores > b_scores).sum())
            ties = int((f_scores == b_scores).sum())
            losses = int((f_scores < b_scores).sum())
            wp, wlo, whi = wilson_ci(wins, n)
            lp, llo, lhi = wilson_ci(losses, n)
            per_axis[axis] = {
                "base_mean": float(b_scores.mean()),
                "fine_mean": float(f_scores.mean()),
                "delta_mean": diff_mean,
                "delta_ci95": [lo, hi],
                "wilcoxon": paired_wilcoxon(b_scores, f_scores),
                "cohens_d_paired": cohens_d_paired(b_scores, f_scores),
                "cliffs_delta": cliffs_delta(b_scores, f_scores),
                "wins": wins, "ties": ties, "losses": losses,
                "win_rate":  {"p": wp, "ci95": [wlo, whi]},
                "loss_rate": {"p": lp, "ci95": [llo, lhi]},
            }
        out[judge] = {"n": n, "n_total": len(recs_all), "axes": per_axis}
    return out


# ----------------------------- Retrieval eval -----------------------------

def retrieval_eval() -> dict:
    ndcg = json.load(open(RES / "ndcg_results.json"))
    gt = json.load(open(RAW / "retrieval_gt.json"))
    n = len(gt)
    out: dict = {"n_questions": n, "annotator_note": "Retrieval ground truth annotated by one author; no inter-rater reliability is available."}

    for cond_name, cond_key in [("semantic", "text_only"), ("timestamp_aware", "timestamp_aware")]:
        out[cond_name] = {}
        for k in ["1", "3", "5"]:
            vec = np.asarray(ndcg[cond_key][k], dtype=float)
            mean, lo, hi = bootstrap_mean_ci(vec)
            out[cond_name][f"NDCG@{k}"] = {"mean": mean, "ci95": [lo, hi], "n": int(len(vec))}

    # Paired Wilcoxon at each k
    out["paired_tests"] = {}
    for k in ["1", "3", "5"]:
        a = np.asarray(ndcg["text_only"][k], dtype=float)
        b = np.asarray(ndcg["timestamp_aware"][k], dtype=float)
        diff_mean, lo, hi = bootstrap_diff_ci(a, b)
        out["paired_tests"][f"NDCG@{k}"] = {
            "delta_mean": diff_mean,
            "delta_ci95": [lo, hi],
            "wilcoxon": paired_wilcoxon(a, b),
            "cohens_d_paired": cohens_d_paired(a, b),
        }
    return out


# ----------------------------- main -----------------------------

def main() -> None:
    llm = llm_eval()
    judge = judge_eval()
    retr = retrieval_eval()

    (OUT / "stat_redo_llm.json").write_text(json.dumps(llm, indent=2))
    (OUT / "stat_redo_judge.json").write_text(json.dumps(judge, indent=2))
    (OUT / "stat_redo_retrieval.json").write_text(json.dumps(retr, indent=2))

    print("WROTE:", (OUT / "stat_redo_llm.json"))
    print("WROTE:", (OUT / "stat_redo_judge.json"))
    print("WROTE:", (OUT / "stat_redo_retrieval.json"))

    # short console summary
    print("\n--- LLM eval headline ---")
    bf = llm["bertscore_f1"]
    print(f"BERTScore F1: Base {bf['base_mean']:.4f}  Fine {bf['fine_mean']:.4f}  Δ={bf['delta_mean']:+.4f}  CI95 [{bf['delta_ci95'][0]:+.4f},{bf['delta_ci95'][1]:+.4f}]  Wilcoxon p={bf['wilcoxon']['p']:.3e}  d_p={bf['cohens_d_paired']:.3f}")
    rl = llm["rougeL"]
    print(f"ROUGE-L:      Base {rl['base_mean']:.4f}  Fine {rl['fine_mean']:.4f}  Δ={rl['delta_mean_fine_vs_base']:+.4f}  CI95 [{rl['delta_ci95_fine_vs_base'][0]:+.4f},{rl['delta_ci95_fine_vs_base'][1]:+.4f}]  Wilcoxon p={rl['wilcoxon_fine_vs_base']['p']:.3e}  d_p={rl['cohens_d_paired_fine_vs_base']:.3f}")
    fe = llm["factual_error"]
    print(f"Factual error: Base {fe['base']['k']}/{fe['base']['n']} ({100*fe['base']['rate']:.2f}%, CI95 [{100*fe['base']['ci95'][0]:.2f}%,{100*fe['base']['ci95'][1]:.2f}%])  RAG {fe['rag']['k']}/{fe['rag']['n']} ({100*fe['rag']['rate']:.2f}%)  Fine {fe['fine']['k']}/{fe['fine']['n']} ({100*fe['fine']['rate']:.2f}%)")
    print(f"McNemar Base-vs-Fine: p={fe['mcnemar_base_vs_fine']['p_exact']:.4f}")

    print("\n--- Retrieval headline ---")
    print("semantic NDCG@1:", retr["semantic"]["NDCG@1"])
    print("ts-aware NDCG@1:", retr["timestamp_aware"]["NDCG@1"])
    print("Δ NDCG@1:", retr["paired_tests"]["NDCG@1"])

    print("\n--- Judge headline (GPT-4o, axis F) ---")
    fax = judge["gpt-4o"]["axes"]["F"]
    print(f"Base {fax['base_mean']:.3f}  Fine {fax['fine_mean']:.3f}  Δ={fax['delta_mean']:+.3f}  CI95 [{fax['delta_ci95'][0]:+.3f},{fax['delta_ci95'][1]:+.3f}]  Wilcoxon p={fax['wilcoxon']['p']:.3e}  win {fax['wins']}/{judge['gpt-4o']['n']} tie {fax['ties']} loss {fax['losses']}")


if __name__ == "__main__":
    main()
