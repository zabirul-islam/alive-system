"""
ALIVE survey re-analysis.

Computes:
  - Cronbach alpha with Feldt (1965) 95% CI per domain.
  - One-sample Wilcoxon vs the neutral midpoint (3.0) on the
    positive-direction scale, per item.
  - Holm-Bonferroni correction across the 23 items.
  - Outputs item_stats_v2.csv, composites_v2.csv, survey_redo.json.

The raw survey responses contain personally identifying information
and are held under the IRB-exempt data-use plan; they are not
included in the public release. The expected file location is
configured by ALIVE_SURVEY_XLSX (default: ../private/survey.xlsx).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
XLSX = Path(os.environ.get(
    "ALIVE_SURVEY_XLSX",
    str(ROOT.parent / "private" / "survey.xlsx"),
))
OUT = ROOT


# --- Item registry (mirrors analyze_survey.py) ---
ITEMS = [
    ("Q1",  "Q1: Ease of Use\n(Score)",                "Usability",     False),
    ("Q2",  "Q2: Response Speed\n(Score)",             "Usability",     False),
    ("Q3",  "Q3: Frequency of Use\n(Score)",           "Usability",     False),
    ("Q5",  "Q5: Improved Understanding\n(Score)",     "Learning",      False),
    ("Q6",  "Q6: Homework Help\n(Score)",              "Learning",      False),
    ("Q7",  "Q7: Exam Preparation\n(Score)",           "Learning",      False),
    ("Q8",  "Q8: Helped Retention\n(Score)",           "Learning",      False),
    ("Q9",  "Q9: Connected Concepts\n(Score)",         "Learning",      False),
    ("Q10", "Q10: Trust in Accuracy\n(Score)",         "Trust",         False),
    ("Q11", "Q11: Verified Answers\n(Score)",          "Trust",         False),
    ("Q12", "Q12★: Accepted w/o Question\n(Score)",    "Trust",         True),
    ("Q13", "Q13: Identified Errors\n(Score)",         "Trust",         False),
    ("Q14", "Q14★: Reduced Attendance\n(Score)",       "Over-reliance", True),
    ("Q15", "Q15: First Instinct = ALIVE\n(Score)",    "Over-reliance", True),
    ("Q17", "Q17: Could Work Independently\n(Score)",  "Over-reliance", False),
    ("Q18", "Q18★: Used as Shortcut\n(Score)",         "Over-reliance", True),
    ("Q19", "Q19: Strengthened Skills\n(Score)",       "Skill dev.",    False),
    ("Q20", "Q20: Academic Integrity\n(Score)",        "Skill dev.",    False),
    ("Q21", "Q21: AI Awareness\n(Score)",              "Skill dev.",    False),
    ("Q22", "Q22: Accessibility\n(Score)",             "Equity",        False),
    ("Q23", "Q23: Equity / Level Field\n(Score)",      "Equity",        False),
    ("Q24", "Q24: Want ALIVE in Future\n(Score)",      "Equity",        False),
    ("Q25", "Q25: Good for High School\n(Score)",      "Equity",        False),
]

DOMAINS = ["Usability", "Learning", "Trust", "Over-reliance", "Skill dev.", "Equity"]


def load_df() -> pd.DataFrame:
    df = pd.read_excel(XLSX, sheet_name="Survey Responses", header=2)
    legend_mask = df.iloc[:, 0].astype(str).str.contains("SCALE LEGEND", na=False)
    df = df[~legend_mask].reset_index(drop=True)
    return df


def cronbach_alpha(item_matrix: np.ndarray) -> float:
    """Cronbach's alpha. Rows = subjects, columns = items."""
    n_items = item_matrix.shape[1]
    if n_items < 2:
        return float("nan")
    item_vars = item_matrix.var(axis=0, ddof=1)
    total_var = item_matrix.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return float("nan")
    return float((n_items / (n_items - 1)) * (1 - item_vars.sum() / total_var))


def feldt_alpha_ci(alpha: float, n_subjects: int, n_items: int, conf: float = 0.95) -> tuple[float, float]:
    """Feldt (1965) confidence interval for Cronbach alpha.

    CI = 1 - (1-alpha) * F  where F has df1=N-1, df2=(N-1)(k-1).
    """
    if not np.isfinite(alpha):
        return (float("nan"), float("nan"))
    a_lo = (1 - conf) / 2.0
    a_hi = 1 - a_lo
    df1 = n_subjects - 1
    df2 = (n_subjects - 1) * (n_items - 1)
    f_lo = stats.f.ppf(a_lo, df1, df2)
    f_hi = stats.f.ppf(a_hi, df1, df2)
    lo = 1 - (1 - alpha) * f_hi
    hi = 1 - (1 - alpha) * f_lo
    return (float(lo), float(hi))


def holm_bonferroni(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, pvals[idx] * (m - rank)))
        adj[idx] = running
    return adj


def main() -> None:
    df = load_df()

    # Build positive-direction matrix: rows = subjects, cols = items
    item_records = []
    cols = []
    for sid, colname, group, reverse in ITEMS:
        s = pd.to_numeric(df[colname], errors="coerce").dropna()
        s_pos = (6.0 - s) if reverse else s
        item_records.append((sid, colname, group, reverse, s, s_pos))
        cols.append((sid, s_pos.reset_index(drop=True)))

    # Stack into matrix
    mat = pd.concat({sid: s for sid, s in cols}, axis=1).dropna(how="any")
    n_subjects = mat.shape[0]
    print(f"n_subjects (complete rows across all items) = {n_subjects}")

    # Per-item one-sample Wilcoxon vs midpoint 3.0
    item_rows = []
    pvals_for_holm = []
    for sid, colname, group, reverse, s_raw, s_pos in item_records:
        n = len(s_pos)
        m_raw = float(s_raw.mean())
        m_pos = float(s_pos.mean())
        sd_pos = float(s_pos.std(ddof=1))
        d_centered = s_pos - 3.0
        nz = d_centered[d_centered != 0]
        if len(nz) >= 1:
            res = stats.wilcoxon(d_centered, alternative="greater", zero_method="wilcox", method="auto")
            p = float(res.pvalue)
            W = float(res.statistic)
        else:
            p, W = 1.0, float("nan")
        # 95% bootstrap CI for the mean
        rng = np.random.default_rng(seed=42)
        bs = rng.choice(s_pos.values, size=(1000, n), replace=True).mean(axis=1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        item_rows.append({
            "id": sid, "group": group, "reverse": reverse, "n": int(n),
            "raw_mean": m_raw, "pos_mean": m_pos, "pos_sd": sd_pos,
            "pos_ci95_lo": float(lo), "pos_ci95_hi": float(hi),
            "W_one_sample": W, "p_raw_one_sided_gt_midpoint": p,
        })
        pvals_for_holm.append(p)

    # Holm-Bonferroni across the 23 items
    p_holm = holm_bonferroni(pvals_for_holm)
    for i, row in enumerate(item_rows):
        row["p_holm"] = float(p_holm[i])
        row["sig_holm_005"] = bool(p_holm[i] < 0.05)

    df_items = pd.DataFrame(item_rows)
    df_items.to_csv(OUT / "item_stats_v2.csv", index=False)
    print(f"WROTE: {OUT / 'item_stats_v2.csv'}  (n_items={len(df_items)})")

    # Per-domain Cronbach alpha + 95% CI
    comp_rows = []
    for d in DOMAINS:
        ids = [sid for sid, _, group, _ in ITEMS if group == d]
        sub_mat = mat[ids].to_numpy(dtype=float)
        n_items = sub_mat.shape[1]
        alpha = cronbach_alpha(sub_mat)
        lo, hi = feldt_alpha_ci(alpha, n_subjects, n_items)
        pos_mean = float(sub_mat.mean())
        pct_pos = float((sub_mat >= 4.0).mean() * 100.0)
        comp_rows.append({
            "domain": d, "n_items": n_items, "n_subjects": n_subjects,
            "pos_mean": pos_mean, "pct_pos_ge4": pct_pos,
            "cronbach_alpha": alpha,
            "alpha_ci95_lo": lo, "alpha_ci95_hi": hi,
        })

    # Full battery alpha
    full_alpha = cronbach_alpha(mat.to_numpy(dtype=float))
    full_lo, full_hi = feldt_alpha_ci(full_alpha, n_subjects, mat.shape[1])
    comp_rows.append({
        "domain": "FULL_BATTERY", "n_items": mat.shape[1], "n_subjects": n_subjects,
        "pos_mean": float(mat.to_numpy().mean()),
        "pct_pos_ge4": float((mat.to_numpy() >= 4.0).mean() * 100.0),
        "cronbach_alpha": full_alpha,
        "alpha_ci95_lo": full_lo, "alpha_ci95_hi": full_hi,
    })

    df_comp = pd.DataFrame(comp_rows)
    df_comp.to_csv(OUT / "composites_v2.csv", index=False)
    print(f"WROTE: {OUT / 'composites_v2.csv'}")

    summary = {
        "n_respondents": int(n_subjects),
        "domains": comp_rows,
        "items": item_rows,
        "holm_threshold": 0.05,
        "n_items_significant_after_holm": int(sum(1 for r in item_rows if r["sig_holm_005"])),
    }
    (OUT / "survey_redo.json").write_text(json.dumps(summary, indent=2))
    print(f"WROTE: {OUT / 'survey_redo.json'}")

    # Console summary
    print("\n--- Domain composites (Cronbach alpha + 95% CI) ---")
    for r in comp_rows:
        print(f"  {r['domain']:14s}  n_items={r['n_items']}  pos_mean={r['pos_mean']:.2f}  "
              f"%pos={r['pct_pos_ge4']:.1f}%  α={r['cronbach_alpha']:.2f} "
              f"[{r['alpha_ci95_lo']:.2f},{r['alpha_ci95_hi']:.2f}]")

    print(f"\nItems with Holm-adjusted p < 0.05 (one-sided > midpoint): "
          f"{summary['n_items_significant_after_holm']}/{len(item_rows)}")


if __name__ == "__main__":
    main()
