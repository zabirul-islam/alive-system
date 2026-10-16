"""R1-6/R1-7 survey re-analysis on top of analysis/survey_redo.py (same registry, same tests).

Adds: Holm reconciliation (current re-run vs submitted item_stats_v2.csv), composite means with
bootstrap CIs and one-sample Wilcoxon vs 3, n bookkeeping (per-item vs complete cases),
Q3 frequency distribution, and a sensitivity analysis excluding respondents who never used ALIVE.
Output: revision/results/survey_r17.json
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))
import survey_redo as S  # noqa: E402

df = S.load_df()
pos = {sid: pd.to_numeric(df[c], errors="coerce").pipe(lambda s: 6.0 - s if rev else s) for sid, c, _, rev in S.ITEMS}
P = pd.DataFrame(pos)
q3 = pd.to_numeric(df[[c for sid, c, _, _ in S.ITEMS if sid == "Q3"][0]], errors="coerce")
q4col = [c for c in df.columns if str(c).startswith("Q4")]
nonuser = (q3 == 1) | (df[q4col[0]].astype(str).str.contains("did not use", case=False) if q4col else False)


def analyse(M):
    items = {}
    pv = []
    for sid in M.columns:
        s = M[sid].dropna(); d = s - 3.0
        p = float(stats.wilcoxon(d, alternative="greater", zero_method="wilcox").pvalue) if (d != 0).any() else 1.0
        items[sid] = {"n": int(len(s)), "mean": round(float(s.mean()), 2)}; pv.append(p)
    for sid, ph, p in zip(M.columns, S.holm_bonferroni(pv), pv):
        items[sid].update(p_raw=p, p_holm=ph, sig=bool(ph < .05))
    cc = M.dropna()
    doms = {}
    for d in S.DOMAINS:
        ids = [sid for sid, _, g, _ in S.ITEMS if g == d]
        sub = cc[ids].to_numpy(float); comp = sub.mean(1)
        a = S.cronbach_alpha(sub); lo, hi = S.feldt_alpha_ci(a, len(cc), len(ids))
        rng = np.random.default_rng(42); bs = [comp[rng.integers(0, len(comp), len(comp))].mean() for _ in range(2000)]
        doms[d] = {"items": ids, "n_complete": int(len(cc)), "mean": round(float(comp.mean()), 2),
                   "ci95": [round(float(np.percentile(bs, 2.5)), 2), round(float(np.percentile(bs, 97.5)), 2)],
                   "p_vs_3": float(stats.wilcoxon(comp - 3, alternative="greater").pvalue),
                   "alpha": round(a, 2), "alpha_ci95": [round(lo, 2), round(hi, 2)]}
    full = S.cronbach_alpha(cc.to_numpy(float))
    return {"n_respondents": int(len(M)), "n_complete_cases": int(len(cc)), "full_mean": round(float(cc.to_numpy(float).mean()), 2), "items": items,
            "n_sig_holm": sum(v["sig"] for v in items.values()),
            "sig_items": [k for k, v in items.items() if v["sig"]], "domains": doms, "full_alpha": round(full, 2)}


out = {"all": analyse(P), "excluding_nonusers": analyse(P[~nonuser.values])}
sub = pd.read_csv(ROOT / "analysis/item_stats_v2.csv")
out["submitted_item_stats_sig"] = sub.loc[sub.sig_holm_005, "id"].tolist()
out["holm_reconciliation"] = sorted(set(out["all"]["sig_items"]) ^ set(out["submitted_item_stats_sig"]))
out["q3_frequency_counts"] = {int(k): int(v) for k, v in q3.value_counts().sort_index().items()}
out["n_nonusers"] = int(nonuser.sum())
out["q4_counts"] = df[q4col[0]].astype(str).value_counts().to_dict() if q4col else None
u = P[["Q1", "Q2"]].dropna().mean(1)  # post hoc: usability without the frequency item Q3
out["usability_q1q2_post_hoc"] = {"n": int(len(u)), "mean": round(float(u.mean()), 2),
                                  "p_vs_3": float(stats.wilcoxon(u - 3, alternative="greater").pvalue)}
json.dump(out, open(ROOT / "revision/results/survey_r17.json", "w"), indent=1, default=float)
for k in ["all", "excluding_nonusers"]:
    o = out[k]
    print(k, "n", o["n_respondents"], "cc", o["n_complete_cases"], "holm", o["n_sig_holm"], o["sig_items"], "fullα", o["full_alpha"])
    for d, v in o["domains"].items():
        print("   ", d, v["mean"], v["ci95"], f"p={v['p_vs_3']:.4f}", "α", v["alpha"], v["alpha_ci95"])
print("submitted sig:", out["submitted_item_stats_sig"], "diff:", out["holm_reconciliation"])
print("Q3 counts", out["q3_frequency_counts"], "nonusers", out["n_nonusers"], "Q4", out["q4_counts"])
