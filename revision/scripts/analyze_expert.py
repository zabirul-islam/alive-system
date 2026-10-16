"""R1-3: analyse the two completed expert workbooks against the private A/B key.

Reports: raw agreement + Cohen's kappa per field, share of valid references, Base vs ALIVE
correctness/clarity/preference on the full subset and on the evidence-supported sub-analysis
(both raters: answerable = Y, reference correct = Y, context supports = Y).
"Reference correct" may be Y, P (partly correct; used by rater 1 although the form asked Y/N) or N: ref_ok = Y,
ref_ok_yp = Y or P (sensitivity). With only one completed workbook, runs in single-rater mode (no agreement).
Usage: python analyze_expert.py  (reads revision/expert/ALIVE_expert_review_rater{1,2}.xlsx; blank ones are skipped)
Output: revision/results/expert_analysis.json
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, binomtest
from sklearn.metrics import cohen_kappa_score

R = Path(__file__).resolve().parents[1]
kp = R / "expert/KEY_do_not_share.json"  # private name during rating; released as unblinding_key.json
key = {int(k): v for k, v in json.load(open(kp if kp.exists() else R / "expert/unblinding_key.json")).items()}
D = [pd.read_excel(R / f"expert/ALIVE_expert_review_rater{r}.xlsx", sheet_name="Items").set_index("Item") for r in [1, 2]]
D = [d for d in D if d["Preference (A/B/Tie)"].notna().all()]  # completed workbooks only
norm = lambda s: s.astype(str).str.strip().str.upper().str[:1]
CORR = {"C": 1.0, "P": 0.5, "I": 0.0}


def unblind(df):
    out = pd.DataFrame(index=df.index)
    for it in df.index:
        for side in "AB":
            m = key[it][side]
            out.loc[it, f"{m}_corr"] = CORR.get(str(df.loc[it, f"{side} correctness (C/P/I)"]).strip().upper()[:1], np.nan)
            out.loc[it, f"{m}_clar"] = pd.to_numeric(df.loc[it, f"{side} clarity (1-5)"], errors="coerce")
            out.loc[it, f"{m}_err"] = str(df.loc[it, f"{side} error type"]).strip().lower()
        p = str(df.loc[it, "Preference (A/B/Tie)"]).strip().upper()
        out.loc[it, "pref"] = key[it][p] if p in "AB" and p else "tie"
    for f, col in [("answerable", "Q answerable from lecture? (Y/N)"), ("ref_ok", "Reference correct? (Y/N)"),
                   ("ctx_ok", "Context supports reference? (Y/N)")]:
        out[f] = norm(df[col]) == "Y"
    out["ref_ok_yp"] = norm(df["Reference correct? (Y/N)"]).isin(["Y", "P"])
    out["ref_label"] = norm(df["Reference correct? (Y/N)"])
    return out


U = [unblind(d) for d in D]
res = {"n_raters": len(U), "n_items": len(U[0]), "agreement": {}}
if len(U) == 2:
    for f in ["answerable", "ref_ok", "ref_ok_yp", "ctx_ok", "base_corr", "alive_corr", "pref"]:
        a, b = U[0][f].astype(str), U[1][f].astype(str)
        res["agreement"][f] = {"raw": round(float((a == b).mean()), 3), "kappa": round(float(cohen_kappa_score(a, b)), 3)}
res["reference_labels"] = [u["ref_label"].value_counts().to_dict() for u in U]
if len(U) == 2:
    pn = [u["ref_label"].replace("P", "N") for u in U]
    res["agreement"]["ref_label_P_as_N"] = {"raw": round(float((pn[0] == pn[1]).mean()), 3), "kappa": round(float(cohen_kappa_score(pn[0], pn[1])), 3)}
    res["ref_rejected_by_any"] = round(float(np.logical_or.reduce([u["ref_label"] == "N" for u in U]).mean()), 3)
    res["ref_rejected_by_both"] = round(float(np.logical_and.reduce([u["ref_label"] == "N" for u in U]).mean()), 3)
for f in ["answerable", "ref_ok", "ref_ok_yp", "ctx_ok"]:
    res[f"share_{f}_all_raters"] = round(float(np.logical_and.reduce([u[f] for u in U]).mean()), 3)
supported = pd.Series(np.logical_and.reduce([u[c] for u in U for c in ["answerable", "ref_ok", "ctx_ok"]]), index=U[0].index)


def compare(mask, tag):
    o = {"n": int(mask.sum())}
    for q in ["corr", "clar"]:
        b = sum(u[f"base_{q}"] for u in U)[mask] / len(U); a = sum(u[f"alive_{q}"] for u in U)[mask] / len(U)
        ok = b.notna() & a.notna()
        o[q] = {"base": round(float(b[ok].mean()), 3), "alive": round(float(a[ok].mean()), 3),
                "p_wilcoxon": float(wilcoxon(a[ok], b[ok]).pvalue) if (a[ok] != b[ok]).any() else 1.0}
    pr = pd.concat([u["pref"][mask] for u in U])
    o["preference"] = {k: round(float((pr == k).mean()), 3) for k in ["alive", "base", "tie"]}
    o["preference_counts_per_rater"] = [{k: int((u["pref"][mask] == k).sum()) for k in ["alive", "base", "tie"]} for u in U]
    na, nb = int((pr == "alive").sum()), int((pr == "base").sum())
    o["preference_binom_p"] = float(binomtest(na, na + nb).pvalue) if na + nb else 1.0
    o["error_types"] = {m: pd.concat([u[f"{m}_err"][mask] for u in U]).value_counts().to_dict() for m in ["base", "alive"]}
    o["share_correct"] = {m: round(float(pd.concat([u[f"{m}_corr"][mask] for u in U]).eq(1.0).mean()), 3) for m in ["base", "alive"]}
    o["share_incorrect"] = {m: round(float(pd.concat([u[f"{m}_corr"][mask] for u in U]).eq(0.0).mean()), 3) for m in ["base", "alive"]}
    res[tag] = o


compare(pd.Series(True, index=U[0].index), "all_items")
compare(supported, "evidence_supported")
compare(pd.Series(np.logical_and.reduce([u["ref_ok_yp"] for u in U]), index=U[0].index), "reference_Y_or_P")
out = R / ("results/expert_analysis.json" if len(U) == 2 else "results/expert_analysis_interim_rater1.json")
json.dump(res, open(out, "w"), indent=1, default=str)
print(json.dumps(res, indent=1, default=str))
