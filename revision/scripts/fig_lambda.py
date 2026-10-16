"""Supplementary figure: NDCG@1 vs temporal weight lambda under each pause scenario (R1-5).
House style from figures4papers (PALETTE, publication rcParams, PDF+PNG 300 dpi).
Output: revision/figures/fig_lambda_sensitivity.{pdf,png}
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[1]
PALETTE = {"blue_main": "#0F4D92", "teal": "#42949E", "violet": "#9A4D8E", "red_strong": "#B64342", "neutral": "#CFCECE"}
plt.rcParams.update({"font.family": ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"], "font.size": 8,
                     "axes.linewidth": 1.0, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
curve = json.load(open(R / "results/retrieval_v2_lambda_curve.json"))
res = json.load(open(R / "results/retrieval_v2.json"))
lams = list(curve); x = range(len(lams))
scen = [("during", "Pause within target slide", PALETTE["blue_main"]), ("lag1", "+1 min lag", PALETTE["teal"]),
        ("lag3", "+3 min lag", PALETTE["violet"]), ("lag5", "+5 min lag", PALETTE["red_strong"]),
        ("random", "Random pause (stress test)", "0.55")]
fig, ax = plt.subplots(figsize=(3.5, 2.7))
for s, lab, col in scen:
    ax.plot(x, [curve[l][s] for l in lams], "-o", color=col, lw=1.4, ms=3.2, label=lab)
    ts = res["arms"]["timestamp_only"][s]["ndcg1"]["mean"]
    ax.plot([len(lams) - 0.6], [ts], marker="<", color=col, ms=4, clip_on=False)
ax.axhline(res["arms"]["semantic"]["during"]["ndcg1"]["mean"], color="0.45", ls="--", lw=0.9)
ax.text(len(lams) - 1.05, res["arms"]["semantic"]["during"]["ndcg1"]["mean"] + 0.02, "semantic (all lectures)", ha="right", fontsize=6.5, color="0.35")
ax.axhline(res["arms"]["semantic_local"]["during"]["ndcg1"]["mean"], color="0.45", ls="-.", lw=0.9)
ax.text(len(lams) - 1.05, res["arms"]["semantic_local"]["during"]["ndcg1"]["mean"] + 0.02, "semantic (current lecture)", ha="right", fontsize=6.5, color="0.35")
cv = lams.index(str(res["cv_chosen_lambdas"][0]))
ax.axvline(cv, color="0.6", ls=":", lw=0.9)
ax.text(cv + 0.08, 0.02, r"CV-selected $\lambda$", fontsize=6.5, color="0.35", va="bottom")
ax.set_xticks(list(x)); ax.set_xticklabels(lams); ax.set_xlim(-0.3, len(lams) - 0.4)
ax.set_ylim(0, 1.08); ax.set_xlabel(r"Temporal weight $\lambda$"); ax.set_ylabel("NDCG@1")
ax.plot([], [], marker="<", ls="none", color="0.35", ms=4, label="Timestamp-only (same color)")
ax.legend(fontsize=6.3, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=1.6, columnspacing=1.0)
fig.tight_layout(pad=0.3)
out = R / "figures"; out.mkdir(exist_ok=True)
for ext in ["pdf", "png"]:
    fig.savefig(out / f"fig_lambda_sensitivity.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
print("saved", out)
