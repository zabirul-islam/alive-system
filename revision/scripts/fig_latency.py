"""Supplementary figure: measured component latency of the deployed ALIVE pipeline (Table SIII).
Replaces the submitted latency figure, which plotted reference values from earlier work.
House style from figures4papers (PALETTE, publication rcParams, PDF+PNG 300 dpi).
Generation means/SDs come from the 20-run benchmark JSONs; the other components are the Table SIII values.
Output: revision/figures/fig_latency.{pdf,png}
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[1]
EV = R.parent / "backend/alive_scripts/evaluation"
PALETTE = {"blue_main": "#0F4D92", "red_strong": "#B64342", "neutral": "#CFCECE"}
plt.rcParams.update({"font.family": ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"], "font.size": 8,
                     "axes.linewidth": 1.0, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
base, alive = (json.load(open(EV / f)) for f in ("latency_base.json", "latency_finetuned.json"))
# (label, mean, sd, value label, colour); speech transcription is drawn separately as its observed range
bars = [("Retrieval", 0.10, None, "<0.10", PALETTE["neutral"]),
        ("Base LLM\ngeneration", base["llm_finetuned_mean"], base["llm_finetuned_std"], f"{base['llm_finetuned_mean']:.3f}", PALETTE["red_strong"]),
        ("ALIVE LLM\ngeneration", alive["llm_finetuned_mean"], alive["llm_finetuned_std"], f"{alive['llm_finetuned_mean']:.3f}", PALETTE["blue_main"]),
        ("Speech\nsynthesis", 0.20, None, r"$\approx$0.2", PALETTE["neutral"]),
        ("Avatar\nsynthesis", 1.85, 0.15, None, PALETTE["neutral"])]
fig, ax = plt.subplots(figsize=(4.4, 2.3))
ax.bar(0, 4.0 - 2.5, bottom=2.5, width=0.62, color=PALETTE["neutral"], hatch="///", edgecolor="0.45", lw=0.6)
ax.text(0, 4.08, "2.5–4.0\n(range)", ha="center", va="bottom", fontsize=6.8)
for i, (lab, m, sd, txt, col) in enumerate(bars, start=1):
    ax.bar(i, m, width=0.62, color=col, yerr=sd, capsize=2.5, error_kw={"lw": 0.9, "ecolor": "0.25"})
    ax.text(i, m + (sd or 0) + 0.08, txt or f"{m:.2f}", ha="center", va="bottom", fontsize=6.8)
ax.set_xticks(range(len(bars) + 1))
ax.set_xticklabels(["Speech\ntranscription"] + [b[0] for b in bars], fontsize=6.8)
ax.set_ylabel("Latency (s)"); ax.set_ylim(0, 4.7)
fig.tight_layout(pad=0.3)
out = R / "figures"; out.mkdir(exist_ok=True)
for ext in ["pdf", "png"]:
    fig.savefig(out / f"fig_latency.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
print("saved", out, f"base {base['llm_finetuned_mean']:.3f}±{base['llm_finetuned_std']:.3f}",
      f"alive {alive['llm_finetuned_mean']:.3f}±{alive['llm_finetuned_std']:.3f}")
