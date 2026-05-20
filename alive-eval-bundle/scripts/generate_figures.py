"""
Generate publication figures for ALIVE-ED paper.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

os.makedirs("figures", exist_ok=True)

BLUE  = "#2563EB"
GREEN = "#16A34A"
GRAY  = "#6B7280"

base = json.load(open("evaluation/results_base.json"))
ft   = json.load(open("evaluation/results_finetuned.json"))
ret  = json.load(open("evaluation/ndcg_results.json"))

def hit_at_k(per_q_scores):
    return np.mean([1.0 if s > 0 else 0.0 for s in per_q_scores])

text_ndcg1 = np.mean(ret["text_only"]["1"])
text_ndcg3 = np.mean(ret["text_only"]["3"])
text_ndcg5 = np.mean(ret["text_only"]["5"])
text_hit1  = hit_at_k(ret["text_only"]["1"])
text_hit5  = hit_at_k(ret["text_only"]["5"])

ts_ndcg1 = np.mean(ret["timestamp_aware"]["1"])
ts_ndcg3 = np.mean(ret["timestamp_aware"]["3"])
ts_ndcg5 = np.mean(ret["timestamp_aware"]["5"])
ts_hit1  = hit_at_k(ret["timestamp_aware"]["1"])
ts_hit5  = hit_at_k(ret["timestamp_aware"]["5"])

print("Retrieval metrics:")
print(f"  text_only:       NDCG@1={text_ndcg1:.4f} NDCG@3={text_ndcg3:.4f} NDCG@5={text_ndcg5:.4f} Hit@1={text_hit1:.4f} Hit@5={text_hit5:.4f}")
print(f"  timestamp_aware: NDCG@1={ts_ndcg1:.4f} NDCG@3={ts_ndcg3:.4f} NDCG@5={ts_ndcg5:.4f} Hit@1={ts_hit1:.4f} Hit@5={ts_hit5:.4f}")

# Figure 1 — LLM metric comparison
metrics = {
    "BERTScore F1": (base["bertscore"]["f1"],        ft["bertscore"]["f1"]),
    "ROUGE-1":      (base["rouge"]["rouge1"]["mean"], ft["rouge"]["rouge1"]["mean"]),
    "ROUGE-2":      (base["rouge"]["rouge2"]["mean"], ft["rouge"]["rouge2"]["mean"]),
    "ROUGE-L":      (base["rouge"]["rougeL"]["mean"], ft["rouge"]["rougeL"]["mean"]),
}
labels    = list(metrics.keys())
base_vals = [v[0] for v in metrics.values()]
ft_vals   = [v[1] for v in metrics.values()]
x = np.arange(len(labels)); width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - width/2, base_vals, width, label="Base Llama-3.1-8B",    color=GRAY, alpha=0.85)
bars_f = ax.bar(x + width/2, ft_vals, width, label="ALIVE-ED (Fine-tuned)", color=BLUE, alpha=0.85)
for bar, b, f in zip(bars_f, base_vals, ft_vals):
    ax.annotate(f"+{f-b:.3f}", xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                xytext=(0,4), textcoords="offset points", ha="center", va="bottom",
                fontsize=8, color=GREEN, fontweight="bold")
ax.set_ylabel("Score", fontsize=12)
ax.set_title("LLM Response Quality: Base vs. Fine-tuned", fontsize=13, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11); ax.set_ylim(0, 0.85)
ax.legend(fontsize=10); ax.yaxis.grid(True, linestyle="--", alpha=0.5); ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("figures/fig_llm_comparison.pdf", dpi=300)
fig.savefig("figures/fig_llm_comparison.png", dpi=300)
print("Saved: figures/fig_llm_comparison.pdf/.png"); plt.close()

# Figure 2 — Per-difficulty
difficulties = ["basic", "intermediate", "advanced"]
base_diff = [base["per_difficulty"][d]["mean_f1"] for d in difficulties]
ft_diff   = [ft["per_difficulty"][d]["mean_f1"]   for d in difficulties]
ns        = [base["per_difficulty"][d]["n"]        for d in difficulties]
x = np.arange(len(difficulties)); width = 0.35

fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(x - width/2, base_diff, width, label="Base Llama-3.1-8B",    color=GRAY, alpha=0.85)
ax.bar(x + width/2, ft_diff,   width, label="ALIVE-ED (Fine-tuned)", color=BLUE, alpha=0.85)
for xi, (b, f, n) in enumerate(zip(base_diff, ft_diff, ns)):
    ax.text(xi, max(b,f)+0.015, f"n={n}", ha="center", fontsize=9, color=GRAY)
ax.set_ylabel("BERTScore F1", fontsize=12)
ax.set_title("BERTScore F1 by Question Difficulty", fontsize=13, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels([d.capitalize() for d in difficulties], fontsize=11)
ax.set_ylim(0.60, 0.82); ax.legend(fontsize=10)
ax.yaxis.grid(True, linestyle="--", alpha=0.5); ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("figures/fig_difficulty_breakdown.pdf", dpi=300)
fig.savefig("figures/fig_difficulty_breakdown.png", dpi=300)
print("Saved: figures/fig_difficulty_breakdown.pdf/.png"); plt.close()

# Figure 3 — Retrieval
ret_labels  = ["NDCG@1", "NDCG@3", "NDCG@5", "Hit@1", "Hit@5"]
text_scores = [text_ndcg1, text_ndcg3, text_ndcg5, text_hit1, text_hit5]
ts_scores   = [ts_ndcg1,   ts_ndcg3,   ts_ndcg5,   ts_hit1,   ts_hit5]
x = np.arange(len(ret_labels)); width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - width/2, text_scores, width, label="Text-Only Retrieval",       color=GRAY, alpha=0.85)
bars_ts = ax.bar(x + width/2, ts_scores, width, label="Timestamp-Aware Retrieval", color=BLUE, alpha=0.85)
for bar, t, s in zip(bars_ts, text_scores, ts_scores):
    d = s - t
    if d > 0.001:
        ax.annotate(f"+{d:.2f}", xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                    xytext=(0,4), textcoords="offset points",
                    ha="center", fontsize=8, color=GREEN, fontweight="bold")
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Retrieval Performance: Text-Only vs. Timestamp-Aware", fontsize=13, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(ret_labels, fontsize=11); ax.set_ylim(0, 0.90)
ax.legend(fontsize=10); ax.yaxis.grid(True, linestyle="--", alpha=0.5); ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("figures/fig_retrieval.pdf", dpi=300)
fig.savefig("figures/fig_retrieval.png", dpi=300)
print("Saved: figures/fig_retrieval.pdf/.png"); plt.close()

# Figure 4 — Latency
components = ["ASR", "Retrieval", "LLM\n(base)", "LLM\n(fine-tuned)", "TTS", "Avatar"]
latencies  = [3.0, 0.1, 1.5, 1.526, 0.2, 5.0]
colors     = [GRAY, GRAY, GRAY, BLUE, GRAY, GRAY]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(components, latencies, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
for bar, val in zip(bars, latencies):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.08,
            f"{val:.2f}s", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_ylabel("Latency (seconds)", fontsize=12)
ax.set_title("End-to-End Latency Breakdown per Interaction", fontsize=13, fontweight="bold")
ax.set_ylim(0, 6.5); ax.yaxis.grid(True, linestyle="--", alpha=0.5); ax.set_axisbelow(True)
ax.legend(handles=[mpatches.Patch(color=BLUE, alpha=0.85, label="ALIVE-ED fine-tuned LLM"),
                   mpatches.Patch(color=GRAY, alpha=0.85, label="Other components")], fontsize=10)
fig.tight_layout()
fig.savefig("figures/fig_latency.pdf", dpi=300)
fig.savefig("figures/fig_latency.png", dpi=300)
print("Saved: figures/fig_latency.pdf/.png"); plt.close()

print("\nAll figures generated in ./figures/")
