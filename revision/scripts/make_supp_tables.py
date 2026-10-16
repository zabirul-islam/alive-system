"""Generate supplementary LaTeX tables directly from revision/results/*.json (no hand-typed numbers).
Output: revision/generated/*.tex, \\input{} by supplementary.tex.
"""
import json
from pathlib import Path

R = Path(__file__).resolve().parents[1]
G = R / "generated"; G.mkdir(exist_ok=True)
J = lambda f: json.load(open(R / "results" / f))
f3 = lambda x: f"{x:.3f}"
ci = lambda c: f"[{c[0]:+.3f}, {c[1]:+.3f}]"


def pfmt(p):
    if p < 1e-12:
        return "$<10^{-12}$"
    if p >= 0.001:
        return f"{p:.3f}"
    e = f"{p:.1e}".split("e"); return f"${e[0]}{{\\times}}10^{{{int(e[1])}}}$"


def write(name, body):
    (G / name).write_text(body); print("wrote", name)


# 2x2 full table
d = J("score_2x2.json")
rows = []
for m in ["bertscore_f1", "rouge1", "rouge2", "rougeL", "nli_entail", "nli_contra", "numeric_err", "words"]:
    lab = {"bertscore_f1": "BERTScore F1", "rouge1": "ROUGE-1", "rouge2": "ROUGE-2", "rougeL": "ROUGE-L", "nli_entail": "NLI entailment",
           "nli_contra": "NLI contradiction", "numeric_err": "Numeric-unit flag rate", "words": "Answer length (words)"}[m]
    v = [d["cells"][c][m] for c in ["base|lecture", "alive|lecture", "base|retrieved", "alive|retrieved"]]
    fmt = (lambda x: f"{x:.1f}") if m == "words" else f3
    if m == "nli_entail": lab = "NLI entailment (first 1,024 chars)"
    if m == "nli_contra": lab = "NLI contradiction (first 1,024 chars)"
    rows.append(f"{lab} & " + " & ".join(fmt(x) for x in v) + r" \\")
ns = J("nli_seg_2x2.json")["cells"]
for m, lab in [("entail", "NLI entailment (all windows)"), ("neutral", "NLI neutral (all windows)"), ("contra", "NLI contradiction (all windows)")]:
    rows.append(f"{lab} & " + " & ".join(f3(ns[c][m]) for c in ["base_lecture", "alive_lecture", "base_retrieved", "alive_retrieved"]) + r" \\")
con = []
for k, lab in [("model_effect|lecture", "ALIVE$-$Base, lecture-restricted"), ("model_effect|retrieved", "ALIVE$-$Base, retrieved"),
               ("context_effect|base", "Retrieved$-$lecture-restricted, Base"), ("context_effect|alive", "Retrieved$-$lecture-restricted, ALIVE")]:
    for m, ml in [("bertscore_f1", "BS-F1"), ("rougeL", "ROUGE-L"), ("nli_entail", "NLI")]:
        x = d["contrasts"][k][m]; con.append(f"{lab} & {ml} & {x['delta']:+.4f} & {ci(x['ci95'])} & {pfmt(x['p_wilcoxon'])} \\\\")
for m, ml in [("bertscore_f1", "BS-F1"), ("rougeL", "ROUGE-L"), ("nli_entail", "NLI")]:
    x = d["interaction"][m]; con.append(f"Interaction (model effect, retrieved $-$ lecture-restricted) & {ml} & {x['delta']:+.4f} & {ci(x['ci95'])} & --- \\\\")
diff = d["by_difficulty_bertscore"]
drows = [f"{k.capitalize()} & " + " & ".join(f3(diff[k][c]) for c in ["base|lecture", "alive|lecture", "base|retrieved", "alive|retrieved"]) + r" \\"
         for k in ["basic", "intermediate", "advanced"]]
write("tab_2x2.tex", r"""\begin{table}[!htbp]
\centering
\caption{Answer quality in the 2$\times$2 design}
\label{tab:sup_2x2}
\small
\begin{tabular}{lcccc}
\toprule
& \multicolumn{2}{c}{\textbf{Lecture-restricted}} & \multicolumn{2}{c}{\textbf{Retrieved (top-4)}} \\
\textbf{Metric} & \textbf{Base} & \textbf{ALIVE} & \textbf{Base} & \textbf{ALIVE} \\
\midrule
""" + "\n".join(rows) + r"""
\midrule
""" + "\n".join(drows) + r"""
\bottomrule
\end{tabular}
\par\medskip
\begin{tabular}{llccc}
\toprule
\textbf{Contrast} & \textbf{Metric} & $\boldsymbol{\Delta}$ & \textbf{95\% CI} & $\boldsymbol{p}$ \\
\midrule
""" + "\n".join(con) + r"""
\bottomrule
\end{tabular}
\tabnotes{508 test questions, one generation run. First block: mean per condition; second block: BERTScore F1 by
question difficulty. Lower table: paired contrasts with bootstrap 95\% CIs and Wilcoxon $p$-values; the CIs condition on this
single run. Retrieved = semantic top-4 over all lectures. The regenerated outputs match the originally submitted predictions
exactly for """ + ", ".join(f"{k} {100*v:.1f}\\%" for k, v in d["reproduction_exact_match"].items()).replace("|", "/") + r""" of the answers.}
\end{table}
""")

# leakage + splits
lk = J("leakage_rescore.json")
lrows = [f"Original split$^*$, {lab} & {lk[k]['bertscore_f1']['n']} & {f3(lk[k]['bertscore_f1']['base'])} & {f3(lk[k]['bertscore_f1']['alive'])} & "
         f"{lk[k]['bertscore_f1']['delta']:+.3f} {ci(lk[k]['bertscore_f1']['ci95'])} & {lk[k]['rougeL']['delta']:+.3f} {ci(lk[k]['rougeL']['ci95'])} & --- \\\\"
         for k, lab in [("all", "all"), ("max_cos_lt_0.90", "cos$<$0.90"), ("max_cos_lt_0.80", "cos$<$0.80"), ("near_dup_ge_0.90", "cos$\\geq$0.90")]]
st = J("splits/stats.json")


def nli_delta(n, fn, sub):  # segment-aware NLI entailment difference (all test items only)
    if sub != "all":
        return "---"
    L = J(f"splits/{n}/nli_seg.json")["labels"]; key = "newfinal_lecture" if "final" in fn else "new_lecture"
    return f"{sum(x == 'entailment' for x in L[key]) / len(L[key]) - sum(x == 'entailment' for x in L['base_lecture']) / len(L['base_lecture']):+.3f}"
for n, lab, fn in [("slide_grouped", "Slide-grouped (best val.)", "score.json"), ("lecture_heldout", "Lecture-held-out (best val.)", "score.json"),
                   ("slide_grouped", "Slide-grouped (final ckpt.)", "score_final.json"), ("lecture_heldout", "Lecture-held-out (final ckpt.)", "score_final.json")]:
    p = R / f"results/splits/{n}/{fn}"
    if p.exists():
        sc = json.load(open(p))
        for sub, sl in [("all", "all"), ("max_cos_lt_0.90", "cos$<$0.90")]:
            b, r = sc[sub]["bertscore_f1"], sc[sub]["rougeL"]
            lrows.append(f"{lab}, {sl} & {sc[sub]['n']} & {f3(b['base'])} & {f3(b['adapted'])} & {b['delta']:+.3f} {ci(b['ci95'])} & "
                         f"{r['delta']:+.3f} {ci(r['ci95'])} & {nli_delta(n, fn, sub)} \\\\")
write("tab_splits.tex", r"""\begin{table}[!htbp]
\centering
\caption{Generalization beyond the pair-level split}
\label{tab:sup_splits}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lcccccc}
\toprule
\textbf{Condition} & $\boldsymbol{n}$ & \textbf{Base BS-F1} & \textbf{Adapted BS-F1} & \textbf{$\Delta$ BS-F1 [95\% CI]} & \textbf{$\Delta$ ROUGE-L [95\% CI]} & \textbf{$\Delta$ NLI$^\dagger$} \\
\midrule
""" + "\n".join(lrows) + r"""
\bottomrule
\end{tabular}
\tabnotes{Context is lecture-restricted throughout. cos is the cosine similarity (all-mpnet-base-v2) between a test question and
its nearest training question. Slide-grouped split: """ + f"{st['slide_grouped']['train']}/{st['slide_grouped']['val']}/{st['slide_grouped']['test']}" + r""" training/validation/test
pairs, with no inferred slide shared between the sets (text from the same generating passage may be shared). Lecture-held-out
split: lectures """ + ", ".join(map(str, st["lecture_heldout"]["test_lectures"])) + r""" held out (""" + f"{st['lecture_heldout']['test']}" + r""" test pairs). Retrained adapters use the
deployed recipe unchanged. $^*$Original-split rows use the originally submitted predictions (Base 0.737), not the regenerated run
of Table~\ref{tab:sup_2x2}. $^\dagger$Segment-aware NLI entailment (any 1,024-character context window), all test items.
Cluster bootstrap of the BS-F1 $\Delta$: slide-grouped, resampling """ + str(J("splits/slide_grouped/cluster.json")["n_clusters"]) + r""" slides, """ + ci(J("splits/slide_grouped/cluster.json")["cluster_ci95"]) + r"""; lecture-held-out,
per lecture: """ + "; ".join(f"{k} {v['delta']:+.3f}" for k, v in sorted(J("splits/lecture_heldout/cluster.json")["per_lecture"].items(), key=lambda kv: int(kv[0].split()[-1]))) + r""".}
\end{table}
""")

# judges
jd = J("judge_2x2.json") if (R / "results/judge_2x2.json").exists() else J("judge_gpt4o_only.json")
jrows = []
for j in [k for k in jd if k not in ("inter_judge", "error_mcnemar")]:
    jrows.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{Judge: {j}}}}} \\\\")
    for ctx, cl in [("lecture", "Lecture-restricted"), ("retrieved", "Retrieved")]:
        for ax in "FPGA":
            c = jd[j]["contrasts"][f"{ctx}:{ax}"]
            jrows.append(f"{cl} & {ax} & {jd[j]['cells'][f'base|{ctx}'][ax]:.2f} & {jd[j]['cells'][f'alive|{ctx}'][ax]:.2f} & {c['delta']:+.2f} {ci(c['ci95'])} & "
                         f"{100*c['win']:.0f}/{100*c['tie']:.0f}/{100*c['loss']:.0f} & {pfmt(c['p_holm'])} \\\\")
        eb, ea = jd[j]["errors"][f"base|{ctx}"], jd[j]["errors"][f"alive|{ctx}"]
        jrows.append(f"{cl} & Any error & {100*eb['any_error_rate']:.1f}\\% & {100*ea['any_error_rate']:.1f}\\% & & & \\\\")
etypes = []
for j in [k for k in jd if k not in ("inter_judge", "error_mcnemar")]:
    for c in ["base|lecture", "alive|lecture", "base|retrieved", "alive|retrieved"]:
        e = jd[j]["errors"][c]
        etypes.append(f"{j} & {c.replace('|', ' / ')} & " + " & ".join(str(e.get(k, 0)) for k in ["none", "conceptual", "mathematical", "causal", "numerical"]) +
                      f" & {sum(v for k, v in e.items() if k not in ('none', 'conceptual', 'mathematical', 'causal', 'numerical', 'any_error_rate'))} \\\\")
agree = ""
if "inter_judge" in jd:
    ij = jd["inter_judge"]
    agree = "Inter-judge agreement (same answer): " + "; ".join(
        f"{ax}: exact {100*ij[ax]['exact']:.0f}\\%, quadratic-weighted $\\kappa$ = {ij[ax]['kappa_qw']:.2f}" for ax in "FPGA") + \
        f"; any-error label: exact {100*ij['any_error']['exact']:.0f}\\%, $\\kappa$ = {ij['any_error']['kappa']:.2f}."
write("tab_judge.tex", r"""\begin{table}[!htbp]
\centering
\caption{External-judge scores for all four conditions}
\label{tab:sup_judge}
\small
\begin{tabular}{llccccc}
\toprule
\textbf{Context} & \textbf{Axis} & \textbf{Base} & \textbf{ALIVE} & \textbf{$\Delta$ [95\% CI]} & \textbf{W/T/L} & $\boldsymbol{p_{\textrm{Holm}}}$ \\
\midrule
""" + "\n".join(jrows) + r"""
\bottomrule
\end{tabular}
\par\medskip
\begin{tabular}{llcccccc}
\toprule
\textbf{Judge} & \textbf{Condition} & \textbf{None} & \textbf{Conceptual} & \textbf{Math.} & \textbf{Causal} & \textbf{Numerical} & \textbf{Other} \\
\midrule
""" + "\n".join(etypes) + r"""
\bottomrule
\end{tabular}
\tabnotes{Each answer was judged independently against the same lecture excerpts. Axes (1--5): F factuality, P pedagogical
clarity, G conciseness and length fit, A agreement with the reference. $\Delta$ = ALIVE $-$ Base with bootstrap 95\% CI;
W/T/L = share of questions where ALIVE scored higher, equal, or lower (\%); $p_{\textrm{Holm}}$ = Wilcoxon, Holm-adjusted within judge.
Lower table: most serious error type per answer (counts). """ + agree + r"""}
\end{table}
""")

# retrieval
for suf, cap, idx in [("", "Retrieval accuracy on the lecture index", "deployed lecture index (1,117 segments)"),
                      ("_textbook", "Retrieval accuracy on the textbook-augmented index", "textbook-augmented index (1,229 entries)")]:
    rv = J(f"retrieval_v2{suf}.json")
    rr = []
    names = {"semantic": "Semantic (all lectures)", "semantic_local": "Semantic (current lecture)", "timestamp_only": "Timestamp only", "hybrid_deployed_gridtimes": "As deployed (30-s grid, $\\lambda$=0.4)",
             "hybrid_deployed_realtimes": "Deployed formula, aligned times", "hybrid_local_cv": "Corrected (Eq.~1, CV $\\lambda$)"}
    for a, nm in names.items():
        for sc, sl in [("during", "Within"), ("lag1", "+1 min"), ("lag3", "+3 min"), ("lag5", "+5 min"), ("random", "Random")]:
            v = rv["arms"][a][sc]
            rr.append(f"{nm if sc == 'during' else ''} & {sl} & {v['ndcg1']['mean']:.2f} [{v['ndcg1']['ci'][0]:.2f}, {v['ndcg1']['ci'][1]:.2f}] & "
                      f"{v['ndcg3']['mean']:.2f} & {v['ndcg5']['mean']:.2f} & {v['hit1']['mean']:.2f} & {v['hit5']['mean']:.2f} \\\\")
        rr.append(r"\addlinespace")
    tests = "; ".join(f"{k.replace('_', ' ').replace(':', ': ')} $\\Delta$={v['delta']:+.3f}, $p_{{\\textrm{{Holm}}}}$={pfmt(v['p_holm'])}"
                      for k, v in rv["tests_ndcg1"].items())
    orig = rv["original_protocol"]
    write(f"tab_retrieval{suf}.tex", r"""\begin{table}[!htbp]
\centering
\caption{""" + cap + r"""}
\label{tab:sup_retrieval""" + suf.replace("_", "") + r"""}
\small
\begin{tabular}{llccccc}
\toprule
\textbf{Method} & \textbf{Pause} & \textbf{NDCG@1 [95\% CI]} & \textbf{NDCG@3} & \textbf{NDCG@5} & \textbf{Hit@1} & \textbf{Hit@5} \\
\midrule
""" + "\n".join(rr[:-1]) + r"""
\bottomrule
\end{tabular}
\tabnotes{""" + idx[0].upper() + idx[1:] + r"""; 50 questions; target-conditioned pause scenarios (Sec.~\ref{sup:retrieval_eval}).
Each value is the mean over 5 pause draws per question, with a bootstrap 95\% CI over questions. Random = pause uniform over the
whole current lecture (stress test added after review). Cross-validated $\lambda$ per fold: """ + ", ".join(map(str, rv["cv_chosen_lambdas"])) + r""".
The original (circular) protocol re-run on this single index gives NDCG@1 """ + f"{orig['semantic']['ndcg1']:.2f}" + r""" (semantic) and """ + f"{orig['timestamp']['ndcg1']:.2f}" + r""" (time-aware);
the submitted 0.300 and 0.640 came from two different indexes. Paired Wilcoxon tests on NDCG@1 (corrected vs.\ reference method),
Holm-adjusted over the """ + str(len(rv["tests_ndcg1"])) + r""" tests: """ + tests + r""".}
\end{table}
""")

# survey
sv = J("survey_r17.json")
srows = []
for d_, lab in [("Usability", "Usability"), ("Learning", "Learning impact"), ("Trust", "Trust and verification"), ("Over-reliance", "Over-reliance avoidance"),
                ("Skill dev.", "Skill development"), ("Equity", "Accessibility and equity")]:
    a, b = sv["all"]["domains"][d_], sv["excluding_nonusers"]["domains"][d_]
    scale = d_ not in ("Trust", "Skill dev.")
    srows.append(f"{lab}{'' if scale else '$^\\dagger$'} & {a['mean']:.2f}" + (f" [{a['ci95'][0]:.2f}, {a['ci95'][1]:.2f}] & {pfmt(a['p_vs_3'])}" if scale else " & ---") +
                 f" & {a['alpha']:.2f} [{a['alpha_ci95'][0]:.2f}, {a['alpha_ci95'][1]:.2f}] & {b['mean']:.2f} & " + (pfmt(b['p_vs_3']) if scale else "---") + " \\\\")
write("tab_survey.tex", r"""\begin{table}[!htbp]
\centering
\caption{Survey domain composites}
\label{tab:sup_survey}
\small
\begin{tabular}{lccccc}
\toprule
& \multicolumn{3}{c}{\textbf{All respondents}} & \multicolumn{2}{c}{\textbf{Excluding non-users}} \\
\textbf{Domain} & \textbf{Mean [95\% CI]} & $\boldsymbol{p}$ & $\boldsymbol{\alpha}$ \textbf{[95\% CI]} & \textbf{Mean} & $\boldsymbol{p}$ \\
\midrule
""" + "\n".join(srows) + r"""
\bottomrule
\end{tabular}
\tabnotes{Complete responses with corrected row alignment ($n{=}""" + str(sv["all"]["n_complete_cases"]) + r"""$); the right-hand columns exclude the
""" + str(sv["n_nonusers"]) + r""" respondents who reported never using ALIVE ($n{=}""" + str(sv["excluding_nonusers"]["n_complete_cases"]) + r"""$ complete). $p$ = one-sided Wilcoxon test against
the neutral value 3, not adjusted for multiplicity; $\alpha$ = Cronbach's $\alpha$ with Feldt 95\% CI. Full-battery $\alpha$ = """ + f"{sv['all']['full_alpha']:.2f}" + r"""
(all) and """ + f"{sv['excluding_nonusers']['full_alpha']:.2f}" + r""" (users only). Items significant after Holm correction: """ + f"{sv['all']['n_sig_holm']}" + r"""/23 (all) and
""" + f"{sv['excluding_nonusers']['n_sig_holm']}" + r"""/23 (users only). $^\dagger$Not a reliable scale; the mean is descriptive only and is not tested.}
\end{table}
""")


# ---- main-text tables (generated so main and supplement share values and rounding) ----
jrows = []
for j, name in [("gpt-4o", "GPT-4o"), ("claude-sonnet-4-6", "Claude Sonnet 4.6")]:
    jrows.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{Judge: {name}}}}} \\\\")
    for ax in "FPGA":
        c = jd[j]["contrasts"][f"lecture:{ax}"]
        jrows.append(f"{ax} & {jd[j]['cells']['base|lecture'][ax]:.2f} & {jd[j]['cells']['alive|lecture'][ax]:.2f} & ${c['delta']:+.2f}$ & "
                     f"{100*c['win']:.0f} / {100*c['tie']:.0f} / {100*c['loss']:.0f} \\\\")
    jrows.append(f"Err & {100*jd[j]['errors']['base|lecture']['any_error_rate']:.1f}\\% & {100*jd[j]['errors']['alive|lecture']['any_error_rate']:.1f}\\% & & \\\\")
    if j == "gpt-4o":
        jrows.append("\\midrule")
write("tab_judge_main.tex", "\n".join(jrows) + "\n")
rv = J("retrieval_v2.json"); rows = []
for a, nm in [("semantic", "Semantic (all lectures)"), ("semantic_local", "Semantic (current lecture)"), ("timestamp_only", "Timestamp only"),
              ("hybrid_deployed_gridtimes", "As deployed"), ("hybrid_local_cv", "Corrected (Eq.~\\ref{eq:retrieval})")]:
    rows.append(nm + " & " + " & ".join(f"{rv['arms'][a][s]['ndcg1']['mean']:.2f}" for s in ["during", "lag1", "lag3", "lag5", "random"]) + " \\\\")
write("tab_retrieval_main.tex", "\n".join(rows) + "\n")

# ---- additional conditions E1-E4 (R2 / AE) ----
je = J("judge_e123.json"); s2 = J("score_2x2.json")["cells"]
JN = [("gpt-4o", "GPT-4o"), ("claude-sonnet-4-6", "Claude")]
EX = [("Lecture-restricted", "base|lecture", "alive|lecture", None),
      ("E1: no context", "base_none|lecture", "alive_none|lecture", "e123/score_e1.json"),
      ("E2: length-matched Base", "base_brief|lecture", "alive|lecture", "e123/score_e2_brief_vs_alive.json"),
      ("E3: best-case end-to-end", "base_e2e|lecture", "alive_e2e|lecture", "e123/score_e3.json")]


def bs_words(f, b, a):
    if f is None:
        return s2[b]["bertscore_f1"], s2[a]["bertscore_f1"], s2[b]["words"], s2[a]["words"]
    x = J(f); return x["all"]["bertscore_f1"]["base"], x["all"]["bertscore_f1"]["adapted"], x["words"]["base"], x["words"]["new"]


mrows = []
for lab, b, a, f in EX:
    bb, ba, wb, wa = bs_words(f, b, a)
    mrows.append(f"{lab} & {wb:.0f} / {wa:.0f} & {bb:.3f} / {ba:.3f} & " +
                 " & ".join(f"{100*je[j]['cells'][b]['err']:.0f} / {100*je[j]['cells'][a]['err']:.0f}" for j, _ in JN) + " \\\\")
if (R / "results/e4/score_e4_lecture.json").exists() and "claude-sonnet-4-6|E4" in je:
    for ctx, lab in [("lecture", "E4: cross-seg., Eq.~(\\ref{eq:retrieval})"), ("retrieved", "E4: cross-seg., semantic")]:
        x = J(f"e4/score_e4_{ctx}.json")
        mrows.append(f"{lab} & {x['words']['base']:.0f} / {x['words']['new']:.0f} & {x['all']['bertscore_f1']['base']:.3f} / {x['all']['bertscore_f1']['adapted']:.3f} & " +
                     " & ".join((f"{100*je[j + '|E4']['cells'][f'base|{ctx}']['err']:.0f} / {100*je[j + '|E4']['cells'][f'alive|{ctx}']['err']:.0f}"
                                 if j + "|E4" in je else "---") for j, _ in JN) + " \\\\")
write("tab_extra_main.tex", "\n".join(mrows) + "\n")

srows = []
for name, lab, f in [("E1", "E1: ALIVE $-$ Base, no context", "e123/score_e1.json"), ("E2a", "E2: ALIVE $-$ length-instructed Base", "e123/score_e2_brief_vs_alive.json"),
                     ("E2b", "E2: length-instructed Base $-$ Base", "e123/score_e2_base_vs_brief.json"), ("E3", "E3: ALIVE $-$ Base, end-to-end", "e123/score_e3.json")]:
    x = J(f)["all"]
    srows.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{{lab}}}}} \\\\")
    for m, ml in [("bertscore_f1", "BS-F1"), ("rougeL", "ROUGE-L"), ("nli_entail", "NLI (first 1,024 chars)")]:
        srows.append(f"& {ml} & {x[m]['delta']:+.3f} {ci(x[m]['ci95'])} & {pfmt(x[m]['p_wilcoxon'])} \\\\")
    for j, jn in JN:
        for ax in "FPGA":
            c = je[j]["contrasts"][f"{name}:{ax}"]
            srows.append(f"& {jn} {ax} & {c['delta']:+.2f} {ci(c['ci95'])} & {pfmt(c['p_holm'])} \\\\")
        c = je[j]["contrasts"][f"{name}:err"]
        srows.append(f"& {jn} errors (A vs.\\ B) & {c['a_err']} vs.\\ {c['b_err']} of {c['n']} & {pfmt(c['mcnemar_p'])} \\\\")
n3 = J("e123/nli_seg_e3.json")["contrasts"]["alive_lecture - base_lecture"]


def seg_delta(f, a, b):  # paired difference in segment-aware entailment share, bootstrap CI
    import numpy as np
    L = J(f)["labels"]; dd = np.array([x == "entailment" for x in L[a]], float) - np.array([x == "entailment" for x in L[b]], float)
    rng = np.random.default_rng(0); bs = [dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(2000)]
    return {"delta": float(dd.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}


n2 = seg_delta("e123/nli_seg_e2.json", "alive_lecture", "base_brief"); n2c = J("e123/nli_seg_e2.json")["cells"]
write("tab_extra.tex", r"""\begin{table}[!htbp]
\centering
\caption{Additional answer conditions E1--E3}
\label{tab:sup_extra}
\small
\begin{tabular}{llcc}
\toprule
\textbf{Contrast} & \textbf{Measure} & \textbf{$\Delta$ [95\% CI]} & $\boldsymbol{p}$ \\
\midrule
""" + "\n".join(srows) + r"""
\bottomrule
\end{tabular}
\tabnotes{508 test questions, one generation run per condition. E1: no lecture context in the prompt. E2: Base instructed to
answer ``in about 25 words, in one or two clear, precise sentences'' (mean """ + f"{J('e123/score_e2_brief_vs_alive.json')['words']['base']:.1f}" + r""" words vs.\ """ +
      f"{J('e123/score_e2_brief_vs_alive.json')['words']['new']:.1f}" + r""" for ALIVE), lecture-restricted
context. E3: each question's slide is inferred, a pause is drawn inside that slide's aligned window, and the corrected retriever
(Eq.~1, $\lambda=0.05$) supplies the top-4 segments. $\Delta$ with bootstrap 95\% CI; $p$ = Wilcoxon for metrics, Holm-adjusted
Wilcoxon within judge for judge axes, exact McNemar for errors. Segment-aware NLI (any 1,024-character context window),
$\Delta$ entailment: E2 ALIVE $-$ length-instructed Base """ +
      f"${n2['delta']:+.3f}$ {ci(n2['ci95'])} ({n2c['alive_lecture']['entail']:.3f} vs.\\ {n2c['base_brief']['entail']:.3f})" + r"""; E3 """ + f"${n3['delta']:+.3f}$ {ci(n3['ci95'])}" + r""".}
\end{table}
""")

# latency under load (E5) and ASR portability (E6)
ll = J("e5/latency_load.json")
write("tab_latency_load.tex", r"""\begin{table}[!htbp]
\centering
\caption{Text-path latency under concurrent load}
\label{tab:sup_load}
\small
\begin{tabular}{cccccc}
\toprule
\textbf{Concurrent} & \textbf{p50 (s)} & \textbf{p95 (s)} & \textbf{Max (s)} & \textbf{Retrieval p50 (s)} & \textbf{Req/s} \\
\midrule
""" + "\n".join(f"{r['concurrency']} & {r['p50_s']:.2f} & {r['p95_s']:.2f} & {r['max_s']:.2f} & {r['retrieval_p50_s']:.2f} & {r['throughput_req_per_s']:.1f} \\\\" for r in ll) + r"""
\bottomrule
\end{tabular}
\tabnotes{Retrieval plus generation, merged ALIVE model served with vLLM on one H100; 64 requests per concurrency level.
Avatar rendering is not included.}
\end{table}
""")
e6 = J("e6_portability.json")
write("tab_e6.tex", r"""\begin{table}[!htbp]
\centering
\caption{Slide timings from raw ASR versus subtitles}
\label{tab:sup_e6}
\small
\begin{tabular}{lcccccc}
\toprule
\textbf{Lecture} & \textbf{ASR seg.} & \textbf{Slides} & \textbf{Overlap} & \textbf{Mean $|\Delta|$ (s)} & \textbf{Median $|\Delta|$ (s)} & \textbf{$\leq$10\,s} \\
\midrule
""" + "\n".join(f"{k.replace('_', ' ')} & {v['asr_segments']} & {v['slides']} & {v['mean_token_overlap']:.2f} & {v['mean_abs_start_diff_s']:.1f} & "
                f"{v['median_abs_start_diff_s']:.1f} & {100*v['share_within_10s']:.0f}\\% \\\\" for k, v in e6.items() if k.startswith("Lecture")) + r"""
\bottomrule
\end{tabular}
\tabnotes{Whisper large-v3 segments aligned to the slide scripts with the same aligner as the subtitles. Overlap = mean
fraction of segment tokens found in the assigned slide; $|\Delta|$ = absolute difference between the ASR-based and
subtitle-based slide start times; $\leq$10\,s = share of slides whose start times differ by at most 10\,s.}
\end{table}
""")

# E4 cross-segment set: BERTScore P/R/F (length context, post hoc) and judge contrasts
pr = J("e4/e4_pr.json"); e4rows = []
for ctx, cl in [("lecture", "Eq.~1 context"), ("retrieved", "Semantic top-4")]:
    o = pr[ctx]
    e4rows.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{cl}}}}} \\\\")
    e4rows.append("Words & " + " & ".join(f"{o[k]['words']:.0f}" for k in ["base", "base_brief", "alive"]) + " & & \\\\")
    for m, ml in [("P", "BERTScore P"), ("R", "BERTScore R"), ("F", "BERTScore F1")]:
        e4rows.append(f"{ml} & " + " & ".join(f"{o[k][m]:.3f}" for k in ["base", "base_brief", "alive"]) +
                      f" & ${o['alive-base:' + m]['delta']:+.3f}$ {ci(o['alive-base:' + m]['ci95'])} & ${o['alive-base_brief:' + m]['delta']:+.3f}$ {ci(o['alive-base_brief:' + m]['ci95'])} \\\\")
    for j, jn in JN:
        cells = je[j + "|E4"]["cells"]; con = je[j + "|E4"]["contrasts"]; s1, s2_ = ("E4_e2e", "E4L_e2e") if ctx == "lecture" else ("E4_sem", "E4L_sem")
        for ax in "FA":
            e4rows.append(f"{jn} {ax} & " + " & ".join(f"{cells[f'{k}|{ctx}'][ax]:.2f}" for k in ["base", "base_brief", "alive"]) +
                          f" & ${con[s1 + ':' + ax]['delta']:+.2f}$ ($p_H$={pfmt(con[s1 + ':' + ax]['p_holm'])}) & ${con[s2_ + ':' + ax]['delta']:+.2f}$ ($p_H$={pfmt(con[s2_ + ':' + ax]['p_holm'])}) \\\\")
        e4rows.append(f"{jn} errors (\\%) & " + " & ".join(f"{100*cells[f'{k}|{ctx}']['err']:.0f}" for k in ["base", "base_brief", "alive"]) +
                      f" & McNemar {pfmt(con[s1 + ':err']['mcnemar_p'])} & McNemar {pfmt(con[s2_ + ':err']['mcnemar_p'])} \\\\")
write("tab_e4.tex", r"""\begin{table}[!htbp]
\centering
\caption{Cross-segment questions (E4)}
\label{tab:sup_e4}
\small
\begin{tabular}{lccccc}
\toprule
\textbf{Measure} & \textbf{Base} & \textbf{Base-L} & \textbf{ALIVE} & \textbf{ALIVE $-$ Base} & \textbf{ALIVE $-$ Base-L} \\
\midrule
""" + "\n".join(e4rows) + r"""
\bottomrule
\end{tabular}
\tabnotes{60 questions; reference answers average """ + f"{pr['ref_words']:.0f}" + r""" words. Base-L = Base with the length instruction of E2. The judges scored
each answer against its two source slides (F factuality, A agreement with the reference). $\Delta$ with bootstrap 95\% CI;
$p_H$ = Wilcoxon, Holm-adjusted within judge over all E4 contrasts. The BERTScore precision/recall split is a post hoc
description of the length effect.}
\end{table}
""")

# second relevance annotation (H1): retrieval NDCG@1 under alternative relevance labels
irr = J("retrieval_irr.json"); SC = ["during", "lag1", "lag3", "lag5", "random"]
ARMS = [("semantic", "Semantic (all lectures)"), ("semantic_local", "Semantic (current lecture)"), ("timestamp_only", "Timestamp only"), ("hybrid_local_cv", "Corrected (Eq.~1)")]
LS = [("retrieval_v2.json", "First author (paper)"), ("retrieval_v2_intersection.json", "Intersection (both chose)"), ("retrieval_v2_union.json", "Union (either chose)"),
      ("retrieval_v2_annotator2.json", "Second annotator, pause on first author's slide"), ("retrieval_v2_annotator2_ownanchor.json", "Second annotator, pause on own first slide$^\\dagger$")]
irows = []
for f, lab in LS:
    rv = J(f); irows.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{lab}}} (CV $\\lambda$: {', '.join(map(str, rv['cv_chosen_lambdas']))})}} \\\\")
    for a, an in ARMS:
        irows.append(f"{an} & " + " & ".join(f"{rv['arms'][a][s]['ndcg1']['mean']:.2f}" for s in SC) + " \\\\")
write("tab_irr.tex", r"""\begin{table}[!htbp]
\centering
\caption{Retrieval NDCG@1 under alternative relevance labels}
\label{tab:sup_irr}
\small
\begin{tabular}{lccccc}
\toprule
\textbf{Method} & \textbf{Within} & \textbf{+1 min} & \textbf{+3 min} & \textbf{+5 min} & \textbf{Random} \\
\midrule
""" + "\n".join(irows) + r"""
\bottomrule
\end{tabular}
\tabnotes{50 questions, same protocol as Table~\ref{tab:sup_retrieval}. Agreement between the annotators: Cohen's
$\kappa$ = """ + f"{irr['kappa_lecture_universe']:.2f}" + r""" over """ + f"{irr['kappa_items']:,}" + r""" slide judgments (all slides of the lectures either annotator used for a question),
mean Dice """ + f"{irr['mean_dice']:.2f}" + r""", mean Jaccard """ + f"{irr['mean_jaccard']:.2f}" + r"""; they shared at least one slide for """ + f"{100*irr['share_any_overlap']:.0f}" + r"""\% of questions and chose
""" + f"{irr['relevant_per_q']['annotator1']:.2f}" + r""" vs.\ """ + f"{irr['relevant_per_q']['annotator2']:.2f}" + r""" slides per question. In the intersection, the """ + str(irr['empty_sets_filled_with_anchor']['intersection']) + r""" questions with no shared
slide keep the first author's first slide. $^\dagger$Post hoc check, added after the fixed-anchor result was seen.}
\end{table}
""")

# expert-reviewed subset (two external raters)
ex = J("expert_analysis.json"); ag = ex["agreement"]
AGL = [("answerable", "Question answerable from lecture"), ("ref_label_P_as_N", "Reference correct (P counted as N)"), ("ctx_ok", "Context supports reference"),
       ("base_corr", "Correctness of Base answer"), ("alive_corr", "Correctness of ALIVE answer"), ("pref", "Preference")]
arows = [f"{lab} & {100*ag[k]['raw']:.0f}\\% & {ag[k]['kappa']:.2f} \\\\" for k, lab in AGL]
crow = []
for tag, lab in [("all_items", "All items"), ("evidence_supported", "Evidence-supported$^\\dagger$"), ("reference_Y_or_P", "Reference accepted as Y or P")]:
    o = ex[tag]; pc = o["preference_counts_per_rater"]
    crow.append(f"{lab} & {o['n']} & {o['corr']['base']:.2f} / {o['corr']['alive']:.2f} & {pfmt(o['corr']['p_wilcoxon'])} & "
                f"{100*o['share_incorrect']['base']:.1f} / {100*o['share_incorrect']['alive']:.1f} & {o['clar']['base']:.2f} / {o['clar']['alive']:.2f} & {pfmt(o['clar']['p_wilcoxon'])} & "
                f"{pc[0]['alive']}/{pc[0]['base']}/{pc[0]['tie']}; {pc[1]['alive']}/{pc[1]['base']}/{pc[1]['tie']} \\\\")
rl = ex["reference_labels"]
write("tab_expert.tex", r"""\begin{table}[!htbp]
\centering
\caption{Expert review of 100 test items}
\label{tab:sup_expert}
\small
\begin{tabular}{lcc}
\toprule
\textbf{Judgment} & \textbf{Raw agreement} & \textbf{Cohen's $\kappa$} \\
\midrule
""" + "\n".join(arows) + r"""
\bottomrule
\end{tabular}
\par\medskip
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{lccccccc}
\toprule
\textbf{Items} & $\boldsymbol{n}$ & \textbf{Correctness} & $\boldsymbol{p}$ & \textbf{Incorrect (\%)} & \textbf{Clarity} & $\boldsymbol{p}$ & \textbf{Preference} \\
 & & \textbf{Base / ALIVE} & & \textbf{Base / ALIVE} & \textbf{Base / ALIVE} & & \textbf{r1; r2} \\
\midrule
""" + "\n".join(crow) + r"""
\bottomrule
\end{tabular}
\tabnotes{Two external raters, lecture-restricted condition. Upper table: agreement between the raters. Lower table: Base vs.\
ALIVE. Correctness is scored correct 1, partial 0.5, incorrect 0 and averaged over raters ($p$ = Wilcoxon); Incorrect = share of
judgments rating the answer incorrect; Preference = ALIVE/Base/tie counts for rater 1; rater 2. Reference labels: rater 1 Y/P/N
""" + f"{rl[0].get('Y',0)}/{rl[0].get('P',0)}/{rl[0].get('N',0)}" + r""", rater 2 Y/N """ + f"{rl[1].get('Y',0)}/{rl[1].get('N',0)}" + r"""; both raters accepted """ + f"{100*ex['share_ref_ok_all_raters']:.0f}" + r"""\% of the references, at least one
rejected """ + f"{100*ex['ref_rejected_by_any']:.0f}" + r"""\%, and both rejected """ + f"{100*ex['ref_rejected_by_both']:.0f}" + r"""\%. $^\dagger$Both raters marked the question answerable, the
reference correct (Y), and the context supportive.}
\end{table}
""")
