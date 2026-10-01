#!/usr/bin/env python3
"""Generate the paper's LaTeX tables and figures from outputs/analysis.

Run ``scripts/analyze_mid_submission.py`` first. Every number in the report's
tables comes from the CSV files written by that script; nothing is typed by hand.
Writes the figure PDFs used by ``report/acl_latex.tex`` to ``report/figures/`` and
the table bodies (inlined in ``report/acl_latex.tex``) to
``outputs/analysis/latex_tables/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from polar_hierarchy.constants import TYPE_LABELS  # noqa: E402

ANALYSIS = PROJECT_ROOT / "outputs" / "analysis"
REPORT = PROJECT_ROOT / "report"
GENERATED = ANALYSIS / "latex_tables"
FIGURES = REPORT / "figures"

MODELS = ("m1", "m2", "m3_one_way", "m3_symmetric", "m4_core")
SHORT = {"m1": "M1", "m2": "M2", "m3_one_way": "M3-1w", "m3_symmetric": "M3-sym", "m4_core": "M4"}
TEX_NAME = {
    "m1": "M1 (independent)",
    "m2": "M2 (shared MTL)",
    "m3_one_way": "M3 one-way",
    "m3_symmetric": "M3 symmetric",
    "m4_core": "M4-core (noisy-OR)",
}
LABEL = {
    "political": "Political",
    "racial_ethnic": "Racial/ethnic",
    "religious": "Religious",
    "gender_sexual_identity": "Gender/sexual",
    "other": "Other",
}
METRIC = {
    "det_macro_f1": "DET macro-F1",
    "det_precision": "DET precision",
    "det_recall": "DET recall",
    "type_macro_f1": "TYPE macro-F1",
    "type_macro_precision": "TYPE macro-P",
    "type_macro_recall": "TYPE macro-R",
    "type_all_macro_f1": "TYPE macro-F1 (all texts)",
    "multilabel_recall": "Label recall (multi-label texts)",
}
COLORS = {
    "m1": "#2a78d6",
    "m2": "#eb6834",
    "m3_one_way": "#1baf7a",
    "m3_symmetric": "#eda100",
    "m4_core": "#e87ba4",
}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#dedcd6"
TEXT_WIDTH, COLUMN_WIDTH = 6.3, 3.03  # inches, ACL two-column layout

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7.5,
        "axes.titlesize": 8,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
    }
)


# --------------------------------------------------------------------------- helpers


def load():
    summary = pd.read_csv(ANALYSIS / "model_summary.csv", header=[0, 1], index_col=0)
    return {
        "summary": summary,
        "seeds": pd.read_csv(ANALYSIS / "seed_level_metrics.csv"),
        "cmp": pd.read_csv(ANALYSIS / "paired_comparisons.csv"),
        "label_cmp": pd.read_csv(ANALYSIS / "per_label_comparisons.csv"),
        "audit": pd.read_csv(ANALYSIS / "gating_audit_per_seed.csv"),
        "det": pd.read_csv(ANALYSIS / "det_error_profile.csv", index_col=0),
        "training": pd.read_csv(ANALYSIS / "training_statistics.csv", index_col=0),
        "thresholds": pd.read_csv(ANALYSIS / "selected_thresholds.csv", header=[0, 1], index_col=0),
    }


def ms(summary, model, metric, digits=3, pct=False):
    mu, sd = summary.loc[model, (metric, "mean")], summary.loc[model, (metric, "std")]
    if pct:
        return f"{100 * mu:.1f}"
    return f"{mu:.{digits}f}\\,{{\\scriptsize$\\pm${sd:.{digits}f}}}"


def p_fmt(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def signed(x):
    return f"{x:+.3f}".replace("-", "$-$")


def write(name, lines):
    (GENERATED / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def bold_best(values, higher=True):
    best = max(values) if higher else min(values)
    return [abs(v - best) < 1e-12 for v in values]


# --------------------------------------------------------------------------- tables


def table_main(d):
    s = d["summary"]
    cols = ["det_macro_f1", "type_macro_f1", "type_macro_precision", "type_macro_recall"]
    best = {c: bold_best([s.loc[m, (c, "mean")] for m in MODELS]) for c in cols}
    lines = [
        "\\begin{tabular}{@{}lrcccccc@{}}",
        "\\toprule",
        "& & \\multicolumn{1}{c}{\\textbf{DET}} & \\multicolumn{3}{c}{\\textbf{TYPE} (gold-polarized texts)}"
        " & \\multicolumn{2}{c}{\\textbf{LVR} (\\%)} \\\\",
        "\\cmidrule(lr){3-3}\\cmidrule(lr){4-6}\\cmidrule(l){7-8}",
        "Model & Params & macro-F1 & macro-F1 & macro-P & macro-R & A & total \\\\",
        "\\midrule",
    ]
    params = {"m1": "368M", "m2": "184M", "m3_one_way": "184M", "m3_symmetric": "184M", "m4_core": "184M"}
    for i, m in enumerate(MODELS):
        cells = []
        for c in cols:
            cell = ms(s, m, c)
            cells.append(f"\\textbf{{{cell}}}" if best[c][i] else cell)
        cells += [ms(s, m, "lvr_a", pct=True), ms(s, m, "lvr_total", pct=True)]
        lines.append(f"{TEX_NAME[m]} & {params[m]} & " + " & ".join(cells) + " \\\\")
        if m == "m2":
            lines.append("\\addlinespace[1pt]")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("table_main", lines)


def table_secondary(d):
    s = d["summary"]
    cols = ["det_precision", "det_recall", "type_all_macro_f1", "type_all_macro_precision",
            "type_all_macro_recall", "multilabel_recall", "singlelabel_recall"]
    lines = [
        "\\begin{tabular}{@{}lccccccc@{}}",
        "\\toprule",
        "& \\multicolumn{2}{c}{\\textbf{DET} (positive class)} & \\multicolumn{3}{c}{\\textbf{TYPE} on all 3{,}222 texts}"
        " & \\multicolumn{2}{c}{\\textbf{Label recall}} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-6}\\cmidrule(l){7-8}",
        "Model & P & R & macro-F1 & macro-P & macro-R & multi-label & single-label \\\\",
        "\\midrule",
    ]
    best = {c: bold_best([s.loc[m, (c, "mean")] for m in MODELS]) for c in cols}
    for i, m in enumerate(MODELS):
        cells = []
        for c in cols:
            cell = ms(s, m, c)
            cells.append(f"\\textbf{{{cell}}}" if best[c][i] else cell)
        lines.append(f"{TEX_NAME[m]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("table_secondary", lines)


def table_comparisons(d, tier):
    c = d["cmp"][d["cmp"]["tier"] == tier]
    holm = tier == "primary"
    lines = [
        "\\begin{tabular}{@{}llcccccc@{}}",
        "\\toprule",
        "RQ & Comparison \\& metric & A & B & $\\Delta$ (A$-$B) & 95\\% CI & "
        + ("$p_{\\text{Holm}}$ & " if holm else "$p$ & ")
        + ("Seeds \\\\" if holm else "Seeds \\\\"),
        "\\midrule",
    ]
    previous = None
    for _, r in c.iterrows():
        if previous is not None and r["rq"] != previous:
            lines.append("\\midrule")
        previous = r["rq"]
        p = r["p_holm"] if holm else r["p_boot"]
        lines.append(
            f"{r['rq']} & {SHORT[r['model_a']]}$-${SHORT[r['model_b']]}: {METRIC.get(r['metric'], r['metric'])} & "
            f"{r['mean_a']:.3f} & {r['mean_b']:.3f} & {signed(r['diff'])} & "
            f"[{signed(r['ci_low'])}, {signed(r['ci_high'])}] & {p_fmt(p)} & {int(r['seeds_a_better'])}/5 \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(f"table_{tier}_comparisons", lines)


def table_per_label_rq2(d):
    lc = d["label_cmp"]
    sub = lc[(lc["model_a"] == "m4_core") & (lc["model_b"] == "m3_symmetric")]
    support = {"political": 1150, "racial_ethnic": 281, "religious": 112,
               "gender_sexual_identity": 72, "other": 126}
    lines = [
        "\\begin{tabular}{@{}lrcc@{}}",
        "\\toprule",
        "Label & $n$ & $\\Delta$ recall [95\\% CI] & $\\Delta$ F1 [95\\% CI] \\\\",
        "\\midrule",
    ]
    for label in TYPE_LABELS:
        cells = []
        for stat in ("recall", "f1"):
            r = sub[(sub["label"] == label) & (sub["stat"] == stat)].iloc[0]
            sig = r["ci_low"] > 0 or r["ci_high"] < 0
            text = f"{signed(r['diff'])} [{signed(r['ci_low'])}, {signed(r['ci_high'])}]"
            cells.append(f"\\textbf{{{text}}}" if sig else text)
        lines.append(f"{LABEL[label]} & {support[label]:,} & ".replace(",", "{,}") + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("table_per_label_rq2", lines)


def table_per_label_all(d):
    """Appendix: full per-label F1 / precision / recall diffs vs every model."""
    lc = d["label_cmp"]
    lines = [
        "\\begin{tabular}{@{}llccc@{}}",
        "\\toprule",
        "Comparison & Label & $\\Delta$ precision & $\\Delta$ recall & $\\Delta$ F1 \\\\",
        "\\midrule",
    ]
    for (a, b), group in lc.groupby(["model_a", "model_b"], sort=False):
        for label in TYPE_LABELS:
            cells = []
            for stat in ("precision", "recall", "f1"):
                r = group[(group["label"] == label) & (group["stat"] == stat)].iloc[0]
                sig = r["ci_low"] > 0 or r["ci_high"] < 0
                text = f"{signed(r['diff'])}"
                cells.append(f"\\textbf{{{text}}}" if sig else text)
            lines.append(f"{SHORT[a]}$-${SHORT[b]} & {LABEL[label]} & " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    write("table_per_label_all_diffs", lines)


def table_per_label_scores(d):
    s = d["summary"]
    lines = [
        "\\begin{tabular}{@{}llccccc@{}}",
        "\\toprule",
        " & Model & Political & Racial/ethnic & Religious & Gender/sexual & Other \\\\",
        " & & ($n$=1{,}150) & ($n$=281) & ($n$=112) & ($n$=72) & ($n$=126) \\\\",
        "\\midrule",
    ]
    for stat, name in (("f1", "F1"), ("precision", "P"), ("recall", "R")):
        for i, m in enumerate(MODELS):
            head = f"\\multirow{{5}}{{*}}{{{name}}}" if i == 0 else ""
            cells = [ms(s, m, f"{label}_{stat}") for label in TYPE_LABELS]
            lines.append(f"{head} & {SHORT[m]} & " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    write("table_per_label_scores", lines)


def table_gating(d):
    a = d["audit"].drop(columns="seed")
    mean, std = a.mean(), a.std()
    removed = mean["labels_removed"]
    rows = [
        ("TYPE labels predicted by M2", "m2_type_labels_predicted", None),
        ("Labels removed by gating", "labels_removed", None),
        ("\\quad wrong, on non-polarized texts", "removed_on_gold_negative", removed),
        ("\\quad wrong, on polarized texts", "removed_fp_on_gold_positive", removed),
        ("\\quad \\textbf{correct}, on polarized texts", "removed_tp_on_gold_positive", removed),
        ("Polarized texts gated (M2 DET misses)", "gold_positive_examples_gated", None),
        ("\\quad with $\\geq$1 correct M2 TYPE label", "gold_positive_examples_gated_with_correct_type",
         mean["gold_positive_examples_gated"]),
        ("DET decisions changed by OR step", "det_predictions_changed", None),
    ]
    lines = ["\\begin{tabular}{@{}lrr@{}}", "\\toprule", "Quantity (per seed) & Mean $\\pm$ sd & Share \\\\",
             "\\midrule"]
    for name, col, denom in rows:
        share = f"{100 * mean[col] / denom:.1f}\\%" if denom else ""
        lines.append(f"{name} & {mean[col]:,.0f} $\\pm$ {std[col]:,.0f} & {share} \\\\".replace(",", "{,}"))
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("table_gating", lines)


def table_det_errors(d):
    det = d["det"]
    lines = ["\\begin{tabular}{@{}lrrrr@{}}", "\\toprule",
             "Model & FP & FN & Pred.\\ pos. & Non-pol.\\ w/ TYPE \\\\", "\\midrule"]
    for m in MODELS:
        r = det.loc[m]
        lines.append(
            f"{TEX_NAME[m]} & {r['false_positives']:.0f} & {r['false_negatives']:.0f} & "
            f"{r['predicted_positive']:,.0f} & {r['gold_negative_with_any_type']:,.0f} \\\\".replace(",", "{,}")
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("table_det_errors", lines)


def table_thresholds(d):
    t = d["thresholds"]
    cols = ["det_threshold", *[f"thr_{x}" for x in TYPE_LABELS]]
    names = {"m1_det": "M1-DET", "m1_type": "M1-TYPE", "m2": "M2", "m4_core": "M4-core"}
    lines = ["\\begin{tabular}{@{}lcccccc@{}}", "\\toprule",
             "Run type & DET & Political & Racial & Religious & Gender & Other \\\\", "\\midrule"]
    for run, name in names.items():
        cells = []
        for c in cols:
            mean = t.loc[run, (c, "mean")]
            if pd.isna(mean):
                cells.append("--")
            else:
                cells.append(f"{mean:.2f} ({int(t.loc[run, (c, 'at_grid_edge')])})")
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("table_thresholds", lines)


def table_training(d):
    t = d["training"]
    names = {"m1_det": "M1-DET", "m1_type": "M1-TYPE", "m2": "M2", "m4_core": "M4-core"}
    lines = ["\\begin{tabular}{@{}lccccc@{}}", "\\toprule",
             "Run type & Best epoch & Range & At cap & Epochs run & Min/run \\\\", "\\midrule"]
    for run, name in names.items():
        r = t.loc[run]
        lines.append(
            f"{name} & {r['best_epoch_mean']:.1f} & {int(r['best_epoch_min'])}--{int(r['best_epoch_max'])} & "
            f"{int(r['best_epoch_at_cap'])} & {r['epochs_run_mean']:.1f} & {r['runtime_min_mean']:.1f} \\\\"
        )
    total = t["runtime_min_total"].sum() / 60
    lines += ["\\midrule", f"\\multicolumn{{6}}{{@{{}}l}}{{Total: 100 runs, {total:.1f} GPU-hours}} \\\\",
              "\\bottomrule", "\\end{tabular}"]
    write("table_training", lines)


# --------------------------------------------------------------------------- figures


def _style(ax, grid_axis="x"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0, pad=2)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _save(fig, name):
    # Fixed metadata keeps the PDFs byte-identical across reruns.
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02,
                metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def bars(ax, s, metric, order, xmax=1.0, digits=3, show_labels=True):
    _style(ax)
    y = np.arange(len(order))
    means = [s.loc[m, (metric, "mean")] for m in order]
    stds = [s.loc[m, (metric, "std")] for m in order]
    ax.barh(y, means, height=0.66, color=[COLORS[m] for m in order], edgecolor="white", linewidth=1)
    ax.errorbar(means, y, xerr=stds, fmt="none", ecolor=INK_2, elinewidth=0.7, capsize=1.5)
    for yi, (mu, sd) in enumerate(zip(means, stds)):
        ax.text(mu + sd + 0.015 * xmax, yi, f"{mu:.{digits}f}", va="center", fontsize=6.5, color=INK)
    ax.set_xlim(0, xmax)
    if show_labels:
        ax.set_yticks(y, [SHORT[m] for m in order], color=INK)
    else:  # shared y-axis: hide labels without overwriting the first panel's
        ax.tick_params(labelleft=False)


def fig_main(d):
    s = d["summary"]
    order = list(MODELS)[::-1]
    metrics = ["det_macro_f1", "type_macro_f1", "type_macro_precision", "type_macro_recall"]
    fig, axes = plt.subplots(1, 4, figsize=(TEXT_WIDTH, 1.55), sharey=True)
    for i, (ax, metric) in enumerate(zip(axes, metrics)):
        bars(ax, s, metric, order, show_labels=i == 0)
        ax.set_title(METRIC[metric], loc="left", color=INK, pad=3)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", ".25", ".50", ".75", "1"])
    fig.tight_layout(w_pad=0.6)
    _save(fig, "fig_main_metrics")


def fig_secondary(d):
    s = d["summary"]
    order = list(MODELS)[::-1]
    metrics = ["type_all_macro_f1", "det_precision", "det_recall", "multilabel_recall"]
    fig, axes = plt.subplots(1, 4, figsize=(TEXT_WIDTH, 1.55), sharey=True)
    for i, (ax, metric) in enumerate(zip(axes, metrics)):
        bars(ax, s, metric, order, show_labels=i == 0)
        ax.set_title(METRIC[metric], loc="left", color=INK, pad=3)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", ".25", ".50", ".75", "1"])
    fig.tight_layout(w_pad=0.6)
    _save(fig, "fig_secondary_metrics")


def fig_lvr(d):
    s = d["summary"]
    order = list(MODELS)[::-1]
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 1.35))
    _style(ax)
    y = np.arange(len(order))
    a = np.array([s.loc[m, ("lvr_a", "mean")] for m in order]) * 100
    b = np.array([s.loc[m, ("lvr_b", "mean")] for m in order]) * 100
    ax.barh(y, a, height=0.62, color="#4a3aa7", edgecolor="white", linewidth=1,
            label="LVR-A (DET=0, TYPE predicted)")
    ax.barh(y, b, left=a, height=0.62, color="#008300", edgecolor="white", linewidth=1,
            label="LVR-B (DET=1, no TYPE)")
    for yi, total in enumerate(a + b):
        ax.text(total + 1.2, yi, f"{total:.1f}%", va="center", fontsize=6.5, color=INK)
    ax.set_yticks(y, [SHORT[m] for m in order], color=INK)
    ax.set_xlim(0, 75)
    ax.set_xlabel("Texts violating the hierarchy (%)", color=INK_2)
    ax.legend(frameon=False, loc="lower right", handlelength=1, borderaxespad=0.1)
    _save(fig, "fig_lvr")


def fig_forest(d):
    cmp_ = d["cmp"][d["cmp"]["tier"] == "primary"].reset_index(drop=True)
    seeds = d["seeds"].set_index(["model", "seed"])
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.0))
    _style(ax)
    ax.grid(axis="y", visible=False)
    y = np.arange(len(cmp_))[::-1]
    short_metric = {"det_macro_f1": "DET F1", "type_macro_f1": "TYPE F1", "type_macro_recall": "TYPE R"}
    for yi, (_, r) in zip(y, cmp_.iterrows()):
        diffs = (seeds.loc[r["model_a"]][r["metric"]] - seeds.loc[r["model_b"]][r["metric"]]).to_numpy()
        ax.scatter(diffs, np.full(len(diffs), yi), s=7, color="#bdbbb3", zorder=2, linewidths=0)
        ax.plot([r["ci_low"], r["ci_high"]], [yi, yi], color=INK, linewidth=1.4, zorder=3)
        ax.scatter([r["diff"]], [yi], s=26, color=COLORS["m4_core"], edgecolor=INK, linewidth=0.6, zorder=4)
    ax.axvline(0, color=INK_2, linewidth=0.8)
    ax.set_yticks(
        y,
        [f"{r['rq']}  M4$-${SHORT[r['model_b']]}  {short_metric[r['metric']]}" for _, r in cmp_.iterrows()],
        color=INK,
    )
    ax.set_xlabel("Paired difference (> 0 favours M4-core)", color=INK_2)
    ax.set_xlim(-0.11, 0.07)
    _save(fig, "fig_primary_comparisons")


def fig_gating(d):
    mean = d["audit"].mean(numeric_only=True)
    items = [
        ("Wrong, non-polarized texts", mean["removed_on_gold_negative"]),
        ("Wrong, polarized texts", mean["removed_fp_on_gold_positive"]),
        ("Correct, polarized texts", mean["removed_tp_on_gold_positive"]),
    ]
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 1.0))
    _style(ax)
    y = np.arange(len(items))[::-1]
    values = [v for _, v in items]
    ax.barh(y, values, height=0.6, color=COLORS["m3_symmetric"], edgecolor="white", linewidth=1)
    for yi, v in zip(y, values):
        ax.text(v + 60, yi, f"{v:,.0f}", va="center", fontsize=6.5, color=INK)
    ax.set_yticks(y, [n for n, _ in items], color=INK)
    ax.set_xlim(0, 4500)
    ax.set_xlabel("TYPE labels removed by M3 gating (per seed)", color=INK_2)
    _save(fig, "fig_gating_audit")


def fig_per_label(d, stat):
    s = d["summary"]
    order = list(MODELS)[::-1]
    support = {"political": 1150, "racial_ethnic": 281, "religious": 112,
               "gender_sexual_identity": 72, "other": 126}
    fig, axes = plt.subplots(1, 5, figsize=(TEXT_WIDTH, 1.55), sharey=True)
    for i, (ax, label) in enumerate(zip(axes, TYPE_LABELS)):
        bars(ax, s, f"{label}_{stat}", order, digits=2, show_labels=i == 0)
        ax.set_title(f"{LABEL[label]} ($n$={support[label]})", loc="left", color=INK, pad=3)
        ax.set_xticks([0, 0.5, 1.0], ["0", ".5", "1"])
    fig.tight_layout(w_pad=0.4)
    _save(fig, f"fig_per_label_{stat}")


def main() -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    d = load()
    table_main(d)
    table_secondary(d)
    table_comparisons(d, "primary")
    table_comparisons(d, "secondary")
    table_per_label_rq2(d)
    table_per_label_all(d)
    table_per_label_scores(d)
    table_gating(d)
    table_det_errors(d)
    table_thresholds(d)
    table_training(d)
    fig_main(d)
    fig_secondary(d)
    fig_lvr(d)
    fig_forest(d)
    fig_gating(d)
    fig_per_label(d, "recall")
    fig_per_label(d, "f1")
    print("Wrote", sorted(p.name for p in GENERATED.iterdir()))
    print("Wrote", sorted(p.name for p in FIGURES.iterdir() if p.suffix == ".pdf"))


if __name__ == "__main__":
    main()
