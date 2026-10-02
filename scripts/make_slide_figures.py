#!/usr/bin/env python3
"""Greyscale figures for the mid-submission slides (slides/figures/).

Reads outputs/analysis (run analyze_mid_submission.py and
analyze_rq2_decomposition.py first). M4-core is drawn in black; other models
in grey, so the charts work in a monochrome Beamer theme.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "outputs" / "analysis"
OUT = ROOT / "slides" / "figures"

INK, MID, LIGHT, GRID = "#111111", "#7a7a7a", "#bdbdbd", "#e2e2e2"
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Nimbus Sans", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 10,
        "pdf.fonttype": 42,
    }
)


def _style(ax, grid="x"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(LIGHT)
    ax.tick_params(colors=INK, length=0)
    ax.grid(axis=grid, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def _save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.03,
                metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def _direction_labels(ax, left, right, y):
    ax.annotate(f"← {left}", xy=(0.0, y), xycoords=("axes fraction", "axes fraction"),
                ha="left", va="top", color=MID, fontsize=9)
    ax.annotate(f"{right} →", xy=(1.0, y), xycoords=("axes fraction", "axes fraction"),
                ha="right", va="top", color=MID, fontsize=9)


def rq1_forest():
    cmp_ = pd.read_csv(ANALYSIS / "paired_comparisons.csv")
    rows = cmp_[(cmp_.tier == "primary") & (cmp_.rq == "RQ1")].reset_index(drop=True)
    seeds = pd.read_csv(ANALYSIS / "seed_level_metrics.csv").set_index(["model", "seed"])
    names = {"m1": "M1", "m2": "M2"}
    metric = {"det_macro_f1": "DET macro-F1", "type_macro_f1": "TYPE macro-F1"}
    fig, ax = plt.subplots(figsize=(6.2, 2.7))
    _style(ax)
    ax.grid(axis="y", visible=False)
    y = np.arange(len(rows))[::-1]
    for yi, (_, r) in zip(y, rows.iterrows()):
        d = (seeds.loc[r.model_a][r.metric] - seeds.loc[r.model_b][r.metric]).to_numpy()
        ax.scatter(d, np.full(len(d), yi), s=16, color=LIGHT, zorder=2, linewidths=0)
        ax.plot([r.ci_low, r.ci_high], [yi, yi], color=INK, linewidth=2.2, zorder=3)
        ax.scatter([r["diff"]], [yi], s=55, color=INK, zorder=4)
        p = "p < 0.001" if r.p_holm < 0.001 else f"p = {r.p_holm:.3f}"
        ax.text(0.012, yi, f"{r['diff']:+.3f}   ({p})", va="center", fontsize=9, color=INK)
    ax.axvline(0, color=MID, linewidth=1)
    ax.set_yticks(y, [f"M4-core − {names[r.model_b]}: {metric[r.metric]}" for _, r in rows.iterrows()])
    ax.set_xlim(-0.11, 0.075)
    ax.set_xlabel("Paired difference (mean over seeds; bar = 95% CI; grey = single seeds)", color=MID)
    _direction_labels(ax, "baseline better", "M4-core better", -0.22)
    _save(fig, "rq1_forest")


def scope_slope():
    s = pd.read_csv(ANALYSIS / "model_summary.csv", header=[0, 1], index_col=0)
    models = [("m1", "M1"), ("m2", "M2"), ("m3_symmetric", "M3"), ("m4_core", "M4-core")]
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(LIGHT)
    ax.tick_params(length=0)
    ax.set_yticks([])
    for key, name in models:
        a = s.loc[key, ("type_macro_f1", "mean")]
        b = s.loc[key, ("type_all_macro_f1", "mean")]
        hero = key == "m4_core"
        color, lw = (INK, 2.6) if hero else (MID, 1.4)
        ax.plot([0, 1], [a, b], color=color, linewidth=lw, marker="o", markersize=6 if hero else 4)
        weight = "bold" if hero else "normal"
        # M4-core and M3 are close at both ends: nudge their labels apart.
        nudge = {"m4_core": 0.009, "m3_symmetric": -0.009}.get(key, 0.0)
        ax.text(-0.04, a + nudge, f"{name}  {a:.3f}", ha="right", va="center", color=color, fontweight=weight)
        ax.text(1.04, b + nudge, f"{b:.3f}  {name}", ha="left", va="center", color=color, fontweight=weight)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0.23, 0.57)
    ax.set_xticks([0, 1], ["TYPE macro-F1\npolarized texts only\n(pre-declared)",
                           "TYPE macro-F1\nall texts\n(exploratory)"])
    _save(fig, "scope_slope")


def rq2_decomposition():
    dec = pd.read_csv(ANALYSIS / "rq2_decomposition.csv")
    dec = dec[(dec.model_a == "m4_core") & (dec.model_b == "m3_symmetric")].set_index("metric")
    cmp_ = pd.read_csv(ANALYSIS / "paired_comparisons.csv")
    macro = cmp_[(cmp_.tier == "primary") & (cmp_.rq == "RQ2") & (cmp_.metric == "type_macro_recall")].iloc[0]
    items = [
        ("Coverage\n(labels on detected texts)", dec.loc["coverage"]),
        ("Conditional recall\n(once detected)", dec.loc["conditional_recall"]),
        ("Micro recall\n(= coverage × cond.)", dec.loc["micro_recall"]),
        ("Macro recall\n(pre-declared)", macro),
    ]
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    _style(ax)
    ax.grid(axis="y", visible=False)
    y = np.arange(len(items))[::-1]
    for yi, (label, r) in zip(y, items):
        d, lo, hi = float(r["diff"]), float(r.ci_low), float(r.ci_high)
        sig = lo > 0 or hi < 0
        ax.barh(yi, d, height=0.55, color=INK if sig else LIGHT, edgecolor=INK, linewidth=0.8)
        ax.plot([lo, hi], [yi, yi], color=MID if sig else INK, linewidth=1.6)
        ax.plot([lo, lo], [yi - 0.12, yi + 0.12], color=MID if sig else INK, linewidth=1.2)
        ax.plot([hi, hi], [yi - 0.12, yi + 0.12], color=MID if sig else INK, linewidth=1.2)
        ax.text(max(hi, 0) + 0.004, yi, f"{d:+.3f}", va="center", fontsize=9, color=INK)
    ax.axvline(0, color=MID, linewidth=1)
    ax.set_yticks(y, [label for label, _ in items], fontsize=9)
    ax.set_xlim(-0.06, 0.11)
    ax.set_xlabel("M4-core − M3 (95% paired bootstrap CI)", color=MID)
    _direction_labels(ax, "M3 better", "M4-core better", -0.2)
    _save(fig, "rq2_decomposition")


def probability_pairs():
    prof = pd.read_csv(ANALYSIS / "rq2_probability_profile.csv")
    labels = [("political", "Political"), ("racial_ethnic", "Racial/\nethnic"),
              ("religious", "Religious"), ("gender_sexual_identity", "Gender/\nsexual"), ("other", "Other")]
    m2 = [prof[(prof.model == "m2") & (prof.label == k)].median_on_label_positive.iloc[0] for k, _ in labels]
    m4 = [prof[(prof.model == "m4_core") & (prof.label == k)].median_on_label_positive.iloc[0] for k, _ in labels]
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    _style(ax, grid="y")
    x = np.arange(len(labels))
    w = 0.36
    ax.bar(x - w / 2, m2, w, color=LIGHT, edgecolor=MID, linewidth=0.8, label="M2")
    ax.bar(x + w / 2, m4, w, color=INK, label="M4-core")
    for xi, a, b in zip(x, m2, m4):
        ax.text(xi - w / 2, a + 0.015, f"{a:.2f}", ha="center", fontsize=8, color=MID)
        ax.text(xi + w / 2, b + 0.015, f"{b:.2f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x, [name for _, name in labels], fontsize=9)
    ax.set_ylim(0, 0.9)
    ax.set_ylabel("Median probability on texts\nthat have the label", color=MID, fontsize=9)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    _save(fig, "probability_pairs")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rq1_forest()
    scope_slope()
    rq2_decomposition()
    probability_pairs()
    print("Wrote", sorted(p.name for p in OUT.glob("*.pdf")))


if __name__ == "__main__":
    main()
