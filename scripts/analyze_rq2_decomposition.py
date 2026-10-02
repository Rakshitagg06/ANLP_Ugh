#!/usr/bin/env python3
"""Exploratory RQ2 analysis: why M4-core and M3 tie on macro TYPE metrics.

Decomposes label-level (micro) TYPE recall on gold-polarized texts into

    micro recall = coverage x conditional recall

where *coverage* is the share of gold TYPE labels whose text is detected
(predicted DET = 1) and *conditional recall* is the share of those labels the
model then predicts. Writes, to ``outputs/analysis``:

- ``rq2_decomposition.csv``: paired M4-core - M3-symmetric differences with
  95% paired-bootstrap CIs (same procedure as the primary analysis);
- ``rq2_per_label_decomposition.csv``: per-label coverage / conditional recall;
- ``rq2_seed_differences.csv``: per-seed primary and secondary differences;
- ``rq2_probability_profile.csv``: median TYPE probability per label on texts
  that have the label, polarized texts without it, and gold-neutral texts.

These comparisons were defined after the primary results were known; report
them as exploratory and uncorrected.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from analyze_mid_submission import SEEDS, load_predictions  # noqa: E402
from polar_hierarchy.constants import TYPE_LABELS  # noqa: E402

OUT = PROJECT_ROOT / "outputs" / "analysis"
MODELS = ("m2", "m3_symmetric", "m4_core")


def count_columns(entry: dict) -> np.ndarray:
    """Per-text counts whose weighted sums give the decomposition metrics."""
    gd = entry["gold_det"] == 1
    gt = entry["gold_types"] == 1
    pt = entry["pred_types"] == 1
    detected = gd & (entry["pred_det"] == 1)
    gold_labels = gt.sum(axis=1) * gd
    hits = (gt & pt).sum(axis=1) * gd
    single = gd & (gt.sum(axis=1) == 1)
    multi = gd & (gt.sum(axis=1) >= 2)
    return np.stack(
        [
            gold_labels,
            hits,
            gold_labels * detected,
            hits * detected,
            (~gd) & pt.any(axis=1),
            ~gd,
            (pt & (~gd)[:, None]).sum(axis=1),
            gold_labels * single,
            hits * single,
            gold_labels * multi,
            hits * multi,
        ],
        axis=1,
    ).astype(np.float64)


def metrics(c: np.ndarray) -> dict:
    return {
        "micro_recall": c[..., 1] / c[..., 0],
        "coverage": c[..., 2] / c[..., 0],
        "conditional_recall": c[..., 3] / c[..., 2],
        "neutral_fpr": c[..., 4] / c[..., 5],
        "neutral_type_labels": c[..., 6],
        "single_label_recall": c[..., 8] / c[..., 7],
        "multilabel_recall": c[..., 10] / c[..., 9],
    }


def probability_profile(runs: Path) -> pd.DataFrame:
    """Median TYPE probabilities pooled over all folds and seeds (M2 vs M4-core)."""
    rows = []
    for model in ("m2", "m4_core"):
        records = []
        for path in sorted(runs.glob(f"{model}-fold*-seed*/predictions/outer_fold_predictions.jsonl")):
            records.extend(pd.read_json(path, lines=True).to_dict("records"))
        probs = np.array([r["type_probabilities"] for r in records])
        gold = np.array([r["gold_types"] for r in records]) == 1
        polarized = np.array([r["gold_det"] for r in records]) == 1
        for k, label in enumerate(TYPE_LABELS):
            rows.append(
                {
                    "model": model,
                    "label": label,
                    "median_on_label_positive": np.median(probs[gold[:, k], k]),
                    "median_on_polarized_without_label": np.median(probs[polarized & ~gold[:, k], k]),
                    "median_on_neutral": np.median(probs[~polarized, k]),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    data = load_predictions(PROJECT_ROOT / "outputs" / "runs")
    n = len(data["m4_core"][SEEDS[0]]["gold_det"])
    rng = np.random.default_rng(2026)
    weights = rng.multinomial(n, np.full(n, 1.0 / n), size=2000).astype(np.float64)

    boot, point = {}, {}
    for model in MODELS:
        per_boot = [metrics(weights @ count_columns(data[model][s])) for s in SEEDS]
        per_point = [metrics(count_columns(data[model][s]).sum(axis=0)) for s in SEEDS]
        boot[model] = {k: np.mean([d[k] for d in per_boot], axis=0) for k in per_boot[0]}
        point[model] = {k: np.array([float(d[k]) for d in per_point]) for k in per_point[0]}

    rows = []
    for a, b in (("m4_core", "m3_symmetric"), ("m3_symmetric", "m2")):
        for metric in point[a]:
            dist = boot[a][metric] - boot[b][metric]
            seed_diff = point[a][metric] - point[b][metric]
            rows.append(
                {
                    "model_a": a,
                    "model_b": b,
                    "metric": metric,
                    "mean_a": point[a][metric].mean(),
                    "sd_a": point[a][metric].std(ddof=1),
                    "mean_b": point[b][metric].mean(),
                    "sd_b": point[b][metric].std(ddof=1),
                    "diff": seed_diff.mean(),
                    "ci_low": np.percentile(dist, 2.5),
                    "ci_high": np.percentile(dist, 97.5),
                    "p_boot": min(1.0, 2 * min(np.mean(dist <= 0), np.mean(dist >= 0))),
                    "seeds_a_better": int((seed_diff > 0).sum()),
                }
            )
    pd.DataFrame(rows).to_csv(OUT / "rq2_decomposition.csv", index=False)

    label_rows = []
    for model in MODELS:
        for k, label in enumerate(TYPE_LABELS):
            coverage, conditional, recall = [], [], []
            for s in SEEDS:
                e = data[model][s]
                gold = (e["gold_det"] == 1) & (e["gold_types"][:, k] == 1)
                detected = gold & (e["pred_det"] == 1)
                hit = detected & (e["pred_types"][:, k] == 1)
                coverage.append(detected.sum() / gold.sum())
                conditional.append(hit.sum() / max(detected.sum(), 1))
                recall.append(((e["pred_types"][:, k] == 1) & gold).sum() / gold.sum())
            label_rows.append(
                {
                    "model": model,
                    "label": label,
                    "coverage": np.mean(coverage),
                    "conditional_recall": np.mean(conditional),
                    "recall": np.mean(recall),
                }
            )
    pd.DataFrame(label_rows).to_csv(OUT / "rq2_per_label_decomposition.csv", index=False)

    seeds = pd.read_csv(OUT / "seed_level_metrics.csv").set_index(["model", "seed"])
    seed_rows = []
    for s in SEEDS:
        m4, m3, m2 = (seeds.loc[(m, s)] for m in ("m4_core", "m3_symmetric", "m2"))
        seed_rows.append(
            {
                "seed": s,
                "m4_type_f1": m4["type_macro_f1"],
                "m3_type_f1": m3["type_macro_f1"],
                "m4_minus_m3_type_f1": m4["type_macro_f1"] - m3["type_macro_f1"],
                "m4_type_recall": m4["type_macro_recall"],
                "m3_type_recall": m3["type_macro_recall"],
                "m4_minus_m3_type_recall": m4["type_macro_recall"] - m3["type_macro_recall"],
                "m4_minus_m3_type_precision": m4["type_macro_precision"] - m3["type_macro_precision"],
                "m4_minus_m3_det_f1": m4["det_macro_f1"] - m3["det_macro_f1"],
                "m4_minus_m3_multilabel_recall": m4["multilabel_recall"] - m3["multilabel_recall"],
                "m3_minus_m2_type_f1": m3["type_macro_f1"] - m2["type_macro_f1"],
                "m3_minus_m2_type_recall": m3["type_macro_recall"] - m2["type_macro_recall"],
            }
        )
    pd.DataFrame(seed_rows).to_csv(OUT / "rq2_seed_differences.csv", index=False)
    probability_profile(PROJECT_ROOT / "outputs" / "runs").to_csv(
        OUT / "rq2_probability_profile.csv", index=False
    )
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
