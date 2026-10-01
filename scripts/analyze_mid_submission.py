#!/usr/bin/env python3
"""Mid-submission analysis: RQ1/RQ2 tables, paired bootstrap CIs, gating audit.

Reads the saved out-of-fold predictions of M1, M2, M3 (one-way and symmetric)
and M4-core for every seed, and writes CSV/JSON/Markdown tables to
``outputs/analysis``. Paper tables and figures are drawn from these files by
``scripts/make_report_assets.py``. All metrics are computed from predictions, never from
W&B summaries.

Protocol (frozen before the experiments; see docs/experiments.md):
- For each seed, the five outer folds are concatenated into one 3,222-example
  out-of-fold prediction set and scored once. Reported values are the mean and
  standard deviation over the five seed-level scores.
- TYPE metrics are computed on gold DET-positive examples (primary). TYPE on all
  examples is reported as a secondary view.
- Paired differences use an example-level bootstrap: the same resampled example
  multiset is applied to both models and all seeds, the per-seed difference is
  averaged over seeds, and the 2.5/97.5 percentiles give the 95% CI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from polar_hierarchy.constants import TYPE_LABELS  # noqa: E402
from polar_hierarchy.metrics import calculate_metrics  # noqa: E402

SEEDS = (13, 21, 42, 87, 100)
FOLDS = range(5)
MODELS = ("m1", "m2", "m3_one_way", "m3_symmetric", "m4_core")
MODEL_NAMES = {
    "m1": "M1 independent",
    "m2": "M2 shared MTL",
    "m3_one_way": "M3 one-way",
    "m3_symmetric": "M3 symmetric",
    "m4_core": "M4-core noisy-OR",
}
LABEL_NAMES = {
    "political": "Political",
    "racial_ethnic": "Racial/ethnic",
    "religious": "Religious",
    "gender_sexual_identity": "Gender/sexual",
    "other": "Other",
}
L = len(TYPE_LABELS)

# Column layout of the per-example indicator matrix.
DET = slice(0, 6)  # tp1 fp1 fn1 tp0 fp0 fn0
TPOS = slice(6, 6 + 3 * L)  # TYPE on gold-positive: tp[L] fp[L] fn[L]
TALL = slice(6 + 3 * L, 6 + 6 * L)  # TYPE on all examples
LVR = slice(6 + 6 * L, 9 + 6 * L)  # a, b, total
N_COL = 9 + 6 * L
ML = slice(N_COL + 1, N_COL + 5)  # multi gold, multi hit, single gold, single hit
K = N_COL + 5

PRIMARY = [
    ("RQ1", "m4_core", "m1", "det_macro_f1"),
    ("RQ1", "m4_core", "m1", "type_macro_f1"),
    ("RQ1", "m4_core", "m2", "det_macro_f1"),
    ("RQ1", "m4_core", "m2", "type_macro_f1"),
    ("RQ2", "m4_core", "m3_symmetric", "type_macro_f1"),
    ("RQ2", "m4_core", "m3_symmetric", "type_macro_recall"),
]
SECONDARY = [
    ("RQ1", "m4_core", "m1", "type_macro_precision"),
    ("RQ1", "m4_core", "m1", "type_macro_recall"),
    ("RQ1", "m4_core", "m2", "type_macro_precision"),
    ("RQ1", "m4_core", "m2", "type_macro_recall"),
    ("RQ1", "m4_core", "m1", "det_precision"),
    ("RQ1", "m4_core", "m1", "det_recall"),
    ("RQ1", "m4_core", "m2", "det_precision"),
    ("RQ1", "m4_core", "m2", "det_recall"),
    ("RQ1", "m4_core", "m1", "type_all_macro_f1"),
    ("RQ1", "m4_core", "m2", "type_all_macro_f1"),
    ("RQ1", "m2", "m1", "det_macro_f1"),
    ("RQ1", "m2", "m1", "type_macro_f1"),
    ("RQ2", "m4_core", "m3_symmetric", "type_macro_precision"),
    ("RQ2", "m4_core", "m3_symmetric", "det_macro_f1"),
    ("RQ2", "m4_core", "m3_symmetric", "multilabel_recall"),
    ("RQ2", "m4_core", "m3_symmetric", "type_all_macro_f1"),
    ("RQ2", "m3_symmetric", "m2", "type_macro_f1"),
    ("RQ2", "m3_symmetric", "m2", "type_macro_recall"),
    ("RQ2", "m3_symmetric", "m2", "type_macro_precision"),
    ("RQ2", "m3_symmetric", "m2", "det_macro_f1"),
    ("RQ2", "m3_symmetric", "m2", "type_all_macro_f1"),
    ("RQ2", "m3_symmetric", "m2", "multilabel_recall"),
]
METRIC_NAMES = {
    "det_macro_f1": "DET macro-F1",
    "det_precision": "DET precision (pos.)",
    "det_recall": "DET recall (pos.)",
    "type_macro_f1": "TYPE macro-F1",
    "type_macro_precision": "TYPE macro-P",
    "type_macro_recall": "TYPE macro-R",
    "type_all_macro_f1": "TYPE macro-F1 (all ex.)",
    "type_all_macro_precision": "TYPE macro-P (all ex.)",
    "type_all_macro_recall": "TYPE macro-R (all ex.)",
    "multilabel_recall": "Label recall, multi-label ex.",
    "singlelabel_recall": "Label recall, single-label ex.",
    "lvr_a": "LVR-A",
    "lvr_b": "LVR-B",
    "lvr_total": "LVR-total",
}


# --------------------------------------------------------------------------- loading


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def prediction_path(runs: Path, model: str, fold: int, seed: int) -> Path:
    if model == "m1":
        return runs / f"m1-fold{fold}-seed{seed}/predictions/outer_fold_predictions.jsonl"
    if model == "m2":
        return runs / f"m2-fold{fold}-seed{seed}/predictions/outer_fold_predictions.jsonl"
    if model in ("m3_one_way", "m3_symmetric"):
        return runs / f"m2-fold{fold}-seed{seed}/m3/{model}_predictions.jsonl"
    return runs / f"m4_core-fold{fold}-seed{seed}/predictions/outer_fold_predictions.jsonl"


def load_predictions(runs: Path) -> dict:
    data: dict = {}
    for model in MODELS:
        for seed in SEEDS:
            records = []
            for fold in FOLDS:
                records.extend(read_jsonl(prediction_path(runs, model, fold, seed)))
            records.sort(key=lambda item: item["example_id"])
            ids = [item["example_id"] for item in records]
            if len(ids) != len(set(ids)):
                raise ValueError(f"Duplicate IDs for {model} seed {seed}")
            entry = {
                "ids": ids,
                "fold": np.asarray([item["fold"] for item in records]),
                "gold_det": np.asarray([item["gold_det"] for item in records], dtype=np.int64),
                "gold_types": np.asarray([item["gold_types"] for item in records], dtype=np.int64),
                "pred_det": np.asarray(
                    [item["final_det_prediction"] for item in records], dtype=np.int64
                ),
                "pred_types": np.asarray(
                    [item["final_type_predictions"] for item in records], dtype=np.int64
                ),
            }
            if model.startswith("m3"):
                entry["raw_det"] = np.asarray([item["raw_det_prediction"] for item in records])
                entry["raw_types"] = np.asarray([item["raw_type_predictions"] for item in records])
            data.setdefault(model, {})[seed] = entry

    reference = data["m1"][SEEDS[0]]
    for model in MODELS:
        for seed in SEEDS:
            entry = data[model][seed]
            if entry["ids"] != reference["ids"]:
                raise ValueError(f"Example IDs differ for {model} seed {seed}")
            if not (
                np.array_equal(entry["gold_det"], reference["gold_det"])
                and np.array_equal(entry["gold_types"], reference["gold_types"])
            ):
                raise ValueError(f"Gold labels differ for {model} seed {seed}")
    return data


# --------------------------------------------------------------------------- metrics


def indicator_matrix(gold_det, gold_types, pred_det, pred_types) -> np.ndarray:
    """Per-example 0/1 indicators whose weighted sums give every metric."""
    gd, gt = gold_det.astype(bool), gold_types.astype(bool)
    pd_, pt = pred_det.astype(bool), pred_types.astype(bool)
    n = len(gd)
    m = np.zeros((n, K), dtype=np.float64)
    m[:, 0], m[:, 1], m[:, 2] = pd_ & gd, pd_ & ~gd, ~pd_ & gd
    m[:, 3], m[:, 4], m[:, 5] = ~pd_ & ~gd, ~pd_ & gd, pd_ & ~gd
    tp, fp, fn = pt & gt, pt & ~gt, ~pt & gt
    pos = gd[:, None]
    base = TPOS.start
    m[:, base : base + L] = tp & pos
    m[:, base + L : base + 2 * L] = fp & pos
    m[:, base + 2 * L : base + 3 * L] = fn & pos
    base = TALL.start
    m[:, base : base + L] = tp
    m[:, base + L : base + 2 * L] = fp
    m[:, base + 2 * L : base + 3 * L] = fn
    any_type = pt.any(axis=1)
    lvr_a, lvr_b = ~pd_ & any_type, pd_ & ~any_type
    m[:, LVR.start], m[:, LVR.start + 1], m[:, LVR.start + 2] = lvr_a, lvr_b, lvr_a | lvr_b
    m[:, N_COL] = 1.0
    n_gold = gt.sum(axis=1)
    multi, single = n_gold >= 2, n_gold == 1
    m[:, ML.start] = np.where(multi, n_gold, 0)
    m[:, ML.start + 1] = np.where(multi, tp.sum(axis=1), 0)
    m[:, ML.start + 2] = np.where(single, n_gold, 0)
    m[:, ML.start + 3] = np.where(single, tp.sum(axis=1), 0)
    return m


def _ratio(numerator, denominator):
    numerator, denominator = np.asarray(numerator, float), np.asarray(denominator, float)
    out = np.zeros(np.broadcast(numerator, denominator).shape)
    np.divide(numerator, denominator, out=out, where=denominator > 0)
    return out


def _prf(tp, fp, fn):
    return _ratio(tp, tp + fp), _ratio(tp, tp + fn), _ratio(2 * tp, 2 * tp + fp + fn)


def metrics_from_counts(c: np.ndarray) -> dict:
    """Vectorised metrics from summed indicators; ``c`` has shape (..., K)."""
    out = {}
    p1, r1, f1 = _prf(c[..., 0], c[..., 1], c[..., 2])
    _, _, f0 = _prf(c[..., 3], c[..., 4], c[..., 5])
    out["det_macro_f1"] = (f1 + f0) / 2
    out["det_precision"], out["det_recall"] = p1, r1
    for prefix, block in (("type", TPOS), ("type_all", TALL)):
        b = block.start
        p, r, f = _prf(c[..., b : b + L], c[..., b + L : b + 2 * L], c[..., b + 2 * L : b + 3 * L])
        out[f"{prefix}_macro_precision"] = p.mean(axis=-1)
        out[f"{prefix}_macro_recall"] = r.mean(axis=-1)
        out[f"{prefix}_macro_f1"] = f.mean(axis=-1)
        if prefix == "type":
            for index, label in enumerate(TYPE_LABELS):
                out[f"{label}_precision"] = p[..., index]
                out[f"{label}_recall"] = r[..., index]
                out[f"{label}_f1"] = f[..., index]
    n = c[..., N_COL]
    out["lvr_a"] = _ratio(c[..., LVR.start], n)
    out["lvr_b"] = _ratio(c[..., LVR.start + 1], n)
    out["lvr_total"] = _ratio(c[..., LVR.start + 2], n)
    out["multilabel_recall"] = _ratio(c[..., ML.start + 1], c[..., ML.start])
    out["singlelabel_recall"] = _ratio(c[..., ML.start + 3], c[..., ML.start + 2])
    return out


def self_check(matrices: dict, data: dict) -> None:
    """The vectorised metrics must equal the project's reference implementation."""
    for model in MODELS:
        entry = data[model][SEEDS[0]]
        reference = calculate_metrics(
            entry["gold_det"], entry["gold_types"], entry["pred_det"], entry["pred_types"], True
        )
        mine = metrics_from_counts(matrices[model][SEEDS[0]].sum(axis=0))
        for key in ("det_macro_f1", "type_macro_f1", "type_macro_precision",
                    "type_macro_recall", "lvr_a", "lvr_b", "lvr_total"):
            if not np.isclose(reference[key], float(mine[key]), atol=1e-12):
                raise AssertionError(f"{model} {key}: {reference[key]} != {float(mine[key])}")


# --------------------------------------------------------------------------- statistics


def holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    m = len(p_values)
    adjusted = np.empty(m)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def paired_comparisons(matrices: dict, n_boot: int, seed: int) -> pd.DataFrame:
    n = next(iter(matrices["m1"].values())).shape[0]
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot).astype(np.float64)

    boot: dict = {}
    point: dict = {}
    for model in MODELS:
        per_seed_boot = [metrics_from_counts(weights @ matrices[model][s]) for s in SEEDS]
        per_seed_point = [metrics_from_counts(matrices[model][s].sum(axis=0)) for s in SEEDS]
        boot[model] = {k: np.mean([d[k] for d in per_seed_boot], axis=0) for k in per_seed_boot[0]}
        point[model] = {k: np.array([float(d[k]) for d in per_seed_point]) for k in per_seed_point[0]}

    rows = []
    for tier, comparisons in (("primary", PRIMARY), ("secondary", SECONDARY)):
        for rq, a, b, metric in comparisons:
            seed_diffs = point[a][metric] - point[b][metric]
            dist = boot[a][metric] - boot[b][metric]
            p_value = min(1.0, 2 * min(np.mean(dist <= 0), np.mean(dist >= 0)))
            rows.append(
                {
                    "tier": tier,
                    "rq": rq,
                    "model_a": a,
                    "model_b": b,
                    "metric": metric,
                    "mean_a": point[a][metric].mean(),
                    "mean_b": point[b][metric].mean(),
                    "diff": seed_diffs.mean(),
                    "ci_low": np.percentile(dist, 2.5),
                    "ci_high": np.percentile(dist, 97.5),
                    "p_boot": p_value,
                    "seed_diff_min": seed_diffs.min(),
                    "seed_diff_max": seed_diffs.max(),
                    "seeds_a_better": int((seed_diffs > 0).sum()),
                }
            )
    frame = pd.DataFrame(rows)
    primary = frame["tier"] == "primary"
    frame.loc[primary, "p_holm"] = holm(frame.loc[primary, "p_boot"].tolist())
    return frame, boot, point


def per_label_comparisons(boot: dict, point: dict) -> pd.DataFrame:
    rows = []
    for a, b in (("m4_core", "m3_symmetric"), ("m4_core", "m2"), ("m4_core", "m1")):
        for label in TYPE_LABELS:
            for stat in ("recall", "precision", "f1"):
                metric = f"{label}_{stat}"
                dist = boot[a][metric] - boot[b][metric]
                rows.append(
                    {
                        "model_a": a,
                        "model_b": b,
                        "label": label,
                        "stat": stat,
                        "diff": (point[a][metric] - point[b][metric]).mean(),
                        "ci_low": np.percentile(dist, 2.5),
                        "ci_high": np.percentile(dist, 97.5),
                        "p_boot": min(1.0, 2 * min(np.mean(dist <= 0), np.mean(dist >= 0))),
                    }
                )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- audits


def gating_audit(data: dict) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        sym = data["m3_symmetric"][seed]
        gd = sym["gold_det"].astype(bool)
        gt = sym["gold_types"].astype(bool)
        raw_t, raw_d = sym["raw_types"].astype(bool), sym["raw_det"].astype(bool)
        fin_t, fin_d = sym["pred_types"].astype(bool), sym["pred_det"].astype(bool)
        removed = raw_t & ~fin_t
        rows.append(
            {
                "seed": seed,
                "m2_type_labels_predicted": int(raw_t.sum()),
                "labels_removed": int(removed.sum()),
                "removed_on_gold_negative": int((removed & ~gd[:, None]).sum()),
                "removed_fp_on_gold_positive": int((removed & gd[:, None] & ~gt).sum()),
                "removed_tp_on_gold_positive": int((removed & gt).sum()),
                "examples_gated": int((raw_t.any(axis=1) & ~raw_d).sum()),
                "gold_positive_examples_gated": int((raw_t.any(axis=1) & ~raw_d & gd).sum()),
                "gold_positive_examples_gated_with_correct_type": int(
                    ((raw_t & gt).any(axis=1) & ~raw_d & gd).sum()
                ),
                "det_predictions_changed": int((raw_d != fin_d).sum()),
                "m2_det_false_negatives": int((~raw_d & gd).sum()),
                "m2_lvr_b_examples": int((raw_d & ~raw_t.any(axis=1)).sum()),
            }
        )
    return pd.DataFrame(rows)


def det_error_profile(data: dict) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        for seed in SEEDS:
            e = data[model][seed]
            gd, pd_ = e["gold_det"].astype(bool), e["pred_det"].astype(bool)
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "false_positives": int((pd_ & ~gd).sum()),
                    "false_negatives": int((~pd_ & gd).sum()),
                    "predicted_positive": int(pd_.sum()),
                    "type_labels_on_gold_negative": int(e["pred_types"][~gd].sum()),
                    "gold_negative_with_any_type": int(e["pred_types"][~gd].any(axis=1).sum()),
                }
            )
    return pd.DataFrame(rows).groupby("model").mean(numeric_only=True).drop(columns="seed")


def run_metadata(runs: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for model in ("m1_det", "m1_type", "m2", "m4_core"):
        for seed in SEEDS:
            for fold in FOLDS:
                run = runs / f"{model}-fold{fold}-seed{seed}"
                metrics = json.loads((run / "metrics/outer_fold_metrics.json").read_text())
                events = [json.loads(x) for x in (run / "logs/events.jsonl").read_text().splitlines()]
                epochs_run = sum(e["event"] == "epoch_end" for e in events)
                rows.append(
                    {
                        "model": model,
                        "seed": seed,
                        "fold": fold,
                        "best_epoch": metrics["best_epoch"],
                        "epochs_run": epochs_run,
                        "runtime_min": metrics["runtime_seconds"] / 60,
                        "det_threshold": metrics["det_threshold"] if model in ("m1_det", "m2") else np.nan,
                        **{
                            f"thr_{label}": metrics["type_thresholds"][i] if model != "m1_det" else np.nan
                            for i, label in enumerate(TYPE_LABELS)
                        },
                    }
                )
    frame = pd.DataFrame(rows)
    training = frame.groupby("model").agg(
        best_epoch_mean=("best_epoch", "mean"),
        best_epoch_min=("best_epoch", "min"),
        best_epoch_max=("best_epoch", "max"),
        best_epoch_at_cap=("best_epoch", lambda s: int((s == 8).sum())),
        epochs_run_mean=("epochs_run", "mean"),
        runtime_min_mean=("runtime_min", "mean"),
        runtime_min_total=("runtime_min", "sum"),
    )
    thr_cols = ["det_threshold", *[f"thr_{x}" for x in TYPE_LABELS]]
    thresholds = frame.groupby("model")[thr_cols].agg(["mean", "std", "min", "max"])
    edge = frame.groupby("model")[thr_cols].agg(lambda s: int(((s <= 0.2) | (s >= 0.8)).sum()))
    edge.columns = pd.MultiIndex.from_product([thr_cols, ["at_grid_edge"]])
    return training, pd.concat([thresholds, edge], axis=1).sort_index(axis=1)


def parameter_counts() -> dict:
    try:
        import torch
        from transformers import AutoConfig, AutoModel

        config = AutoConfig.from_pretrained(
            "microsoft/deberta-v3-base", revision="de19fe7db5162df5f3d8f0b41321c0267288fd74"
        )
        with torch.device("meta"):
            encoder = AutoModel.from_config(config)
        enc = sum(p.numel() for p in encoder.parameters())
    except Exception:  # Offline fallback: published size of DeBERTa-v3-base backbone.
        enc = 183_831_552
    h = 768
    det, typ = h + 1, 5 * h + 5
    return {
        "encoder": enc,
        "m1": 2 * enc + det + typ,
        "m2": enc + det + typ,
        "m3_one_way": enc + det + typ,
        "m3_symmetric": enc + det + typ,
        "m4_core": enc + typ,
    }


# --------------------------------------------------------------------------- tables


def fmt(mean, std, digits=3):
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def write_tables(summary, comparisons, per_label_cmp, audit, det_errors, training, params,
                 support, out: Path):
    md = []

    main_cols = ["det_macro_f1", "type_macro_f1", "type_macro_precision", "type_macro_recall",
                 "lvr_a", "lvr_b", "lvr_total"]
    header = "| Model | Params | " + " | ".join(METRIC_NAMES[c] for c in main_cols) + " |"
    md.append("### Main results\n\n" + header + "\n|" + "---|" * (len(main_cols) + 2))
    for m in MODELS:
        cells = []
        for c in main_cols:
            mu, sd = summary.loc[m, (c, "mean")], summary.loc[m, (c, "std")]
            if c.startswith("lvr"):
                cells.append(fmt(100 * mu, 100 * sd, 1) + "%")
            else:
                cells.append(fmt(mu, sd))
        md.append(f"| {MODEL_NAMES[m]} | {params[m] / 1e6:.0f}M | " + " | ".join(cells) + " |")

    sec_cols = ["det_precision", "det_recall", "type_all_macro_f1", "type_all_macro_precision",
                "type_all_macro_recall", "multilabel_recall", "singlelabel_recall"]
    md.append("\n### Secondary metrics\n\n| Model | " + " | ".join(METRIC_NAMES[c] for c in sec_cols)
              + " |\n|" + "---|" * (len(sec_cols) + 1))
    for m in MODELS:
        md.append(f"| {MODEL_NAMES[m]} | " + " | ".join(
            fmt(summary.loc[m, (c, "mean")], summary.loc[m, (c, "std")]) for c in sec_cols) + " |")

    md.append("\n### Per-label F1 (gold-polarized examples)\n\n| Model | "
              + " | ".join(f"{LABEL_NAMES[x]} (n={support[x]})" for x in TYPE_LABELS)
              + " |\n|" + "---|" * (L + 1))
    for stat in ("f1", "precision", "recall"):
        if stat != "f1":
            md.append(f"\n**Per-label {stat}**\n\n| Model | "
                      + " | ".join(LABEL_NAMES[x] for x in TYPE_LABELS) + " |\n|" + "---|" * (L + 1))
        for m in MODELS:
            vals = [(summary.loc[m, (f"{x}_{stat}", "mean")], summary.loc[m, (f"{x}_{stat}", "std")])
                    for x in TYPE_LABELS]
            md.append(f"| {MODEL_NAMES[m]} | " + " | ".join(fmt(a, b) for a, b in vals) + " |")

    md.append("\n### Paired comparisons (difference = A − B; 95% bootstrap CI; 2,000 resamples)\n\n"
              "| Tier | RQ | A − B | Metric | A | B | Diff | 95% CI | p (boot) | p (Holm) | Seeds A>B |\n"
              "|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in comparisons.iterrows():
        holm_p = "" if pd.isna(r.get("p_holm")) else f"{r['p_holm']:.3f}"
        md.append(
            f"| {r['tier']} | {r['rq']} | {MODEL_NAMES[r['model_a']]} − {MODEL_NAMES[r['model_b']]} | "
            f"{METRIC_NAMES[r['metric']]} | {r['mean_a']:.3f} | {r['mean_b']:.3f} | {r['diff']:+.3f} | "
            f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] | {r['p_boot']:.3f} | {holm_p} | "
            f"{r['seeds_a_better']}/5 |"
        )

    md.append("\n### Per-label paired differences (A − B)\n\n"
              "| A − B | Label | Stat | Diff | 95% CI | p (boot) |\n|---|---|---|---|---|---|")
    for _, r in per_label_cmp.iterrows():
        md.append(f"| {MODEL_NAMES[r['model_a']]} − {MODEL_NAMES[r['model_b']]} | "
                  f"{LABEL_NAMES[r['label']]} | {r['stat']} | {r['diff']:+.3f} | "
                  f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] | {r['p_boot']:.3f} |")

    mean, std = audit.mean(numeric_only=True), audit.std(numeric_only=True)
    md.append("\n### M2 → M3 gating audit (per seed over all 3,222 examples; mean ± sd)\n\n"
              "| Quantity | Value |\n|---|---|")
    for col in audit.columns:
        if col != "seed":
            md.append(f"| {col.replace('_', ' ')} | {mean[col]:.1f} ± {std[col]:.1f} |")

    md.append("\n### DET error profile (mean per seed over 3,222 examples)\n\n| Model | "
              + " | ".join(c.replace("_", " ") for c in det_errors.columns) + " |\n|"
              + "---|" * (len(det_errors.columns) + 1))
    for m in MODELS:
        md.append(f"| {MODEL_NAMES[m]} | " + " | ".join(f"{v:.1f}" for v in det_errors.loc[m]) + " |")

    md.append("\n### Training statistics (25 runs per trained condition)\n\n| Run type | "
              + " | ".join(c.replace("_", " ") for c in training.columns) + " |\n|"
              + "---|" * (len(training.columns) + 1))
    for name, row in training.iterrows():
        md.append(f"| {name} | " + " | ".join(f"{v:.1f}" for v in row) + " |")

    (out / "tables.md").write_text("\n".join(md) + "\n")


# --------------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="outputs/runs")
    parser.add_argument("--output-dir", default="outputs/analysis")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    args = parser.parse_args()

    runs, out = PROJECT_ROOT / args.runs, PROJECT_ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    data = load_predictions(runs)
    matrices = {
        m: {s: indicator_matrix(e["gold_det"], e["gold_types"], e["pred_det"], e["pred_types"])
            for s, e in data[m].items()}
        for m in MODELS
    }
    self_check(matrices, data)

    reference = data["m1"][SEEDS[0]]
    gold_pos = reference["gold_det"].astype(bool)
    support = {x: int(reference["gold_types"][:, i].sum()) for i, x in enumerate(TYPE_LABELS)}
    n_gold_types = reference["gold_types"].sum(axis=1)
    dataset = {
        "examples": int(len(gold_pos)),
        "gold_det_positive": int(gold_pos.sum()),
        "gold_det_negative": int((~gold_pos).sum()),
        "type_support": support,
        "multi_label_examples": int((n_gold_types >= 2).sum()),
        "single_label_examples": int((n_gold_types == 1).sum()),
    }

    seed_rows = []
    for m in MODELS:
        for s in SEEDS:
            values = metrics_from_counts(matrices[m][s].sum(axis=0))
            seed_rows.append({"model": m, "seed": s, **{k: float(v) for k, v in values.items()}})
    seed_frame = pd.DataFrame(seed_rows)
    seed_frame.to_csv(out / "seed_level_metrics.csv", index=False)
    summary = seed_frame.drop(columns="seed").groupby("model").agg(["mean", "std"]).loc[list(MODELS)]
    summary.to_csv(out / "model_summary.csv")

    comparisons, boot, point = paired_comparisons(matrices, args.bootstrap, args.bootstrap_seed)
    comparisons.to_csv(out / "paired_comparisons.csv", index=False)
    per_label_cmp = per_label_comparisons(boot, point)
    per_label_cmp.to_csv(out / "per_label_comparisons.csv", index=False)

    audit = gating_audit(data)
    audit.to_csv(out / "gating_audit_per_seed.csv", index=False)
    det_errors = det_error_profile(data).loc[list(MODELS)]
    det_errors.to_csv(out / "det_error_profile.csv")
    training, thresholds = run_metadata(runs)
    training.to_csv(out / "training_statistics.csv")
    thresholds.to_csv(out / "selected_thresholds.csv")
    params = parameter_counts()

    write_tables(summary, comparisons, per_label_cmp, audit, det_errors, training, params,
                 support, out)

    results = {
        "dataset": dataset,
        "parameters": params,
        "bootstrap": {"resamples": args.bootstrap, "seed": args.bootstrap_seed},
        "summary": {
            m: {k: {"mean": summary.loc[m, (k, "mean")], "std": summary.loc[m, (k, "std")]}
                for k in summary.columns.get_level_values(0).unique()}
            for m in MODELS
        },
        "comparisons": comparisons.to_dict(orient="records"),
        "gating_audit_mean": audit.drop(columns="seed").mean().to_dict(),
        "det_error_profile": det_errors.to_dict(orient="index"),
        "training": training.to_dict(orient="index"),
    }
    (out / "results.json").write_text(json.dumps(results, indent=2, default=float) + "\n")
    print((out / "tables.md").read_text())


if __name__ == "__main__":
    main()
