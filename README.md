# Structure over Scale: Hierarchy-Constrained Modeling for English Online Polarization Detection

Team **Ugh** — ANLP course project on [SemEval-2026 Task 9 (POLAR)](https://aclanthology.org/2026.semeval-1.453/), English, Subtasks 1–2.

In POLAR, a text is polarized (**DET**, Subtask 1) exactly when at least one
polarization type applies (**TYPE**, Subtask 2: Political, Racial/ethnic,
Religious, Gender/sexual, Other). This project asks whether that hierarchy
should be built into the model rather than repaired after prediction, and
compares four ways of handling it under one controlled setup.

## Links

| Resource | Link |
|---|---|
| Mid-submission report (ACL format) | [report/report.pdf](report/report.pdf) |
| Weights & Biases — all 100 training runs (public) | <https://wandb.ai/rakshitagg06-iiit-hyderabad/anlp-project> |
| Code | <https://github.com/Rakshitagg06/ANLP_Ugh> |
| Hugging Face models | Not yet released. The mid-submission sweep is evaluated by cross-validation and keeps no checkpoints; the final selected model will be trained with checkpoints and released on Hugging Face for the final submission. |

The W&B project holds each run's configuration, loss and metric curves, and
its out-of-fold predictions and metrics as artifacts.

## Models

All models use `microsoft/deberta-v3-base` (184M parameters) with mean pooling
and linear heads.

| Model | Description | Consistent by design? |
|---|---|:---:|
| **M1** | Two independent encoders: one for DET, one for TYPE | no |
| **M2** | One shared encoder with separate DET and TYPE heads | no |
| **M3** | M2 predictions repaired after decoding: types removed when DET = 0 (one-way); DET then set to OR(types) (symmetric) | yes |
| **M4-core** | One encoder with TYPE heads only; DET = noisy-OR of the type probabilities, `p_det = 1 − Π(1 − p_k)` | yes |

## Main results

Out-of-fold over the 3,222 English training texts; mean ± sd over 5 seeds.
TYPE scores use gold-polarized texts, as fixed before the experiments.

| Model | DET macro-F1 | TYPE macro-F1 | TYPE macro-R | TYPE macro-F1 (all texts) | Hierarchy violations |
|---|---:|---:|---:|---:|---:|
| M1 | **0.799** ± 0.007 | **0.543** ± 0.013 | **0.600** ± 0.023 | 0.299 ± 0.013 | 61.9% |
| M2 | 0.793 ± 0.003 | 0.497 ± 0.015 | 0.595 ± 0.019 | 0.255 ± 0.029 | 61.6% |
| M3 (symmetric) | 0.793 ± 0.003 | 0.453 ± 0.022 | 0.468 ± 0.036 | 0.382 ± 0.017 | 0.0% |
| M4-core | 0.784 ± 0.004 | 0.461 ± 0.007 | 0.474 ± 0.022 | **0.396** ± 0.008 | 0.0% |

- **RQ1** — does noisy-OR (M4-core) beat M1 and M2? **Not supported.** M4-core
  detects more polarized texts but raises more false alarms, and its consistency
  caps TYPE recall on the pre-declared metric.
- **RQ2** — at equal consistency, does training-time structure keep more TYPE
  recall than post-hoc gating (M3)? **Inconclusive** (TYPE F1 +0.008, 95% CI
  [−0.009, +0.022]). Gating removes 23% of M2's correct type labels; M4-core
  detects more texts but weakens rare labels.

Full analyses: [docs/C1_RQ1.md](docs/C1_RQ1.md) and [docs/C2_RQ2.md](docs/C2_RQ2.md).

## Repository layout

```text
configs/               training configs (m1_det, m1_type, m2, m4_core) and fold config
src/polar_hierarchy/   data, models, losses (noisy-OR), trainer, metrics, M3 reconciliation
scripts/               folds, training, sweep runner, post-processing, analysis, report assets
tests/                 unit tests (noisy-OR, LVR, reconciliation, thresholds, checkpoints)
slurm/                 SLURM templates and the fixed run plans (slurm/run_plans/*.tsv)
outputs/analysis/      result tables (CSV/JSON/Markdown) generated from the saved predictions
report/                report.pdf, its LaTeX source (acl_latex.tex), custom.bib, figures/
docs/                  RQ1/RQ2 analyses, results summary, protocol, data schema
```

## Setup

Python 3.10+ and a CUDA GPU (runs used an 8 GB RTX 4060).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev,wandb]'
```

The configs pin the encoder to a safetensors revision of
`microsoft/deberta-v3-base` that is tensor-identical to `main`.

**Data.** The POLAR data are not distributed with this repository. Place the
official English files at:

```text
data/train/eng.csv
data/dev/eng.csv
data/test/eng.csv
```

## Reproducing the results

```bash
# 1. Fixed 5-fold multilabel-stratified split (seed 2026)
python scripts/create_folds.py --config configs/create_folds.yaml \
  --output data/processed/english_train_folds.csv --seed 2026

# 2. All 100 training runs (4 models x 5 folds x 5 seeds); resumable
bash scripts/run_local_sweep.sh

# 3. Join M1-DET + M1-TYPE into M1, and derive both M3 variants from M2
bash scripts/postprocess_m1_m3.sh

# 4. Metrics, paired bootstrap tests, gating audit -> outputs/analysis/
python scripts/analyze_mid_submission.py
python scripts/analyze_rq2_decomposition.py

# 5. Report tables and figures (figures -> report/figures/)
python scripts/make_report_assets.py
```

`run_local_sweep.sh` logs to W&B and reads the API key from
`.secrets/keys.sh` (git-ignored), which should contain
`export WANDB_API_KEY=...`. To train a single run without W&B:

```bash
python scripts/train.py --config configs/m4_core.yaml --fold 0 --seed 42 \
  --set logging.wandb.enabled=false
```

Each run writes `outputs/runs/<model>-fold<k>-seed<s>/` with its predictions,
metrics, plots, logs and resolved config. Model weights are not kept unless
`--set training.keep_checkpoint=true` is passed.

**Tests:** `pytest -q` (no model download needed).

## Protocol in brief

- One fixed five-fold split; seeds 13, 21, 42, 87, 100.
- Early stopping, checkpoint selection and all thresholds use only an inner
  10% validation split; each held-out fold is predicted once.
- For each seed, the five held-out folds are scored as one 3,222-text set;
  results are mean ± sd over seeds.
- Comparisons use a paired bootstrap over texts (2,000 resamples) with Holm
  correction over the six pre-declared RQ1/RQ2 tests.
- All reported numbers are computed from saved predictions, not from W&B
  summaries.

Details: [docs/experiments.md](docs/experiments.md) and the report.
