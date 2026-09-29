# POLAR Hierarchy Project

Research code for **Structure over Scale: Hierarchy-Constrained Modeling for English Online Polarization Detection**.

The repository implements the controlled mid-submission comparison:

- M1-DET and M1-TYPE: independent task-specific encoders;
- M2: shared encoder with unconstrained DET and TYPE heads;
- M3: deterministic post-processing of M2 predictions;
- M4-core: TYPE heads with differentiable noisy-OR detection.

The exact scientific and execution plan is in [mid_submission_execution_plan.md](mid_submission_execution_plan.md).
The current Ada-outage handoff, remaining task accounting, local execution
procedure, artifact merge, and final analysis steps are in
[docs/remaining_runs_local_recovery_plan.md](docs/remaining_runs_local_recovery_plan.md).

## Repository structure

```text
configs/                 experiment configurations
data/raw/                original data; ignored by Git
data/processed/          canonical data and fixed folds; ignored by Git
docs/                    data, experiment, Ada, and W&B documentation
scripts/                 training, fold generation, reconciliation, analysis
slurm/                   Ada/SLURM templates
src/polar_hierarchy/     reusable Python package
tests/                   unit tests for hierarchy and metrics
outputs/                 generated runs, predictions, metrics, logs, and plots
```

## Installation

Use Python 3.10 or 3.11 on the final machine.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,wandb,parquet]'
```

Installing PyTorch may require an Ada-specific CUDA wheel or module. Follow the cluster's recommended PyTorch installation rather than blindly using the local command above.

## Data preparation

The supplied English files are stored under `data/raw/` and inventoried in [data/README.md](data/README.md). Dataset contents are intentionally ignored by Git. The training code uses the canonical columns documented in [docs/data_schema.md](docs/data_schema.md).

After mapping the official data into the configured columns, generate fixed outer folds:

```bash
python scripts/create_folds.py \
  --config configs/create_folds.yaml \
  --output data/processed/english_train_folds.csv \
  --seed 2026
```

Inspect the printed fold-level label counts before training.

Then save a complete data audit and label-distribution graph:

```bash
python scripts/validate_data.py \
  --config configs/m2.yaml \
  --output-dir outputs/data_audit
```

## One sanity run

```bash
python scripts/train.py --config configs/m2.yaml --fold 0 --seed 42
```

For a quick smoke test, override epochs and the output directory:

```bash
python scripts/train.py \
  --config configs/m2.yaml \
  --fold 0 \
  --seed 42 \
  --set training.epochs=1 \
  --set experiment.output_dir=outputs/smoke
```

Each run creates:

```text
outputs/runs/<model>-fold<fold>-seed<seed>/
  checkpoints/                 empty unless retention is explicitly enabled
  logs/run.log
  logs/events.jsonl
  metrics/outer_fold_metrics.json
  plots/loss_curves.png
  plots/metric_curves.png
  plots/per_label_f1.png
  plots/*_probability_*.png
  predictions/outer_fold_predictions.jsonl
  environment.json
  resolved_config.json
```

For the 100-run Ada sweep, the best epoch is held in CPU memory only long
enough to perform the outer-fold evaluation. This avoids filling the shared
home-directory quota. To retain a deliberately selected run, add
`--set training.keep_checkpoint=true`; retained checkpoints are written under
`checkpoints/best/`.

## Creating M3

M3 has no training loop. Generate it from a completed M2 prediction file:

```bash
python scripts/reconcile_m3.py \
  --m2-predictions outputs/runs/m2-fold0-seed42/predictions/outer_fold_predictions.jsonl \
  --output-dir outputs/runs/m2-fold0-seed42/m3
```

This saves both `m3_one_way` and `m3_symmetric` predictions plus the gating audit.

## Combining M1

M1 uses separate DET and TYPE models. Combine matching predictions before computing joint metrics:

```bash
python scripts/combine_m1.py \
  --det-predictions outputs/runs/m1_det-fold0-seed42/predictions/outer_fold_predictions.jsonl \
  --type-predictions outputs/runs/m1_type-fold0-seed42/predictions/outer_fold_predictions.jsonl \
  --output outputs/runs/m1-fold0-seed42/predictions/outer_fold_predictions.jsonl \
  --metrics-output outputs/runs/m1-fold0-seed42/metrics/outer_fold_metrics.json
```

On Ada, matching M1 components are combined from the immutable 25-row plan via
`bash slurm/submit_m1_wave.sh START COUNT`.

## W&B

W&B is disabled by default. The committed configurations target entity `manavberiwal006-iiit-hyderabad` and project `anlp-project`. Enable logging only after authenticating securely:

```bash
python scripts/train.py \
  --config configs/m4_core.yaml \
  --fold 0 \
  --seed 42 \
  --set logging.wandb.enabled=true
```

If compute nodes have no outbound network:

```bash
export WANDB_MODE=offline
```

Never store `WANDB_API_KEY` in tracked files, YAML, SLURM scripts, shell history, or the shared account's profile. Ada training jobs require the key to be exported in the submitting shell and never fall back to the shared account's W&B CLI login. They also force the W&B destination to `manavberiwal006-iiit-hyderabad/anlp-project`. See [docs/ada_and_wandb.md](docs/ada_and_wandb.md).

## Ada/SLURM

The templates use the supplied Ada scheduling policy:

```text
-p u22 -A research --qos=medium --constraint=2080ti --exclude=gnode066
```

Ada permits at most eight submitted jobs and four running jobs per user. Generate the immutable run plans first:

```bash
python scripts/generate_run_plan.py
```

The frozen Ada `spell` environment is never modified. Before the first model
run, submit `sbatch slurm/setup_project_deps.sbatch`; this installs the missing
SentencePiece tokenizer dependency into the Git-ignored project directory
`.deps/` using `pip --target`.

After one validated sanity run, submit at most one eight-task wave:

```bash
bash slurm/submit_wave.sh 0 8
```

Wait for slots to clear before submitting the next wave (`8 8`, `16 8`, and so on). After all M2 runs finish, submit M3 in the same way with `slurm/submit_m3_wave.sh`. Review [docs/ada_and_wandb.md](docs/ada_and_wandb.md) before submission.

## Tests

```bash
pytest -q
```

The tests cover noisy-OR equivalence, both LVR directions, symmetric M3 consistency, and threshold behaviour. They do not download a transformer model.

## Reproducibility rules

- Use the same fixed folds and seed list for every model.
- Tune checkpoints and thresholds only on the inner validation split.
- Never tune against the outer fold.
- Preserve raw probabilities and predictions.
- Derive M3 from the exact matching M2 run.
- Generate paper metrics from saved out-of-fold predictions, not W&B summaries alone.
- Record the exact encoder revision, source commit, Ada job ID, and resolved config.

## Current status

The framework is implemented, but no model has been trained because the official data and runtime dependencies are not yet present in this workspace. Hyperparameters in `configs/` are starting values and must be validated with the first Ada sanity run.
