# Remaining Runs, Ada Recovery, and Final Analysis Plan

**Status date:** 29 September 2026  
**Repository:** `ANLP_Ugh`  
**Purpose:** finish the experiment safely while Ada is unavailable, then merge the Ada artifacts and produce the final analysis.

## 1. Current experiment status

The immutable training plan contains task IDs `0` through `99`. There is no task
100. Each group of four tasks is one fold/seed combination:

1. M1-DET;
2. M1-TYPE;
3. M2;
4. M4-core.

The complete sweep is therefore:

```text
4 trained conditions × 5 folds × 5 seeds = 100 GPU training tasks
```

### Training status known from this session

- Tasks `0–75` were run on Ada.
- Task `53` ended with Slurm state `FAILED` only because W&B timed out while
  uploading images. Its training and outer evaluation are valid: it has 645
  predictions, one `run_complete` event, a nonempty metrics file, and all four
  plots expected for M1-TYPE. It must not be retrained merely because of the
  Slurm state.
- Tasks `60–67` and `68–75` left the queue, but their final `sacct` rows were not
  pasted into this record. Verify them when Ada returns.
- Tasks `76–99` have not been run. **Twenty-four training tasks remain.**

### Postprocessing status

M1 combination and M3 reconciliation use separate CPU plan IDs `0–24`, one for
each fold/seed pair. IDs `0–18` were scheduled incrementally during the Ada
runs, but their complete artifact set must be audited when Ada returns. The
remaining training tasks correspond to postprocessing IDs `19–24`:

| CPU plan ID | Fold | Seed |
|---:|---:|---:|
| 19 | 3 | 100 |
| 20 | 4 | 13 |
| 21 | 4 | 21 |
| 22 | 4 | 42 |
| 23 | 4 | 87 |
| 24 | 4 | 100 |

## 2. What is and is not available locally

Already present locally:

- `data/raw/eng_train.csv`;
- `data/raw/eng_dev.csv`;
- `data/raw/eng_test.csv`;
- `slurm/run_plans/training.tsv`;
- `slurm/run_plans/m1.tsv`;
- `slurm/run_plans/m3.tsv`;
- all source code and configurations.

Not currently present locally:

- `data/processed/english_train_folds.csv` from Ada;
- the complete Ada `outputs/runs/` tree;
- a Python environment containing PyTorch, Transformers, and the other project
  dependencies.

The missing Ada outputs are **not required to train tasks 76–99**. They are
required later for the complete 100-run aggregation, M1/M3 auditing, paired
confidence intervals, and error analysis.

## 3. Reproducibility decision

Running tasks `76–99` locally is feasible, but it changes the hardware from Ada
CUDA GPUs to Apple Silicon MPS or CPU. Record this as a limitation. The
fold/seed grouping remains internally fair because every remaining fold/seed
combination contains all four trained conditions.

For the cleanest hardware-controlled experiment, wait for Ada. If schedule
pressure is more important, use the local procedure below and retain each
run's `environment.json`.

## 4. Create an isolated local environment

The Mac currently has only the system Python 3.9 and none of the ML
dependencies. Python 3.11 is preferred. Do not install into the system Python.

Install Python 3.11 if it is not already available:

```bash
brew install python@3.11
```

Create the project environment:

```bash
cd /Users/manavberiwal/Documents/ANLP_PROJECT/ANLP_Ugh

/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

This installation requires internet access and will download PyTorch and the
other dependencies. The first training task will also download
`microsoft/deberta-v3-base`. Keep several GB of free disk space.

Verify the environment:

```bash
python - <<'PY'
import torch
import transformers
import pandas
import iterstrat

print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("MPS available:", torch.backends.mps.is_available())
PY
```

If `MPS available` is `False`, the runs will use CPU and may be extremely slow.

## 5. Recreate and validate the fixed folds

Generate folds using the same configuration and seed used on Ada:

```bash
python scripts/create_folds.py \
  --config configs/create_folds.yaml \
  --output data/processed/english_train_folds.csv \
  --seed 2026

python scripts/validate_data.py \
  --config configs/m2.yaml \
  --output-dir outputs/data_audit
```

Save the local checksum:

```bash
shasum -a 256 data/processed/english_train_folds.csv \
  | tee outputs/data_audit/local_folds_sha256.txt
```

When Ada returns, copy its processed fold file to a temporary filename and
compare the fold assignments before accepting the mixed-machine experiment.
A byte-identical checksum is ideal. At minimum, every example ID must have the
same fold. If the mappings differ, tasks `76–99` must be rerun with Ada's fold
file before final aggregation.

## 6. Disable W&B for local runs

Local files are the authoritative outputs. Each training command below passes:

```text
--set logging.wandb.enabled=false
```

No W&B login or API key is required. Each run still saves metrics, predictions,
logs, resolved configuration, environment metadata, and PNG graphs beneath
`outputs/runs/`.

## 7. Run task 76 as the local validation task

Do not start all 24 runs immediately. First run task 76 alone:

```bash
cd /Users/manavberiwal/Documents/ANLP_PROJECT/ANLP_Ugh
source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false

TASK_ID=76
ROW=$(awk -F '\t' -v id="$TASK_ID" 'NR > 1 && $1 == id {print; exit}' \
  slurm/run_plans/training.tsv)
IFS=$'\t' read -r _ MODEL CONFIG FOLD SEED <<< "$ROW"

python scripts/train.py \
  --config "$CONFIG" \
  --fold "$FOLD" \
  --seed "$SEED" \
  --set experiment.output_dir=outputs/runs \
  --set logging.wandb.enabled=false
```

Task 76 should create:

```text
outputs/runs/m1_det-fold3-seed100/
```

Validate it:

```bash
RUN=outputs/runs/m1_det-fold3-seed100
test -s "$RUN/metrics/outer_fold_metrics.json"
test -s "$RUN/predictions/outer_fold_predictions.jsonl"
grep -q '"event": "run_complete"' "$RUN/logs/events.jsonl"
find "$RUN/plots" -type f -name '*.png' -print
```

Also inspect `environment.json`, runtime, warnings, loss curves, and the outer
prediction row count. Do not proceed if task 76 fails or the Mac becomes
unstable.

## 8. Run tasks 77–99 sequentially and resumably

Keep the Mac connected to power, prevent sleep, and keep the terminal session
open. Do not run multiple transformer jobs concurrently on the local GPU.

The loop below skips a task only when its `run_complete` event already exists.
It stops instead of overwriting a partial run directory.

```bash
cd /Users/manavberiwal/Documents/ANLP_PROJECT/ANLP_Ugh
source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false
mkdir -p outputs/local_runner
set -o pipefail

for TASK_ID in $(seq 77 99); do
  ROW=$(awk -F '\t' -v id="$TASK_ID" \
    'NR > 1 && $1 == id {print; exit}' slurm/run_plans/training.tsv)
  IFS=$'\t' read -r _ MODEL CONFIG FOLD SEED <<< "$ROW"
  RUN_DIR="outputs/runs/${MODEL}-fold${FOLD}-seed${SEED}"

  if [[ -f "$RUN_DIR/logs/events.jsonl" ]] && \
     grep -q '"event": "run_complete"' "$RUN_DIR/logs/events.jsonl"; then
    echo "Skipping completed task $TASK_ID: $RUN_DIR"
    continue
  fi

  if [[ -d "$RUN_DIR" ]]; then
    echo "Partial directory exists for task $TASK_ID: $RUN_DIR" >&2
    echo "Move it to a clearly named archive before retrying." >&2
    break
  fi

  echo "Starting task $TASK_ID: model=$MODEL fold=$FOLD seed=$SEED"
  python scripts/train.py \
    --config "$CONFIG" \
    --fold "$FOLD" \
    --seed "$SEED" \
    --set experiment.output_dir=outputs/runs \
    --set logging.wandb.enabled=false \
    2>&1 | tee "outputs/local_runner/task-${TASK_ID}.log" || break
done
```

If a task fails, do not delete its directory immediately. Inspect the log,
archive the partial directory, correct the cause, and resume the same loop.

## 9. What to do while Ada remains unavailable

The local tasks can be trained and reviewed independently. Do not attempt the
complete experiment aggregation yet because Ada contains most of the
prediction files.

For each local run, verify:

1. one `run_complete` event;
2. nonempty outer metrics;
3. nonempty outer predictions;
4. no NaN or traceback in `logs/run.log`;
5. plausible training/validation curves;
6. expected plots;
7. no retained large checkpoint when `training.keep_checkpoint=false`.

## 10. Merge Ada outputs without overwriting local work

When Ada returns, first verify the historical jobs:

```bash
sacct -j 2722293,2722313 \
  --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS
```

Then copy Ada's `outputs/runs/` into a separate local staging directory, for
example:

```text
outputs/from_ada/runs/
```

Preview the merge:

```bash
rsync -avn --ignore-existing \
  outputs/from_ada/runs/ outputs/runs/
```

Perform it only after reviewing the preview:

```bash
rsync -av --ignore-existing \
  outputs/from_ada/runs/ outputs/runs/
```

`--ignore-existing` prevents Ada files from overwriting local run directories.
If the same run exists in both places, compare it manually rather than choosing
one silently.

Also copy these for audit/reference:

- Ada `data/processed/english_train_folds.csv`, under a temporary name;
- `outputs/data_audit/`;
- `outputs/slurm/`;
- optionally the local `wandb/` directory for recovery of unsynchronized runs.

## 11. Audit all training runs

After merging, each trained condition must have 25 run directories:

```bash
for MODEL in m1_det m1_type m2 m4_core; do
  printf '%-10s ' "$MODEL"
  find outputs/runs -maxdepth 1 -type d \
    -name "${MODEL}-fold*-seed*" | wc -l
done
```

Expected result:

```text
m1_det     25
m1_type    25
m2         25
m4_core    25
```

Count successful training completion events:

```bash
find outputs/runs -path '*/logs/events.jsonl' -print0 \
  | xargs -0 grep -l '"event": "run_complete"' \
  | wc -l
```

The expected count for trained conditions is 100. Task 53 should count because
its local event was written before the W&B timeout.

## 12. Complete or repair all M1 and M3 postprocessing locally

After all inputs have been merged, the following loop safely fills any missing
M1 or M3 output for all 25 fold/seed pairs. It does not require a GPU or W&B.

```bash
cd /Users/manavberiwal/Documents/ANLP_PROJECT/ANLP_Ugh
source .venv/bin/activate

for TASK_ID in $(seq 0 24); do
  ROW=$(awk -F '\t' -v id="$TASK_ID" \
    'NR > 1 && $1 == id {print; exit}' slurm/run_plans/m1.tsv)
  IFS=$'\t' read -r _ FOLD SEED <<< "$ROW"

  DET="outputs/runs/m1_det-fold${FOLD}-seed${SEED}/predictions/outer_fold_predictions.jsonl"
  TYPE="outputs/runs/m1_type-fold${FOLD}-seed${SEED}/predictions/outer_fold_predictions.jsonl"
  M1_DIR="outputs/runs/m1-fold${FOLD}-seed${SEED}"
  M2="outputs/runs/m2-fold${FOLD}-seed${SEED}/predictions/outer_fold_predictions.jsonl"
  M3_DIR="outputs/runs/m2-fold${FOLD}-seed${SEED}/m3"

  if [[ ! -s "$M1_DIR/predictions/outer_fold_predictions.jsonl" ]]; then
    mkdir -p "$M1_DIR/metrics" "$M1_DIR/predictions"
    python scripts/combine_m1.py \
      --det-predictions "$DET" \
      --type-predictions "$TYPE" \
      --output "$M1_DIR/predictions/outer_fold_predictions.jsonl" \
      --metrics-output "$M1_DIR/metrics/outer_fold_metrics.json" || break
  fi

  if [[ ! -s "$M3_DIR/m3_symmetric_predictions.jsonl" ]]; then
    python scripts/reconcile_m3.py \
      --m2-predictions "$M2" \
      --output-dir "$M3_DIR" || break
  fi
done
```

Expected final counts:

```bash
find outputs/runs -maxdepth 1 -type d -name 'm1-fold*-seed*' | wc -l
find outputs/runs -path '*/m3/m3_one_way_predictions.jsonl' | wc -l
find outputs/runs -path '*/m3/m3_symmetric_predictions.jsonl' | wc -l
```

Each command should return `25`.

## 13. Generate the existing final summaries

Once every prediction file is present:

```bash
python scripts/summarize_predictions.py \
  outputs/runs/m1-fold*-seed*/predictions/outer_fold_predictions.jsonl \
  outputs/runs/m2-fold*-seed*/predictions/outer_fold_predictions.jsonl \
  outputs/runs/m2-fold*-seed*/m3/m3_one_way_predictions.jsonl \
  outputs/runs/m2-fold*-seed*/m3/m3_symmetric_predictions.jsonl \
  outputs/runs/m4_core-fold*-seed*/predictions/outer_fold_predictions.jsonl \
  --output-dir outputs/analysis
```

This produces:

```text
outputs/analysis/seed_level_metrics.csv
outputs/analysis/model_summary.csv
outputs/analysis/model_comparison.png
```

The existing summarizer calculates seed-level metrics plus model means and
standard deviations. It does not yet produce the paired 95% confidence
intervals required by the research plan. Add and validate that analysis before
making the final claim.

## 14. What W&B can contribute while Ada is down

W&B is useful now. It can support a **preliminary**, not final, analysis.

Available from synchronized training runs:

- training and validation loss curves;
- validation metric trajectories and early-stopping behavior;
- outer DET macro-F1;
- outer TYPE macro precision, recall, and F1;
- per-label precision, recall, F1, and support;
- hierarchy-violation metrics logged by trained models;
- run configuration, model type, fold, seed, and runtime;
- uploaded plots for runs whose image upload completed.

Useful analyses that can already be made from W&B:

1. **Training stability:** identify divergence, overfitting, early stopping, and
   seed/fold sensitivity.
2. **Preliminary performance ranking:** compare M2 and M4-core, and inspect the
   separate M1-DET/M1-TYPE components across available folds and seeds.
3. **Consistency-performance trade-off:** test whether M4-core eliminates
   hierarchy violations and whether this changes DET or TYPE performance.
4. **Rare-label behavior:** compare per-label performance, especially gender/
   sexual identity, religious, racial/ethnic, and `other` labels.
5. **Efficiency:** compare runtime and convergence epochs across architectures.

These directly support the research question: whether enforcing the DET–TYPE
hierarchy through postprocessing or model structure improves logical
consistency without unacceptable predictive-performance loss.

### Limits of W&B-only analysis

The current code did not upload full prediction JSONL files as W&B artifacts.
W&B therefore cannot by itself provide:

- M1 combined results, because M1 combination is CPU postprocessing;
- M3 one-way or symmetric results, because M3 is CPU postprocessing;
- paired example-level bootstrap confidence intervals;
- paired error analysis on the same examples;
- threshold and probability analyses requiring the saved prediction records;
- a complete recovery of task 53's missing image upload;
- definitive final tables if some runs did not synchronize.

Use the W&B Runs table to export preliminary scalar results to CSV, but label
them clearly as provisional. Do not use W&B summaries alone for the final
scientific conclusion.

### Contribution that can already be motivated

Even before final aggregation, the project contribution is clear:

- a controlled comparison of independent, shared multitask, post-hoc
  reconciled, and architecturally constrained hierarchy strategies;
- explicit measurement of hierarchy violations in both directions;
- evaluation of the trade-off between predictive quality and guaranteed
  logical consistency;
- seed- and fold-controlled analysis with per-label behavior for imbalanced
  polarization types;
- reproducible local predictions and configurations rather than reliance on a
  dashboard alone.

The magnitude and statistical strength of those contributions must wait for
the merged prediction files and paired confidence intervals.

## 15. Final research workflow after aggregation

1. Validate all 100 trained runs and 50 derived M1/M3 outputs.
2. Generate model means, standard deviations, plots, and per-label tables.
3. Add paired example-level bootstrap 95% confidence intervals while preserving
   model, fold, seed, and example pairing.
4. Compare primary conditions: M1, M2, M3-symmetric, and M4-core. Treat
   M3-one-way as a diagnostic condition.
5. Analyze hierarchy violations, performance trade-offs, rare labels, and
   representative paired errors.
6. Select a model using the frozen evaluation criteria, not the official test
   set.
7. If final test predictions or a reusable checkpoint are required, implement
   and run a separate final-fit workflow for the selected condition with
   checkpoint retention enabled. The current 100-run sweep intentionally does
   not retain large checkpoints.
8. Record the mixed Ada/local hardware limitation if local tasks are used.
9. Archive the final tables, plots, predictions, resolved configs, environment
   metadata, run plan, and analysis code with the report.

## 16. Immediate next action

Choose one of the following:

- **Preferred for experimental consistency:** wait for Ada, verify tasks
  `60–75`, and submit tasks `76–99` there.
- **Preferred for schedule continuity:** create the local Python 3.11
  environment, regenerate folds with seed 2026, validate task 76 on MPS, and
  then run tasks `77–99` sequentially with W&B disabled.

Do not start local tasks until the environment check, fold audit, and task-76
validation all pass.
