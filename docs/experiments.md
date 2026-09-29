# Experiment protocol

## Fixed conditions

The primary comparison contains M1, M2, symmetric M3, and M4-core. `M3-one-way` is a diagnostic condition. Label-aware attention is post-mid work because including it only in M4 would confound the hierarchy comparison.

## Outer folds and seeds

Use the same five outer folds for all models. The SLURM template currently uses:

```text
13, 21, 42, 87, 100
```

Changing this list requires changing both the documentation and SLURM scripts before any final run.

## Model selection

Every outer-fold run creates an internal validation split from the remaining data. Early stopping and thresholds use only this internal split. The outer fold is evaluated once after the best checkpoint is restored.

## Thresholds

The default configuration searches 0.20 through 0.80 in increments of 0.05. TYPE uses per-label thresholds. M3 reuses the matching M2 thresholds. M4 hard DET is always the OR of thresholded TYPE predictions.

Threshold range and granularity are hyperparameters. Freeze them before the final sweep and keep the search budget identical across relevant models.

## Required run review

Before accepting a run, inspect:

1. `run.log` for warnings or NaNs;
2. `events.jsonl` for complete epochs;
3. loss and metric curves;
4. selected thresholds;
5. prediction record count;
6. label supports and per-label scores;
7. M3/M4 LVR invariants;
8. W&B configuration and Ada job metadata.

## Full-run accounting

```text
M1-DET: 25 trained/evaluated runs
M1-TYPE: 25 trained/evaluated runs
M2: 25 trained/evaluated runs
M4-core: 25 trained/evaluated runs
M1: deterministic combination of 25 matching M1-DET/M1-TYPE outputs
M3: deterministic processing of 25 M2 outputs
```

This is 100 GPU training jobs, not 125.

Because `/share1` is unavailable from compute nodes and 100 retained FP32
checkpoints would exceed the shared-home quota, best epoch weights are held in
CPU memory only through outer-fold evaluation. All metrics, thresholds,
probabilities, predictions, plots, configs, and logs remain persistent. Selected
models can be rerun later with checkpoint retention enabled.

On Ada these jobs are submitted from the immutable run plan in waves of no more than eight tasks because pending array tasks count against the per-user submission limit.
