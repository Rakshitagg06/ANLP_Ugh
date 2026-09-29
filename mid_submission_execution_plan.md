# Mid-Submission Execution Plan
## Structure over Scale: Hierarchy-Constrained Modeling for English Online Polarization Detection

## 1. Objective

The mid-submission will present a controlled first-stage study of the hierarchy between POLARDETECT (DET) and the five POLARTYPE (TYPE) labels. It will focus on C1/RQ1 (structured detection through noisy-OR) and C2/RQ2 (post-hoc consistency versus structurally trained consistency).

The target is a 7-8 page report excluding references, subject to confirmation against the official course brief. Later work will cover label-aware representations and the model-scale comparison.

## 2. Frozen research questions

### RQ1

Does deriving DET from TYPE probabilities improve DET and/or TYPE macro-F1 over independent task-specific models and an unconstrained shared-encoder multi-task model?

```text
M4-core vs M1
M4-core vs M2
```

### RQ2

At equal hard-prediction consistency, does a model trained with the hierarchy preserve more TYPE recall and F1 than post-hoc deterministic reconciliation?

```text
M4-core vs M3
```

RQ2 is a controlled model comparison. A causal interpretation requires all non-hierarchy components to be matched.

## 3. Contributions

- **C1:** Derive DET from TYPE probabilities using differentiable noisy-OR.
- **C2:** Compare post-hoc and training-time consistency using symmetric LVR and an audit of labels removed by gating.
- **C3 (post-mid):** Analyse label-aware representations, particularly on multi-label examples. Do not claim label-aware attention itself as novel.
- **C4 (post-mid):** Compare performance against parameter count for the 184M encoder and published 12B-27B systems.

Call C4 a **parameter-efficiency** or **performance-versus-model-scale** comparison. Do not call it compute-matched unless FLOPs, GPU-hours, or comparable costs are measured.

## 4. Dataset and label checks

Use the official English POLAR data described in the proposal:

```text
Train: 3,222
Development: 160
Test: 1,452 (supplied file includes labels; official status unconfirmed)
```

The supplied `eng_test.csv` contains binary label columns, despite the proposal's statement that official test gold was unavailable. Until the provenance and permitted use of these labels are confirmed, treat this file as a strictly held-out external evaluation set and never use it for model selection, threshold tuning, or debugging.

Before implementation, verify from the official task schema:

1. Every DET-positive training example has at least one TYPE label.
2. Every DET-negative example has no defined positive TYPE label.
3. Whether `Other` is mutually exclusive with the four named dimensions or may co-occur.
4. The official definitions of DET and TYPE macro-F1.
5. The required handling of labels with zero support.

The supplied files answer item 3: `Other` co-occurs with named dimensions in 121 training, 6 development, and 55 test examples. Therefore it must be modelled as an ordinary fifth multilabel category and must not be described as an exclusive residual category.

Use the 160-example development set only as an external sanity check. Main conclusions will use out-of-fold predictions from the 3,222 training examples.

## 5. Frozen model definitions

All main models use the same DeBERTa-v3-base architecture and tokenizer version. Record the exact model revision.

### M1 - Independent models

Train two separate encoders:

```text
M1-DET: encoder + binary DET head
M1-TYPE: separate encoder + five-label TYPE head
```

The TYPE loss is evaluated only where gold TYPE labels are defined. M1 predictions are not reconciled at inference. Because M1 contains two encoders, report its parameter count and training cost separately from the one-encoder models.

### M2 - Soft multi-task model

```text
one shared encoder
+ binary DET head
+ ordinary five-label TYPE head
```

```math
L = L_det + lambda * L_type
```

`L_type` is masked to gold DET-positive examples. M2 has no structural constraint and may violate the hierarchy.

### M3 - Post-hoc deterministic reconciliation

M3 is not trained separately. Derive it from raw M2 predictions using M2's frozen thresholds.

Let:

```text
d_raw = thresholded M2 DET prediction
t_raw[k] = thresholded M2 TYPE prediction for label k
```

Main symmetric reconciliation:

```text
t_m3[k] = t_raw[k] AND d_raw
d_m3 = OR over t_m3[k]
```

This guarantees symmetric `LVR = 0`.

Also retain a diagnostic **M3-one-way** condition matching the common gate:

```text
if d_raw = 0:
    set all TYPE predictions to 0
otherwise:
    keep TYPE predictions unchanged
```

M3-one-way removes `DET=0, TYPE-positive` violations but may retain `DET=1, no TYPE` violations. Never describe it as having zero symmetric LVR unless measurement confirms that.

Save both pre- and post-reconciliation predictions.

### M4-core - Controlled structured model

```text
one shared encoder
+ the same ordinary TYPE-head design used in M2
+ no independent DET head
```

For five TYPE probabilities `p_k`:

```math
p_det = 1 - product_k(1 - p_k)
```

```math
L = BCE(y_det, p_det) + lambda * masked_BCE(y_type, p_type)
```

Implement noisy-OR stably using clamping or log-space operations.

Hard inference:

```text
t_m4[k] = 1[p_k >= threshold_k]
d_m4 = OR over t_m4[k]
```

State the distinction accurately:

- Noisy-OR supplies differentiable structural coupling during training.
- The hard OR decoder guarantees zero binary LVR at inference.

M4-core is the primary model for RQ1 and RQ2 because it isolates the hierarchy mechanism.

### M4-label-aware - Post-mid extension

After the core comparison, replace the ordinary TYPE head with label-aware representations and test the special `Other` representation. This condition answers C3/RQ3 and must not be the only M4 result in the controlled M3-versus-M4 comparison.

## 6. Logical Violation Rate

```text
LVR = mean(d_pred != OR(type_pred))
```

Report:

```text
LVR-A: DET = 0 and at least one TYPE = 1
LVR-B: DET = 1 and every TYPE = 0
LVR-total: LVR-A + LVR-B as proportions of all examples
```

Expected properties:

```text
M1: may violate in either direction
M2: may violate in either direction
M3-one-way: LVR-A = 0; LVR-B may be non-zero
M3 symmetric: LVR-total = 0
M4-core hard predictions: LVR-total = 0
```

Unit tests must cover all four combinations of DET and `any(TYPE)`.

## 7. Cross-validation and thresholds

### Outer evaluation

Create one fixed five-fold split and reuse it for every model and seed. Use multilabel stratification over TYPE and DET where possible. If duplicate or near-duplicate texts exist, group them into the same fold before stratification.

Save:

```text
example_id -> fold_id
split-generation script
split seed
label counts per fold
```

### Seeds

Use five pre-declared seeds, for example:

```text
13, 21, 42, 87, 100
```

The exact list may change before experiments start but must then remain frozen.

### Inner validation

For each outer fold and seed:

1. Hold out the outer fold for final evaluation.
2. Create a fixed internal validation split from the other four folds.
3. Select the best epoch state and thresholds only on internal validation.
4. Freeze them.
5. Evaluate the outer fold once.

Never use the outer held-out fold for early stopping, threshold tuning, or model selection.

### Threshold policy

Pre-declare whether TYPE uses one shared threshold or five per-label thresholds. Per-label thresholds are acceptable if every model receives the same search procedure and budget.

M1 and M2 may select a DET threshold. M3 must reuse M2's frozen thresholds. M4's reported hard DET prediction is the OR of its thresholded TYPE predictions. Save every selected threshold.

## 8. Metrics and statistics

Match the official evaluation implementation exactly. Report at minimum:

```text
DET macro-F1
TYPE macro-F1 across the five labels
macro TYPE precision and recall
per-label precision, recall, F1, and support
LVR-A, LVR-B, and LVR-total
```

Label any additional micro or sample-averaged scores explicitly.

### Aggregation

For each seed, concatenate predictions from all five outer folds and calculate one complete out-of-fold score over all 3,222 examples. Report mean and standard deviation across the five seed-level scores. Do not treat the 25 fold-seed scores as independent datasets.

### Paired comparisons

Use the same folds, seeds, and examples for all models. Report paired metric differences with 95% confidence intervals using paired resampling of saved out-of-fold predictions while preserving model pairing and seed identity.

Primary comparisons:

```text
RQ1: M4-core - M1
RQ1: M4-core - M2
RQ2: M4-core - M3
```

### Gating audit

For M2 to M3, calculate:

```text
number of predicted TYPE labels removed
number of true-positive TYPE labels removed
number of false-positive TYPE labels removed
number of DET predictions changed by symmetric reconciliation
change in TYPE precision, recall, and macro-F1
```

This analysis is required for C2.

## 9. Saved outputs

For every outer-fold evaluation, save JSONL or Parquet records containing:

```text
example_id, fold, seed, model
gold_det, gold_types
raw_det_probability and raw_det_prediction, where applicable
type_probabilities and raw_type_predictions
final_det_prediction and final_type_predictions
det_threshold and type_thresholds
best epoch, best validation score, and checkpoint-retention status
config identifier or hash
```

For M3, retain both raw M2 and reconciled predictions. Also save the resolved configuration, final metrics, training history, model-selection metadata, and environment manifest. On Ada, the best state is held in CPU memory through outer-fold evaluation and is not retained for the 100-run sweep because compute nodes cannot access `/share1` and persistent FP32 checkpoints would exceed the shared-home quota. Selected models may be rerun later with checkpoint retention enabled.

## 10. Ada execution plan

### Can the models run simultaneously?

Yes, after the sanity stage and subject to the team's GPU/job quota.

Independent training jobs:

```text
M1-DET:  5 folds x 5 seeds = 25 jobs
M1-TYPE: 5 folds x 5 seeds = 25 jobs
M2:      5 folds x 5 seeds = 25 jobs
M4-core: 5 folds x 5 seeds = 25 jobs
M3:      no training; derive from completed M2 predictions
```

This is 100 training jobs plus inexpensive M3/M3-one-way post-processing jobs. Submit the training conditions as SLURM arrays. Ada will run as many concurrently as the account, partition, and quota allow; the remainder will queue.

M3 post-processing must depend on the corresponding successful M2 evaluation job.

### Safe launch sequence

1. Run a CPU-only data and metric test.
2. Run one tiny GPU smoke test on a small subset.
3. Run one complete fold/seed sanity job for M1-DET, M1-TYPE, M2, and M4-core.
4. Generate M3 from the sanity M2 predictions.
5. Verify losses, thresholds, files, LVR, best-state restoration, and W&B.
6. Estimate wall time, GPU memory, scratch use, and cost.
7. Submit full arrays with a conservative concurrency cap.

Logical arrays:

```text
m1_det[0-24]
m1_type[0-24]
m2[0-24]
m4_core[0-24]
m3_postprocess[0-24] after corresponding m2 jobs
```

Do not request more GPU memory or wall time than the sanity runs justify.

### Ada information required

Confirm before writing SLURM scripts:

```text
Ada hostname and login method
SLURM account or project code
GPU partition and allowed GPU types
maximum concurrent jobs/GPUs and QOS, if required
per-job CPU, RAM, GPU, and wall-time limits
home, project, and scratch quotas
recommended checkpoint and scratch locations
internet policy on compute nodes
CUDA, compiler, Python, and module versions
whether Conda or Apptainer/Singularity is preferred
dataset location and permissions
repository access method
pre-emption and checkpointing policy
```

Never place datasets, W&B keys, or other secrets in Git.

### Suggested layout

```text
configs/
  m1_det.yaml
  m1_type.yaml
  m2.yaml
  m4_core.yaml
  seeds.yaml
src/
  data/
  models/
  training/
  evaluation/
scripts/
  train.py
  evaluate.py
  reconcile_m3.py
  run_smoke_test.sh
  submit_array.sh
slurm/
  train_array.sbatch
  m3_postprocess.sbatch
outputs/
  predictions/
  metrics/
  configs/
analysis/
  rq1.py
  rq2.py
  bootstrap.py
  make_tables.py
```

Use one shared implementation controlled by resolved configurations.

## 11. Weights & Biases plan

W&B tracks experiments but is not the only location for predictions or results.

```text
project: polar-hierarchy
entity: <team-or-user-entity>
run name: <model>-fold<fold>-seed<seed>
```

Example: `m4core-fold2-seed42`.

### W&B information required

Confirm:

```text
W&B account and team/entity
project name and visibility
secure API-key method on Ada
whether compute nodes have outbound internet
artifact storage limits
```

Provide `WANDB_API_KEY` through a secure environment or cluster secret mechanism. Never put it in source files, YAML, SLURM scripts, command output, or Git.

If compute nodes cannot reach W&B, use:

```text
WANDB_MODE=offline
```

Store offline run directories in persistent project storage and sync them later from a network-enabled login node.

### Required run configuration

Log:

```text
model condition
encoder name and exact revision
fold, seed, dataset/split version
learning rate, batch size, gradient accumulation
maximum length, optimizer, scheduler, weight decay
lambda and class weights
epochs and early stopping
DET and TYPE thresholds
checkpoint-selection metric
code commit, when available
Ada job ID and GPU type
```

### Required metrics

Log:

```text
train and internal-validation loss
DET macro-F1
TYPE macro-F1
macro TYPE precision and recall
per-label precision, recall, and F1
LVR-A, LVR-B, and LVR-total
best epoch, runtime, and peak GPU memory if available
```

Final tables and confidence intervals must be regenerated from saved out-of-fold predictions rather than W&B summary values alone.

## 12. Experiment schedule

### Stage A - Pipeline and sanity

Complete label/metric checks, fixed folds, metric tests, threshold selection, M1/M2/M3/M4-core, prediction saving, one W&B test, and Ada smoke/sanity jobs.

Exit criteria:

```text
no data leakage
normal loss curves
all artifacts saved
M3 symmetric LVR = 0
M4 hard-prediction LVR = 0
M3-one-way directional behavior is correct
failures are visible and resumable
W&B metadata is complete
```

### Stage B - Full experiment

1. Submit M1-DET, M1-TYPE, M2, and M4-core arrays.
2. Let Ada parallelize within the account quota.
3. Run M3 post-processing after each M2 result.
4. Resubmit only failed or missing tasks.
5. Build seed-level out-of-fold prediction files.
6. Run paired RQ1 and RQ2 analyses.
7. Produce tables and confidence intervals.

## 13. Mid-submission results

Main table:

```text
Model | DET F1 | TYPE F1 | TYPE P | TYPE R | LVR-A | LVR-B | Total LVR
```

Include M1, M2, M3-one-way, M3 symmetric, and M4-core. Also report per-label F1 and support.

For RQ1, compare M4-core with M1 and M2 using paired differences and uncertainty. For RQ2, compare symmetric M3 with M4-core and use the gating audit to explain changes.

If M4-core preserves more recall, the result supports the training-time structural-coupling hypothesis under this controlled setup; it does not establish it universally. Report negative or inconclusive results without changing the research question after seeing them.

## 14. Report structure

Subject to the official brief:

1. Introduction and problem statement - about 0.75 page.
2. Literature review - about 1.75-2 pages.
3. Models and methodology - about 1.5 pages.
4. Progress and results - about 2-2.5 pages.
5. Remaining work and timeline - about 0.75 page.
6. Current conclusion - about 0.25 page.

Essential RQ1/RQ2 evidence stays in the main report. Detailed hyperparameters, thresholds, and diagnostics may go in an appendix.

## 15. Post-mid work

### C3/RQ3

Compare M4-core with label-aware attention and alternative `Other` handling. Analyse single-label and multi-label examples separately.

### C4/RQ4

Plot published performance against parameter count on a log-scaled x-axis with separate DET and TYPE panels. Do not use raw `F1 / parameter` as the only efficiency measure.

## 16. Final priority order

```text
1. Confirm the official mid-submission brief.
2. Confirm ontology and official evaluation code.
3. Obtain Ada account, quota, storage, and software details.
4. Create the private W&B project and secure authentication.
5. Freeze models, folds, seeds, metrics, and thresholds.
6. Implement and unit-test the shared pipeline.
7. Run local/CPU tests and Ada smoke tests.
8. Run one complete fold/seed sanity condition.
9. Verify consistency and saved artifacts.
10. Estimate resources and submit capped arrays.
11. Generate M3 from M2 results.
12. Assemble out-of-fold predictions per seed.
13. Complete paired RQ1/RQ2 analysis.
14. Produce tables and write the mid-submission.
15. Add W&B/model links to README if permitted.
16. Begin C3/RQ3 and C4/RQ4 after the core study is stable.
```

## 17. Definition of done

```text
model and inference definitions are frozen
folds and seeds are reproducible
thresholds are selected without outer-fold leakage
M1, M2, M3, and M4-core results are available
M3 and M4 show the stated LVR behavior
raw out-of-fold predictions are saved
RQ1/RQ2 have paired estimates and confidence intervals
M2-to-M3 gating damage is quantified
W&B and Ada job metadata are traceable
the report separates completed evidence from future work
```
