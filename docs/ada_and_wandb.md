# Ada and W&B operations

## Confirmed Ada configuration

The supplied operational guide specifies:

```text
host: ada.iiit.ac.in
partition: u22
account: research
QOS: medium
GPU constraint: 2080ti
excluded node: gnode066
MaxSubmitPU: 8
MaxJobsPU: 4
Python module: u22/python/3.12.4
environment: ~/envs/spell
```

Still confirm storage quotas, project checkout location, compute-node internet access, and whether the frozen environment contains this project's dependencies.

Do not submit the full arrays until a complete one-fold/one-seed sanity run establishes memory and runtime requirements.

## Parallel execution

M1-DET, M1-TYPE, M2, and M4-core are independent and can run simultaneously. The generated run plan interleaves the four model families.

Every array task counts against `MaxSubmitPU`, including pending tasks. Therefore the 100-task plan must be submitted in waves of at most eight total tasks. `%4` caps each wave at four running jobs.

M3 is CPU-only post-processing and starts after the M2 array completes successfully.
M1 combination is also CPU-only and starts after the corresponding M1-DET and
M1-TYPE runs complete successfully.

## Environment activation

The templates use the required activation order:

```bash
module purge
module load u22/python/3.12.4
source ~/envs/spell/bin/activate
```

Use `python`, not `python3`, after activation.

The shared `spell` environment must not be modified. DeBERTa-v3 tokenization
requires SentencePiece, which is supplied through a repository-local `.deps/`
overlay. Install it once with the CPU-only setup job:

```bash
sbatch slurm/setup_project_deps.sbatch
```

The job uses `pip --target "$POLAR_REPO/.deps"`; it does not install into
`~/envs/spell`. All supplied SLURM scripts prepend `.deps/` to `PYTHONPATH`.

## W&B authentication

W&B is disabled by default. Configurations and Ada job scripts target entity `manavberiwal006-iiit-hyderabad` and project `anlp-project`. Because Ada is being used through a shared Unix account, training jobs deliberately do not fall back to credentials created by `wandb login`.

The key pasted into chat must be revoked and replaced before any further run. Do not send the replacement through chat or place it in Git, YAML, a SLURM script, shell history, or the shared account's profile. At the start of each Termius session, read the replacement key without echoing it and export it only in that shell:

```bash
read -rsp "W&B API key: " WANDB_API_KEY
echo
export WANDB_API_KEY
```

Confirm only that the variable exists; never print its value:

```bash
test -n "${WANDB_API_KEY:-}" && echo "WANDB_API_KEY is set" || echo "WANDB_API_KEY is missing"
```

`slurm/submit_wave.sh` refuses to submit training without this variable. Slurm inherits it via `--export=ALL`, and each training task forces `WANDB_ENTITY=manavberiwal006-iiit-hyderabad` and `WANDB_PROJECT=anlp-project`. This prevents another user's shared CLI login from receiving the runs.

If compute nodes can access W&B, enable it through configuration overrides. If not, set `WANDB_MODE=offline`, write offline runs to persistent storage, and run `wandb sync` later on a network-enabled host.

## Failure recovery

Every array task writes to a deterministic run directory. Check the corresponding SLURM `.out` and `.err` files, correct the cause, and resubmit only missing task IDs. Do not delete successful runs merely to restart one failed task.

After W&B has initialized, transient metric, image, or finalization upload errors
are warnings rather than training failures. Local metrics, predictions, and plots
remain authoritative and can be synchronized later; W&B authentication or
initialization errors still fail early so runs are not silently sent elsewhere.

Before resuming a partially written run, move it to a clearly named archive or add explicit resume support. The current trainer intentionally starts cleanly and does not silently resume optimizer state.

## Output storage

Ada's `/share1` directories are unavailable on GPU compute nodes, while the
home directory is common to the login and compute nodes. Full-sweep outputs
therefore remain under `outputs/runs/` in the repository.

To stay inside the home quota, `training.keep_checkpoint` is `false` for the
100-run comparison. Each job holds its best FP32 model state in CPU memory,
restores it for outer-fold evaluation, then releases it. The persistent outputs
are the resolved configuration, environment metadata, logs, metrics, plots,
thresholds, and per-example probabilities/predictions. Optimizer state is never
retained. A deliberately selected run can later be repeated with
`--set training.keep_checkpoint=true` to save `checkpoints/best/`.

## Submission procedure

Generate run plans once and commit them with the code:

```bash
python scripts/generate_run_plan.py
```

On Ada, check current usage:

```bash
squeue -u "$USER"
```

Submit a wave from the repository root:

```bash
bash slurm/submit_wave.sh 0 8
```

After jobs finish, inspect both status and logs before submitting the next indices:

```bash
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,Timelimit
bash slurm/submit_wave.sh 8 8
```

Only the user should execute cluster submission or cancellation commands. Username and password should be entered directly into SSH and never placed in scripts or chat.

## Drag-and-drop deployment

When Git is unavailable, upload the complete `ANLP_Ugh` directory with Termius SFTP to `/home2/eashaan.thakur/manav/ANLP_Ugh`. Preserve the internal directory structure. After upload, run `chmod +x scripts/*.py scripts/*.sh slurm/*.sh slurm/*.sbatch` from the repository root.

Submit `slurm/setup_project_deps.sbatch` once, then `slurm/prepare_data.sbatch`.
Only after both complete successfully should `slurm/smoke_m2.sbatch` be
submitted. Full training waves remain blocked until the smoke job is reviewed.
