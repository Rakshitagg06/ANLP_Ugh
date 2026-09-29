#!/usr/bin/env bash
set -euo pipefail

START="${1:?Usage: bash slurm/submit_wave.sh START_TASK_ID [COUNT]}"
REQUESTED_COUNT="${2:-8}"
PLAN="${POLAR_RUN_PLAN:-${PWD}/slurm/run_plans/training.tsv}"
POLAR_OUTPUT_ROOT="${PWD}/outputs/runs"
mkdir -p outputs/slurm
mkdir -p "${POLAR_OUTPUT_ROOT}"

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  printf 'WANDB_API_KEY is not set. Export it in this shell before submitting training jobs.\n' >&2
  exit 1
fi
# Ensure the key is inherited even if it was initially assigned without export.
export WANDB_API_KEY

if [[ ! -w "${POLAR_OUTPUT_ROOT}" ]]; then
  printf 'Output root is not writable: %s\n' "${POLAR_OUTPUT_ROOT}" >&2
  exit 1
fi

if [[ ! -f "${PLAN}" ]]; then
  printf 'Run plan not found: %s\nGenerate it with: python scripts/generate_run_plan.py\n' "${PLAN}" >&2
  exit 1
fi

EXISTING=$(squeue -h -u "${USER}" | wc -l | tr -d ' ')
AVAILABLE=$((8 - EXISTING))
if (( AVAILABLE <= 0 )); then
  printf 'No submission slots available: %d jobs are already submitted.\n' "${EXISTING}" >&2
  exit 1
fi

COUNT="${REQUESTED_COUNT}"
if (( COUNT > AVAILABLE )); then COUNT="${AVAILABLE}"; fi
if (( START + COUNT > 100 )); then COUNT=$((100 - START)); fi
if (( COUNT <= 0 )); then
  printf 'No training tasks remain from start index %d.\n' "${START}" >&2
  exit 1
fi

END=$((START + COUNT - 1))
sbatch \
  --array="${START}-${END}%4" \
  --export="ALL,POLAR_REPO=${PWD},POLAR_RUN_PLAN=${PLAN},POLAR_OUTPUT_ROOT=${POLAR_OUTPUT_ROOT}" \
  slurm/train_array.sbatch

printf 'Submitted training tasks %d-%d. Wait for slots before submitting the next wave.\n' "${START}" "${END}"
