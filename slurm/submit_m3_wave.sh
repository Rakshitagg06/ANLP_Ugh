#!/usr/bin/env bash
set -euo pipefail

START="${1:?Usage: bash slurm/submit_m3_wave.sh START_TASK_ID [COUNT]}"
REQUESTED_COUNT="${2:-8}"
PLAN="${POLAR_M3_RUN_PLAN:-${PWD}/slurm/run_plans/m3.tsv}"
POLAR_OUTPUT_ROOT="${PWD}/outputs/runs"
mkdir -p outputs/slurm

if [[ ! -d "${POLAR_OUTPUT_ROOT}" || ! -w "${POLAR_OUTPUT_ROOT}" ]]; then
  printf 'Output root is missing or not writable: %s\n' "${POLAR_OUTPUT_ROOT}" >&2
  exit 1
fi

if [[ ! -f "${PLAN}" ]]; then
  printf 'M3 run plan not found: %s\n' "${PLAN}" >&2
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
if (( START + COUNT > 25 )); then COUNT=$((25 - START)); fi
if (( COUNT <= 0 )); then
  printf 'No M3 tasks remain from start index %d.\n' "${START}" >&2
  exit 1
fi

END=$((START + COUNT - 1))
sbatch \
  --array="${START}-${END}%4" \
  --export="ALL,POLAR_REPO=${PWD},POLAR_M3_RUN_PLAN=${PLAN},POLAR_OUTPUT_ROOT=${POLAR_OUTPUT_ROOT}" \
  slurm/m3_postprocess.sbatch

printf 'Submitted M3 tasks %d-%d.\n' "${START}" "${END}"
