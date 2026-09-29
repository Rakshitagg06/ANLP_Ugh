#!/usr/bin/env bash
#SBATCH --job-name=polar-m1-combine
#SBATCH --partition=u22
#SBATCH --account=research
#SBATCH --qos=medium
#SBATCH --constraint=2080ti
#SBATCH --exclude=gnode066
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=outputs/slurm/%x-%A_%a.out
#SBATCH --error=outputs/slurm/%x-%A_%a.err

set -euo pipefail

: "${POLAR_REPO:?POLAR_REPO must point to the Ada checkout}"
: "${POLAR_M1_RUN_PLAN:?POLAR_M1_RUN_PLAN must point to m1.tsv}"
: "${SLURM_ARRAY_TASK_ID:?This script must run as an array task}"

cd "${POLAR_REPO}"
mkdir -p outputs/slurm

ROW=$(awk -F '\t' -v id="${SLURM_ARRAY_TASK_ID}" 'NR > 1 && $1 == id { print; exit }' "${POLAR_M1_RUN_PLAN}")
if [[ -z "${ROW}" ]]; then
  printf 'No M1 run-plan row for task ID %s\n' "${SLURM_ARRAY_TASK_ID}" >&2
  exit 1
fi
IFS=$'\t' read -r TASK_ID FOLD SEED <<< "${ROW}"

POLAR_OUTPUT_ROOT="${POLAR_OUTPUT_ROOT:-${POLAR_REPO}/outputs/runs}"
DET_DIR="${POLAR_OUTPUT_ROOT}/m1_det-fold${FOLD}-seed${SEED}"
TYPE_DIR="${POLAR_OUTPUT_ROOT}/m1_type-fold${FOLD}-seed${SEED}"
M1_DIR="${POLAR_OUTPUT_ROOT}/m1-fold${FOLD}-seed${SEED}"
mkdir -p "${M1_DIR}/metrics" "${M1_DIR}/predictions"

module purge
module load u22/python/3.12.4
source "${HOME}/envs/spell/bin/activate"
export PYTHONPATH="${POLAR_REPO}/.deps:${POLAR_REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"

python scripts/combine_m1.py \
  --det-predictions "${DET_DIR}/predictions/outer_fold_predictions.jsonl" \
  --type-predictions "${TYPE_DIR}/predictions/outer_fold_predictions.jsonl" \
  --output "${M1_DIR}/predictions/outer_fold_predictions.jsonl" \
  --metrics-output "${M1_DIR}/metrics/outer_fold_metrics.json"

printf 'M1 combination completed for fold=%s seed=%s.\n' "${FOLD}" "${SEED}"
