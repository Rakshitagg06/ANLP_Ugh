#!/usr/bin/env bash
#SBATCH --job-name=polar-m3
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
: "${POLAR_M3_RUN_PLAN:?POLAR_M3_RUN_PLAN must point to m3.tsv}"
: "${SLURM_ARRAY_TASK_ID:?This script must run as an array task}"

cd "${POLAR_REPO}"
mkdir -p outputs/slurm

ROW=$(awk -F '\t' -v id="${SLURM_ARRAY_TASK_ID}" 'NR > 1 && $1 == id { print; exit }' "${POLAR_M3_RUN_PLAN}")
if [[ -z "${ROW}" ]]; then
  printf 'No M3 run-plan row for task ID %s\n' "${SLURM_ARRAY_TASK_ID}" >&2
  exit 1
fi
IFS=$'\t' read -r TASK_ID FOLD SEED <<< "${ROW}"
POLAR_OUTPUT_ROOT="${POLAR_OUTPUT_ROOT:-${POLAR_REPO}/outputs/runs}"
RUN_DIR="${POLAR_OUTPUT_ROOT}/m2-fold${FOLD}-seed${SEED}"

module purge
module load u22/python/3.12.4
source "${HOME}/envs/spell/bin/activate"
export PYTHONPATH="${POLAR_REPO}/.deps:${POLAR_REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"

python scripts/reconcile_m3.py \
  --m2-predictions "${RUN_DIR}/predictions/outer_fold_predictions.jsonl" \
  --output-dir "${RUN_DIR}/m3"
