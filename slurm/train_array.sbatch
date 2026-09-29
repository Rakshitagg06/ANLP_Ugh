#!/usr/bin/env bash
#SBATCH --job-name=polar-train
#SBATCH --partition=u22
#SBATCH --account=research
#SBATCH --qos=medium
#SBATCH --constraint=2080ti
#SBATCH --exclude=gnode066
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/slurm/%x-%A_%a.out
#SBATCH --error=outputs/slurm/%x-%A_%a.err

set -euo pipefail

: "${POLAR_REPO:?POLAR_REPO must point to the Ada checkout}"
: "${POLAR_RUN_PLAN:?POLAR_RUN_PLAN must point to training.tsv}"
: "${SLURM_ARRAY_TASK_ID:?This script must run as an array task}"

cd "${POLAR_REPO}"
mkdir -p outputs/slurm

ROW=$(awk -F '\t' -v id="${SLURM_ARRAY_TASK_ID}" 'NR > 1 && $1 == id { print; exit }' "${POLAR_RUN_PLAN}")
if [[ -z "${ROW}" ]]; then
  printf 'No run-plan row for task ID %s\n' "${SLURM_ARRAY_TASK_ID}" >&2
  exit 1
fi

IFS=$'\t' read -r TASK_ID MODEL CONFIG FOLD SEED <<< "${ROW}"
printf 'task=%s model=%s config=%s fold=%s seed=%s\n' \
  "${TASK_ID}" "${MODEL}" "${CONFIG}" "${FOLD}" "${SEED}"

POLAR_OUTPUT_ROOT="${POLAR_OUTPUT_ROOT:-${POLAR_REPO}/outputs/runs}"
mkdir -p "${POLAR_OUTPUT_ROOT}"
printf 'output_root=%s\n' "${POLAR_OUTPUT_ROOT}"

# Ada activation order from the supplied operational guide. Do not reorder.
module purge
module load u22/python/3.12.4
source "${HOME}/envs/spell/bin/activate"
export PYTHONPATH="${POLAR_REPO}/.deps:${POLAR_REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"

python -c "import sentencepiece; print('SentencePiece:', sentencepiece.__version__)"

# Ada is a shared Unix account, so never fall back to credentials created by
# another user's `wandb login`. The submitter must export the key in their own
# shell; Slurm inherits it through submit_wave.sh's --export=ALL.
: "${WANDB_API_KEY:?Export WANDB_API_KEY in the submitting shell before launching training jobs}"
export WANDB_API_KEY
export WANDB_ENTITY="manavberiwal006-iiit-hyderabad"
export WANDB_PROJECT="anlp-project"
printf 'W&B target: %s/%s (authentication from WANDB_API_KEY).\n' \
  "${WANDB_ENTITY}" "${WANDB_PROJECT}"

python scripts/train.py \
  --config "${CONFIG}" \
  --fold "${FOLD}" \
  --seed "${SEED}" \
  --set "experiment.output_dir=${POLAR_OUTPUT_ROOT}" \
  --set logging.wandb.enabled=true \
  --set "logging.wandb.entity=${WANDB_ENTITY}" \
  --set "logging.wandb.project=${WANDB_PROJECT}"
