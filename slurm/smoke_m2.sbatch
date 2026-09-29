#!/usr/bin/env bash
#SBATCH --job-name=polar-smoke-m2
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
#SBATCH --time=01:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail

POLAR_REPO="${POLAR_REPO:-${SLURM_SUBMIT_DIR}}"
cd "${POLAR_REPO}"

module purge
module load u22/python/3.12.4
source "${HOME}/envs/spell/bin/activate"
export PYTHONPATH="${POLAR_REPO}/.deps:${POLAR_REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"

python -c "import sentencepiece; print('SentencePiece:', sentencepiece.__version__)"

: "${WANDB_API_KEY:?Export WANDB_API_KEY in the submitting shell before launching the smoke job}"
export WANDB_API_KEY
export WANDB_ENTITY="manavberiwal006-iiit-hyderabad"
export WANDB_PROJECT="anlp-project"
printf 'W&B target: %s/%s (authentication from WANDB_API_KEY).\n' \
  "${WANDB_ENTITY}" "${WANDB_PROJECT}"

python scripts/train.py \
  --config configs/m2.yaml \
  --fold 0 \
  --seed 42 \
  --set training.epochs=1 \
  --set training.early_stopping_patience=1 \
  --set "experiment.output_dir=outputs/smoke/${SLURM_JOB_ID}" \
  --set logging.wandb.enabled=true \
  --set "logging.wandb.entity=${WANDB_ENTITY}" \
  --set "logging.wandb.project=${WANDB_PROJECT}" \
  --set logging.wandb.tags='[smoke,m2]'

printf 'M2 smoke run completed successfully.\n'
