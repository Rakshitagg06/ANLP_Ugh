#!/usr/bin/env bash
#SBATCH --job-name=polar-prepare
#SBATCH --partition=u22
#SBATCH --account=research
#SBATCH --qos=medium
#SBATCH --constraint=2080ti
#SBATCH --exclude=gnode066
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:0
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail

POLAR_REPO="${POLAR_REPO:-${SLURM_SUBMIT_DIR}}"
cd "${POLAR_REPO}"

module purge
module load u22/python/3.12.4
source "${HOME}/envs/spell/bin/activate"
export PYTHONPATH="${POLAR_REPO}/.deps:${POLAR_REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"

python scripts/create_folds.py \
  --config configs/create_folds.yaml \
  --output data/processed/english_train_folds.csv \
  --seed 2026

python scripts/validate_data.py \
  --config configs/m2.yaml \
  --output-dir outputs/data_audit

python -m compileall -q src scripts tests

if python -c 'import pytest' >/dev/null 2>&1; then
  python -m pytest -q
else
  printf 'pytest is not installed in the frozen environment; unit tests skipped.\n'
fi

printf 'Data preparation completed successfully.\n'
