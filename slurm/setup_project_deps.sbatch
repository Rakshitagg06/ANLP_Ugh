#!/usr/bin/env bash
#SBATCH --job-name=polar-deps
#SBATCH --partition=u22
#SBATCH --account=research
#SBATCH --qos=medium
#SBATCH --constraint=2080ti
#SBATCH --exclude=gnode066
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:0
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail

POLAR_REPO="${POLAR_REPO:-${SLURM_SUBMIT_DIR}}"
cd "${POLAR_REPO}"
mkdir -p outputs/slurm .deps

# Keep the shared spell environment read-only. Missing packages are installed
# only in this repository's ignored .deps directory.
module purge
module load u22/python/3.12.4
source "${HOME}/envs/spell/bin/activate"

python -m pip install \
  --disable-pip-version-check \
  --no-input \
  --upgrade \
  --target "${POLAR_REPO}/.deps" \
  'sentencepiece>=0.2,<1'

export PYTHONPATH="${POLAR_REPO}/.deps:${POLAR_REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"
python -c "import sentencepiece; print('SentencePiece:', sentencepiece.__version__); print('Loaded from:', sentencepiece.__file__)"

printf 'Project-local dependencies installed successfully; spell was not modified.\n'
