#!/usr/bin/env bash
set -euo pipefail

MODEL_CONFIG="${1:-configs/m2.yaml}"
FOLD="${FOLD:-0}"
SEED="${SEED:-42}"

python scripts/train.py \
  --config "${MODEL_CONFIG}" \
  --fold "${FOLD}" \
  --seed "${SEED}" \
  --set training.epochs=1 \
  --set training.early_stopping_patience=1 \
  --set experiment.output_dir=outputs/smoke
