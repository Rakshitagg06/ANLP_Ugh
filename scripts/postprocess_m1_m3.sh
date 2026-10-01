#!/usr/bin/env bash
# Build M1 (joined M1-DET + M1-TYPE) and both M3 variants (from M2) for all 25
# fold/seed pairs in slurm/run_plans/m1.tsv. Skips outputs that already exist.
#
# Usage: bash scripts/postprocess_m1_m3.sh
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=".deps:src${PYTHONPATH:+:${PYTHONPATH}}"
RUNS=outputs/runs

for TASK_ID in $(seq 0 24); do
  IFS=$'\t' read -r _ FOLD SEED <<< \
    "$(awk -F '\t' -v id="$TASK_ID" 'NR > 1 && $1 == id {print; exit}' slurm/run_plans/m1.tsv)"
  M1_DIR="$RUNS/m1-fold${FOLD}-seed${SEED}"
  M3_DIR="$RUNS/m2-fold${FOLD}-seed${SEED}/m3"

  if [[ ! -s "$M1_DIR/predictions/outer_fold_predictions.jsonl" ]]; then
    python3 scripts/combine_m1.py \
      --det-predictions "$RUNS/m1_det-fold${FOLD}-seed${SEED}/predictions/outer_fold_predictions.jsonl" \
      --type-predictions "$RUNS/m1_type-fold${FOLD}-seed${SEED}/predictions/outer_fold_predictions.jsonl" \
      --output "$M1_DIR/predictions/outer_fold_predictions.jsonl" \
      --metrics-output "$M1_DIR/metrics/outer_fold_metrics.json" > /dev/null
  fi

  if [[ ! -s "$M3_DIR/m3_symmetric_predictions.jsonl" ]]; then
    python3 scripts/reconcile_m3.py \
      --m2-predictions "$RUNS/m2-fold${FOLD}-seed${SEED}/predictions/outer_fold_predictions.jsonl" \
      --output-dir "$M3_DIR" > /dev/null
  fi
done

echo "M1: $(find "$RUNS" -maxdepth 1 -type d -name 'm1-fold*-seed*' | wc -l)/25," \
  "M3: $(find "$RUNS" -path '*/m3/m3_symmetric_predictions.jsonl' | wc -l)/25"
