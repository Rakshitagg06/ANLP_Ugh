#!/usr/bin/env bash
# Run all 100 training tasks from slurm/run_plans/training.tsv on the local GPU.
#
# Tasks run seed-major (all folds and models of seed 13, then 21, ...), so each
# completed seed immediately yields a full 3,222-example out-of-fold result.
# Completed runs (with a run_complete event) are skipped; a partial run
# directory is moved to outputs/failed_runs/ and retried once.
#
# Usage: bash scripts/run_local_sweep.sh
set -uo pipefail

cd "$(dirname "$0")/.."
source .secrets/keys.sh
export PYTHONPATH=".deps:src" TOKENIZERS_PARALLELISM=false WANDB_SILENT=true
export HF_HUB_DISABLE_PROGRESS_BARS=1

PLAN=slurm/run_plans/training.tsv
OUT=outputs/runs
LOGDIR=outputs/local_runner
mkdir -p "$OUT" "$LOGDIR" outputs/failed_runs

is_complete() {
  [[ -f "$1/logs/events.jsonl" ]] && grep -q '"event": "run_complete"' "$1/logs/events.jsonl"
}

# Seed-major ordering: sort by seed, then fold, then task id.
mapfile -t ROWS < <(tail -n +2 "$PLAN" | sort -t$'\t' -k5,5n -k4,4n -k1,1n)
TOTAL=${#ROWS[@]}
DONE=0
FAILED=()

for ROW in "${ROWS[@]}"; do
  IFS=$'\t' read -r TASK_ID MODEL CONFIG FOLD SEED <<< "$ROW"
  RUN_DIR="$OUT/${MODEL}-fold${FOLD}-seed${SEED}"
  DONE=$((DONE + 1))

  if is_complete "$RUN_DIR"; then
    echo "[$DONE/$TOTAL] skip task $TASK_ID ($RUN_DIR complete)"
    continue
  fi

  for ATTEMPT in 1 2; do
    if [[ -d "$RUN_DIR" ]]; then
      mv "$RUN_DIR" "outputs/failed_runs/$(basename "$RUN_DIR")-$(date +%Y%m%d%H%M%S)"
    fi
    echo "[$DONE/$TOTAL] $(date '+%F %T') task $TASK_ID model=$MODEL fold=$FOLD seed=$SEED attempt=$ATTEMPT"
    python3 scripts/train.py \
      --config "$CONFIG" --fold "$FOLD" --seed "$SEED" \
      --set "experiment.output_dir=$OUT" \
      --set logging.wandb.enabled=true \
      --set logging.wandb.entity=rakshitagg06-iiit-hyderabad \
      --set logging.wandb.project=anlp-project \
      --set logging.wandb.mode=online \
      > "$LOGDIR/task-${TASK_ID}.log" 2>&1
    if is_complete "$RUN_DIR"; then
      grep -o '"det_macro_f1": [0-9.]*\|"type_macro_f1": [0-9.]*\|"best_epoch": [0-9]*' \
        "$RUN_DIR/metrics/outer_fold_metrics.json" | tr '\n' ' '; echo
      break
    fi
    echo "  task $TASK_ID attempt $ATTEMPT failed; see $LOGDIR/task-${TASK_ID}.log"
    [[ $ATTEMPT == 2 ]] && FAILED+=("$TASK_ID")
  done
done

echo "Sweep finished $(date '+%F %T'). Failed tasks: ${FAILED[*]:-none}"
