#!/usr/bin/env bash
set -euo pipefail

SCENE=$1
OUT=$2
shift 2

MODES=(
  "residual_concat"
  "film_residual"
  "star_block_conv"
  "att_channel"
  "att_spatial"
  "att_cross"
)

GPUS=(0 1)
MAX_JOBS=${#GPUS[@]}

job_count=0

for MODE in "${MODES[@]}"; do
  GPU=${GPUS[$(( job_count % MAX_JOBS ))]}
  SAVE="${OUT}/$(basename $SCENE)_${MODE}.pt"
  LOG="${OUT}/${MODE}.log"

  echo "[QUEUE] GPU${GPU} → ${MODE}"

  (
    export CUDA_VISIBLE_DEVICES=$GPU
    python train_ace.py "$SCENE" "$SAVE" \
      --intrinsics_fusion "$MODE" "$@" \
      > "$LOG" 2>&1
  ) &

  job_count=$((job_count + 1))

  if (( job_count % MAX_JOBS == 0 )); then
    echo "[QUEUE] 等待当前 2 进程训练结束..."
    wait
  fi
done

wait
echo "[QUEUE] ✅ 全部训练完成"
