#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <scene_path> <output_dir> [train_ace.py args...]" >&2
  echo "Example: $0 datasets/Cambridge_GreatCourt outputs --epochs 8" >&2
  exit 1
fi

SCENE_PATH=$1
shift
OUTPUT_DIR=$1
shift

mkdir -p "${OUTPUT_DIR}"

MODES=("add" "concat_conv" "gated_add" "film")
SCENE_NAME=$(basename "${SCENE_PATH}")

for MODE in "${MODES[@]}"; do
  SAVE_PATH="${OUTPUT_DIR}/${SCENE_NAME}_${MODE}.pt"
  echo "[train_intrinsics_fusion_sweep] Training mode '${MODE}' -> ${SAVE_PATH}" >&2
  python train_ace.py "${SCENE_PATH}" "${SAVE_PATH}" --intrinsics_fusion "${MODE}" "$@"
  echo "[train_intrinsics_fusion_sweep] Finished mode '${MODE}'." >&2
  echo >&2
done

echo "[train_intrinsics_fusion_sweep] All modes completed. Models saved in ${OUTPUT_DIR}." >&2
