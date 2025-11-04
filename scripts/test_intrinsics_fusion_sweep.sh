#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <scene_path> <models_dir> [test args...]"
  exit 1
fi

SCENE_PATH=$1
shift
MODELS_DIR=$1
shift

EVAL_ROOT="$(readlink -f ${MODELS_DIR})/eval"
mkdir -p "${EVAL_ROOT}"

MODES=(
  "mlp_add"
  "conv_add"
  "residual_concat"
  "film_residual"
  "star_block_conv"
  "att_channel"
  "att_spatial"
  "att_cross"
)

# Remove user session flags
CLEAN_ARGS=()
for ARG in "$@"; do
  [[ "$ARG" == --session* ]] && echo "[WARN] ignored $ARG" && continue
  CLEAN_ARGS+=("$ARG")
done

for MODE in "${MODES[@]}"; do
  MODEL=$(ls "${MODELS_DIR}"/*${MODE}.pt 2>/dev/null || true)
  [[ -z "$MODEL" ]] && echo "[SKIP] No model for $MODE" && continue

  MODEL=$(readlink -f "$MODEL")
  OUT_DIR="${EVAL_ROOT}/${MODE}"
  mkdir -p "${OUT_DIR}"

  LOG_FILE="${OUT_DIR}/eval.log"
  SESSION="fusion_${MODE}"

  echo "[TEST] $MODE"
  echo " model   = $MODEL"
  echo " session = $SESSION"
  echo " log     = $LOG_FILE"
  echo

  # ✅ Use absolute path for script & log, no cd issues
  python "$(readlink -f test_ace.py)" \
    "${SCENE_PATH}" \
    "${MODEL}" \
    --intrinsics_fusion "${MODE}" \
    --session "${SESSION}" \
    "${CLEAN_ARGS[@]}" \
    > "${LOG_FILE}" 2>&1
done

echo "✅ All fusion tests completed. Logs at: ${EVAL_ROOT}/<mode>/eval.log"
