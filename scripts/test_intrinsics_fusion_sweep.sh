#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <scene_path> <models_dir> [test_ace.py args...]" >&2
  echo "Example: $0 datasets/Cambridge_GreatCourt outputs --session eval" >&2
  exit 1
fi

SCENE_PATH=$1
shift
MODELS_DIR=$1
shift

SESSION_PREFIX=${SESSION_PREFIX:-fusion_}

if [ ! -d "${MODELS_DIR}" ]; then
  echo "[test_intrinsics_fusion_sweep] Models directory '${MODELS_DIR}' does not exist." >&2
  exit 2
fi

shopt -s nullglob
MODEL_FILES=("${MODELS_DIR}"/*.pt)
shopt -u nullglob

if [ ${#MODEL_FILES[@]} -eq 0 ]; then
  echo "[test_intrinsics_fusion_sweep] No .pt models found in '${MODELS_DIR}'." >&2
  exit 3
fi

for MODEL_PATH in "${MODEL_FILES[@]}"; do
  MODEL_NAME=$(basename "${MODEL_PATH}")
  if [[ ${MODEL_NAME} =~ _([a-zA-Z0-9_]+)\.pt$ ]]; then
    MODE_SUFFIX=${BASH_REMATCH[1]}
  else
    MODE_SUFFIX="auto"
  fi
  SESSION_VALUE="${SESSION_PREFIX}${MODE_SUFFIX}"
  echo "[test_intrinsics_fusion_sweep] Evaluating ${MODEL_NAME} (session=${SESSION_VALUE})." >&2

  EXTRA_ARGS=()
  SESSION_SPECIFIED=false
  for ARG in "$@"; do
    if [[ ${ARG} == --session ]] || [[ ${ARG} == --session=* ]]; then
      SESSION_SPECIFIED=true
      break
    fi
  done
  if ! ${SESSION_SPECIFIED}; then
    EXTRA_ARGS=(--session "${SESSION_VALUE}")
  fi

  python test_ace.py "${SCENE_PATH}" "${MODEL_PATH}" "${EXTRA_ARGS[@]}" "$@"
  echo "[test_intrinsics_fusion_sweep] Finished ${MODEL_NAME}." >&2
  echo >&2
done

echo "[test_intrinsics_fusion_sweep] Evaluation complete for models in ${MODELS_DIR}." >&2
