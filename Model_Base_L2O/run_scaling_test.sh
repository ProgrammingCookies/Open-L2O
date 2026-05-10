#!/bin/bash
# Run train_and_plot_figure6.sh repeatedly across a range of layer counts to
# collect scaling data (training time, GPU memory) for LISTA, LFISTA, ALISTA.
#
# Each run appends to profiles/training_profiles.jsonl and the final run prints
# the full comparison table.
#
# Usage:
#   bash run_scaling_test.sh [options]
#
# Options:
#   --layers     Space-separated list of layer counts to test  (default: "2 4 8 16")
#   --lam        LASSO lambda                                   (default: 0.005)
#   --m          Measurement dimension                          (default: 25)
#   --n          Signal dimension                               (default: 50)
#   --noise_std  Noise std (0.0 = clean data)                  (default: 0.0)
#   --base_dir   Base dir passed through to training script     (default: .)
#
# Example:
#   bash run_scaling_test.sh
#   bash run_scaling_test.sh --layers "2 4 8 16" --noise_std 0.1

set -e

# ---------- defaults ----------
LAYERS="2 4 8 16"
LAM=0.005
M=25
N=50
NOISE_STD=0.0
BASE_DIR="."

# ---------- parse arguments ----------
while [[ $# -gt 0 ]]; do
  case $1 in
    --layers)    LAYERS="$2";    shift 2 ;;
    --lam)       LAM="$2";       shift 2 ;;
    --m)         M="$2";         shift 2 ;;
    --n)         N="$2";         shift 2 ;;
    --noise_std) NOISE_STD="$2"; shift 2 ;;
    --base_dir)  BASE_DIR="$2";  shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================"
echo " Scaling test"
echo "  layers:    ${LAYERS}"
echo "  lam=${LAM}  m=${M}  n=${N}  noise_std=${NOISE_STD}"
echo "========================================"

for L in ${LAYERS}; do
  echo ""
  echo "========================================"
  echo " Running num_layers=${L}"
  echo "========================================"
  bash "${SCRIPT_DIR}/train_and_plot_figure6.sh" \
    --lam        "${LAM}" \
    --m          "${M}" \
    --n          "${N}" \
    --num_layers "${L}" \
    --noise_std  "${NOISE_STD}" \
    --base_dir   "${BASE_DIR}" \
    --SKIP_L2O_MODEL_FREE
done

echo ""
echo "========================================"
echo " All runs complete. Final scaling table:"
echo "========================================"
(cd "${SCRIPT_DIR}" && python compare_scaling.py --base_dir "${BASE_DIR}")
