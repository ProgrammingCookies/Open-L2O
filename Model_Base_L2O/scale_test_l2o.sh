#!/bin/bash
# GPU memory scaling test for model-free L2O models across different (m, n) problem sizes.
# Trains L2O-DM, L2O-DM+, and L2O-RNNProp for each (m, n) pair and records GPU memory
# usage via gpu_mem_sampler into profiles/training_profiles.jsonl.
#
# Usage:
#   bash scale_test_l2o.sh [options]
#
# Options:
#   --lam      LASSO lambda              (default: 0.005)
#   --epochs   Training epochs           (default: 100)
#   --mn_pairs Colon-separated M:N pairs (default: "5:10 25:50 50:100 100:200")
#
# Example:
#   bash scale_test_l2o.sh
#   bash scale_test_l2o.sh --mn_pairs "5:10 25:50"

set -e

# ---------- defaults ----------
LAM=0.005
EPOCHS=100
MN_PAIRS="5:10 25:50 50:100 100:200"

# ---------- parse arguments ----------
while [[ $# -gt 0 ]]; do
  case $1 in
    --lam)      LAM="$2";      shift 2 ;;
    --epochs)   EPOCHS="$2";   shift 2 ;;
    --mn_pairs) MN_PAIRS="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
L2O_DIR="${SCRIPT_DIR}/../Model_Free_L2O/L2O-DM and L2O-RNNProp"

# Tee all output to a timestamped log file
LOG_FILE="${SCRIPT_DIR}/scale_test_l2o_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG_FILE") 2>&1
echo "Logging to: ${LOG_FILE}"

echo "========================================"
echo " L2O GPU Memory Scaling Test"
echo "  lam=${LAM}  epochs=${EPOCHS}  iterations_per_epoch=1000"
echo "  pairs: ${MN_PAIRS}"
echo "========================================"

for MN in ${MN_PAIRS}; do
  M="${MN%%:*}"
  N="${MN##*:}"

  echo ""
  echo "========================================"
  echo " Problem: m=${M}  n=${N}"
  echo "========================================"

  # Generate data if needed
  DATA_DIR="${SCRIPT_DIR}/../data/${M}_${N}"
  echo "--- Generating data (skipped if already exists) ---"
  (cd "${SCRIPT_DIR}/.." && python generate_lasso_data.py \
    --m "${M}" \
    --n "${N}" \
    --output_dir "${DATA_DIR}")

  # Store scale-test models separately so they don't interfere with main runs
  DM_MODEL_DIR="${L2O_DIR}/l2o_models/scale_test/DM_lasso-${LAM}_m${M}_n${N}"
  DM_ENH_MODEL_DIR="${L2O_DIR}/l2o_models/scale_test/DM_enhanced_lasso-${LAM}_m${M}_n${N}"
  RNNPROP_MODEL_DIR="${L2O_DIR}/l2o_models/scale_test/RNNProp_lasso-${LAM}_m${M}_n${N}"

  echo "--- Training L2O-DM ---"
  (
    cd "${L2O_DIR}"
    python train_dm.py \
      --problem "lasso_${M}_${N}" \
      --if_cl False --if_mt False \
      --num_steps 1000 \
      --unroll_length 20 \
      --num_epochs "${EPOCHS}" \
      --learning_rate 0.001 \
      --save_path "${DM_MODEL_DIR}" \
      --profile_dir "${SCRIPT_DIR}/profiles"
  )

  echo "--- Training L2O-DM+ (enhanced) ---"
  (
    cd "${L2O_DIR}"
    python train_dm.py \
      --problem "lasso_${M}_${N}" \
      --if_cl True --if_mt True \
      --num_mt 1 \
      --optimizers adam \
      --unroll_length 20 \
      --num_epochs "${EPOCHS}" \
      --learning_rate 0.001 \
      --save_path "${DM_ENH_MODEL_DIR}" \
      --profile_dir "${SCRIPT_DIR}/profiles"
  )

  echo "--- Training L2O-RNNProp ---"
  (
    cd "${L2O_DIR}"
    python train_rnnprop.py \
      --problem "lasso_${M}_${N}" \
      --if_cl False --if_mt False \
      --num_steps 1000 \
      --unroll_length 20 \
      --num_epochs "${EPOCHS}" \
      --learning_rate 0.001 \
      --save_path "${RNNPROP_MODEL_DIR}" \
      --profile_dir "${SCRIPT_DIR}/profiles"
  )

done

echo ""
echo "========================================"
echo " Scaling comparison table"
echo "========================================"
(cd "${SCRIPT_DIR}" && python compare_scaling.py --base_dir ".")

echo ""
echo "Done. Full results in: ${SCRIPT_DIR}/profiles/training_profiles.jsonl"
