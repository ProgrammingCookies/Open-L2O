#!/bin/bash
# Train all Figure 6 models for the LASSO task, then plot Figure 6.
# Model-based  : LISTA, LFISTA, ALISTA
# Model-free   : L2O-DM, L2O-DM+ (enhanced: CL+MT), L2O-RNNProp
#
# Usage:
#   bash train_and_plot_figure6.sh [options]
#
# Options:
#   --lam                  LASSO lambda            (default: 0.005)
#   --m                    Measurement dimension   (default: 25)
#   --n                    Signal dimension        (default: 50)
#   --num_layers           Unrolled layers         (default: 16)
#   --base_dir             Base dir for model-based models (default: .)
#   --l2o_epochs           Training epochs for all L2O models (default: 100)
#   --noise_std            Gaussian noise std added to measurements (default: 0.0)
#                          When > 0 a clean copy of each split is also saved alongside.
#   --SKIP_L2O_MODEL_FREE             Skip all L2O model-free training/eval
#
# Example:
#   bash train_and_plot_figure6.sh --lam 0.005 --m 25 --n 50
#   bash train_and_plot_figure6.sh --lam 0.005 --m 25 --n 50 --noise_std 0.1

set -e

# ---------- defaults ----------
LAM=0.005
M=25
N=50
NUM_LAYERS=16
BASE_DIR="."
L2O_EPOCHS=100
NOISE_STD=0.0
SKIP_L2O_MODEL_FREE=false

# ---------- parse arguments ----------
while [[ $# -gt 0 ]]; do
  case $1 in
    --lam)                   LAM="$2";                   shift 2 ;;
    --m)                     M="$2";                     shift 2 ;;
    --n)                     N="$2";                     shift 2 ;;
    --num_layers)            NUM_LAYERS="$2";            shift 2 ;;
    --base_dir)              BASE_DIR="$2";              shift 2 ;;
    --l2o_epochs)            L2O_EPOCHS="$2";            shift 2 ;;
    --noise_std)             NOISE_STD="$2";             shift 2 ;;
    --SKIP_L2O_MODEL_FREE)              SKIP_L2O_MODEL_FREE=true;              shift ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Tee all output (stdout + stderr) to a timestamped log file
LOG_FILE="${SCRIPT_DIR}/train_figure6_m${M}_n${N}_lam${LAM}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG_FILE") 2>&1
echo "Logging to: ${LOG_FILE}"

# Noise-aware data and model directory suffixes
if awk "BEGIN { exit ($NOISE_STD > 0 ? 0 : 1) }"; then
  NOISE_SUFFIX="_noise${NOISE_STD}"
else
  NOISE_SUFFIX=""
fi

DATA_DIR="${SCRIPT_DIR}/../data/${M}_${N}${NOISE_SUFFIX}"
COMMON="--task lasso --lasso_lam ${LAM} --base_dir ${BASE_DIR} \
  --data_dir ${DATA_DIR} --num_layers ${NUM_LAYERS} \
  --noise_std ${NOISE_STD} \
  --num_train_images 12800 --num_val_images 1280 --val_batch_size 1280"

# L2O directories
L2O_DIR="${SCRIPT_DIR}/../Model_Free_L2O/L2O-DM and L2O-RNNProp"

DM_MODEL_DIR="${L2O_DIR}/l2o_models/DM_lasso-${LAM}_m${M}_n${N}${NOISE_SUFFIX}"
DM_ENH_MODEL_DIR="${L2O_DIR}/l2o_models/DM_enhanced_lasso-${LAM}_m${M}_n${N}${NOISE_SUFFIX}"
RNNPROP_MODEL_DIR="${L2O_DIR}/l2o_models/RNNProp_lasso-${LAM}_m${M}_n${N}${NOISE_SUFFIX}"

DM_EVAL_DIR="${L2O_DIR}/l2o_evals/DM_lasso-${LAM}_m${M}_n${N}${NOISE_SUFFIX}"
DM_ENH_EVAL_DIR="${L2O_DIR}/l2o_evals/DM_enhanced_lasso-${LAM}_m${M}_n${N}${NOISE_SUFFIX}"
RNNPROP_EVAL_DIR="${L2O_DIR}/l2o_evals/RNNProp_lasso-${LAM}_m${M}_n${N}${NOISE_SUFFIX}"

DM_PICKLE="${DM_EVAL_DIR}/L2L_eval_loss_record.pickle-lasso_fixed_dm"
DM_ENH_PICKLE="${DM_ENH_EVAL_DIR}/L2L_eval_loss_record.pickle-lasso_fixed_dm"
RNNPROP_PICKLE="${RNNPROP_EVAL_DIR}/L2L_eval_loss_record.pickle-lasso_fixed_rnnprop"

echo "========================================"
echo " Generating LASSO data"
echo "  m=${M}  n=${N}  noise_std=${NOISE_STD}"
echo "  output_dir=${DATA_DIR}"
echo "========================================"
(cd "${SCRIPT_DIR}/.." && python generate_lasso_data.py \
  --m "${M}" \
  --n "${N}" \
  --noise_std "${NOISE_STD}" \
  --output_dir "${DATA_DIR}")

echo "========================================"
echo " Training all Figure 6 models"
echo "  lam=${LAM}  m=${M}  n=${N}  layers=${NUM_LAYERS}"
echo "  data_dir=${DATA_DIR}"
echo "========================================"

# ---- model-based ----

echo "--- LISTA ---"
(cd "${SCRIPT_DIR}" && python train.py --model_name "lista" ${COMMON} \
  --exp_name "Lista_lasso-${LAM}_m${M}_n${N}${NOISE_SUFFIX}_L${NUM_LAYERS}")

echo "--- LFISTA ---"
(cd "${SCRIPT_DIR}" && python train.py --model_name "lfista" ${COMMON} \
  --exp_name "Lfista_lasso-${LAM}_m${M}_n${N}${NOISE_SUFFIX}_L${NUM_LAYERS}")

echo "--- ALISTA ---"
(cd "${SCRIPT_DIR}" && python train.py --model_name "alista" ${COMMON} \
  --exp_name "Alista_lasso-${LAM}_m${M}_n${N}${NOISE_SUFFIX}_L${NUM_LAYERS}")

# ---- model-free L2O ----

if [ "${SKIP_L2O_MODEL_FREE}" = false ]; then

  echo ""
  echo "========================================"
  echo " Training L2O-DM (basic)"
  echo "========================================"
  if [ -f "${DM_MODEL_DIR}/cw.l2l-0" ]; then
    echo "Skipping — model already exists at ${DM_MODEL_DIR}"
  else
    (
      cd "${L2O_DIR}"
      python train_dm.py \
        --problem "lasso_${M}_${N}" \
        --if_cl False --if_mt False \
        --num_steps 1000 \
        --unroll_length 20 \
        --num_epochs "${L2O_EPOCHS}" \
        --learning_rate 0.001 \
        --save_path "${DM_MODEL_DIR}" \
        --profile_dir "${SCRIPT_DIR}/profiles"
    )
  fi

  echo ""
  echo "========================================"
  echo " Evaluating L2O-DM on fixed test data"
  echo "========================================"
  if [ -f "${DM_PICKLE}" ]; then
    echo "Skipping — pickle already exists at ${DM_PICKLE}"
  else
    (
      cd "${L2O_DIR}"
      python evaluate_lasso_fixed.py \
        --model_type dm \
        --path "${DM_MODEL_DIR}" \
        --data_dir "${DATA_DIR}" \
        --lam "${LAM}" \
        --num_steps "${NUM_LAYERS}" \
        --output_path "${DM_EVAL_DIR}"
    )
  fi

  echo ""
  echo "========================================"
  echo " Training L2O-DM+ (enhanced: CL + MT)"
  echo "========================================"
  if [ -f "${DM_ENH_MODEL_DIR}/cw.l2l-0" ]; then
    echo "Skipping — model already exists at ${DM_ENH_MODEL_DIR}"
  else
    (
      cd "${L2O_DIR}"
      python train_dm.py \
        --problem "lasso_${M}_${N}" \
        --if_cl True --if_mt True \
        --num_mt 1 \
        --optimizers adam \
        --unroll_length 20 \
        --num_epochs "${L2O_EPOCHS}" \
        --learning_rate 0.001 \
        --save_path "${DM_ENH_MODEL_DIR}" \
        --profile_dir "${SCRIPT_DIR}/profiles"
    )
  fi

  echo ""
  echo "========================================"
  echo " Evaluating L2O-DM+ on fixed test data"
  echo "========================================"
  if [ -f "${DM_ENH_PICKLE}" ]; then
    echo "Skipping — pickle already exists at ${DM_ENH_PICKLE}"
  else
    (
      cd "${L2O_DIR}"
      python evaluate_lasso_fixed.py \
        --model_type dm \
        --path "${DM_ENH_MODEL_DIR}" \
        --data_dir "${DATA_DIR}" \
        --lam "${LAM}" \
        --num_steps "${NUM_LAYERS}" \
        --output_path "${DM_ENH_EVAL_DIR}"
    )
  fi

  echo ""
  echo "========================================"
  echo " Training L2O-RNNProp"
  echo "========================================"
  if [ -f "${RNNPROP_MODEL_DIR}/rp.l2l-0" ]; then
    echo "Skipping — model already exists at ${RNNPROP_MODEL_DIR}"
  else
    (
      cd "${L2O_DIR}"
      python train_rnnprop.py \
        --problem "lasso_${M}_${N}" \
        --if_cl False --if_mt False \
        --num_steps 1000 \
        --unroll_length 20 \
        --num_epochs "${L2O_EPOCHS}" \
        --learning_rate 0.001 \
        --save_path "${RNNPROP_MODEL_DIR}" \
        --profile_dir "${SCRIPT_DIR}/profiles"
    )
  fi

  echo ""
  echo "========================================"
  echo " Evaluating L2O-RNNProp on fixed test data"
  echo "========================================"
  if [ -f "${RNNPROP_PICKLE}" ]; then
    echo "Skipping — pickle already exists at ${RNNPROP_PICKLE}"
  else
    (
      cd "${L2O_DIR}"
      python evaluate_lasso_fixed.py \
        --model_type rnnprop \
        --path "${RNNPROP_MODEL_DIR}" \
        --data_dir "${DATA_DIR}" \
        --lam "${LAM}" \
        --num_steps "${NUM_LAYERS}" \
        --output_path "${RNNPROP_EVAL_DIR}"
    )
  fi

fi  # SKIP_L2O_MODEL_FREE

# ---- Plot ----

echo ""
echo "========================================"
echo " Plotting Figure 6"
echo "========================================"

PLOT_ARGS=(--lista --lfista --alista
  --lam "${LAM}"
  --data_dir "${DATA_DIR}"
  --base_dir "${BASE_DIR}"
  --num_layers "${NUM_LAYERS}"
  --noise_std "${NOISE_STD}"
  --fig_out "../Figs/figure6_m${M}_n${N}_lam${LAM}${NOISE_SUFFIX}.png")

if [ "${SKIP_L2O_MODEL_FREE}" = false ]; then
  PLOT_ARGS+=(
    --l2o_dm_pickle "${DM_PICKLE}"
    --l2o_dm_enhanced_pickle "${DM_ENH_PICKLE}"
    --l2o_rnnprop_pickle "${RNNPROP_PICKLE}")
fi

(cd "${SCRIPT_DIR}" && python plot_figure6.py "${PLOT_ARGS[@]}")

echo ""
echo "========================================"
echo " Scaling comparison table"
echo "========================================"
(cd "${SCRIPT_DIR}" && python compare_scaling.py --base_dir "${BASE_DIR}")

echo ""
echo "Done. Figure saved to: ../Figs/figure6_m${M}_n${N}_lam${LAM}${NOISE_SUFFIX}.png"
