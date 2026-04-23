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
#   --l2o_epochs           Training epochs for L2O-DM / L2O-RNNProp (default: 3000)
#   --l2o_enhanced_epochs  Training epochs for L2O-DM+ (default: 10000)
#   --skip_l2o             Skip all L2O model-free training/eval
#
# Example:
#   bash train_and_plot_figure6.sh --lam 0.005 --m 25 --n 50

set -e

# ---------- defaults ----------
LAM=0.005
M=25
N=50
NUM_LAYERS=16
BASE_DIR="."
L2O_EPOCHS=3000
L2O_ENHANCED_EPOCHS=10000
SKIP_L2O=false

# ---------- parse arguments ----------
while [[ $# -gt 0 ]]; do
  case $1 in
    --lam)                   LAM="$2";                   shift 2 ;;
    --m)                     M="$2";                     shift 2 ;;
    --n)                     N="$2";                     shift 2 ;;
    --num_layers)            NUM_LAYERS="$2";            shift 2 ;;
    --base_dir)              BASE_DIR="$2";              shift 2 ;;
    --l2o_epochs)            L2O_EPOCHS="$2";            shift 2 ;;
    --l2o_enhanced_epochs)   L2O_ENHANCED_EPOCHS="$2";   shift 2 ;;
    --skip_l2o)              SKIP_L2O=true;              shift ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_DIR="${BASE_DIR}/data/${M}_${N}"
COMMON="--task lasso --lasso_lam ${LAM} --base_dir ${BASE_DIR} \
  --data_dir ${DATA_DIR} --num_layers ${NUM_LAYERS} \
  --num_train_images 12800 --num_val_images 1280 --val_batch_size 1280"

# L2O directories
L2O_DIR="${SCRIPT_DIR}/../Model_Free_L2O/L2O-DM and L2O-RNNProp"

DM_MODEL_DIR="${L2O_DIR}/l2o_models/DM_lasso-${LAM}_m${M}_n${N}"
DM_ENH_MODEL_DIR="${L2O_DIR}/l2o_models/DM_enhanced_lasso-${LAM}_m${M}_n${N}"
RNNPROP_MODEL_DIR="${L2O_DIR}/l2o_models/RNNProp_lasso-${LAM}_m${M}_n${N}"

DM_EVAL_DIR="${L2O_DIR}/l2o_evals/DM_lasso-${LAM}_m${M}_n${N}"
DM_ENH_EVAL_DIR="${L2O_DIR}/l2o_evals/DM_enhanced_lasso-${LAM}_m${M}_n${N}"
RNNPROP_EVAL_DIR="${L2O_DIR}/l2o_evals/RNNProp_lasso-${LAM}_m${M}_n${N}"

DM_PICKLE="${DM_EVAL_DIR}/L2L_eval_loss_record.pickle-lasso_fixed_dm"
DM_ENH_PICKLE="${DM_ENH_EVAL_DIR}/L2L_eval_loss_record.pickle-lasso_fixed_dm"
RNNPROP_PICKLE="${RNNPROP_EVAL_DIR}/L2L_eval_loss_record.pickle-lasso_fixed_rnnprop"

echo "========================================"
echo " Training all Figure 6 models"
echo "  lam=${LAM}  m=${M}  n=${N}  layers=${NUM_LAYERS}"
echo "  data_dir=${DATA_DIR}"
echo "========================================"

# ---- model-based ----

echo "--- LISTA ---"
(cd "${SCRIPT_DIR}" && python train.py --model_name "lista" ${COMMON} \
  --exp_name "Lista_lasso-${LAM}_m${M}_n${N}")

echo "--- LFISTA ---"
(cd "${SCRIPT_DIR}" && python train.py --model_name "lfista" ${COMMON} \
  --exp_name "Lfista_lasso-${LAM}_m${M}_n${N}")

echo "--- ALISTA ---"
(cd "${SCRIPT_DIR}" && python train.py --model_name "alista" ${COMMON} \
  --exp_name "Alista_lasso-${LAM}_m${M}_n${N}")

# ---- model-free L2O ----

if [ "${SKIP_L2O}" = false ]; then

  echo ""
  echo "========================================"
  echo " Training L2O-DM (basic)"
  echo "========================================"
  (
    cd "${L2O_DIR}"
    python train_dm.py \
      --problem lasso_25_50 \
      --if_cl False --if_mt False \
      --num_steps 100 \
      --unroll_length 20 \
      --num_epochs "${L2O_EPOCHS}" \
      --learning_rate 0.001 \
      --save_path "${DM_MODEL_DIR}"
  )

  echo ""
  echo "========================================"
  echo " Evaluating L2O-DM on fixed test data"
  echo "========================================"
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

  echo ""
  echo "========================================"
  echo " Training L2O-DM+ (enhanced: CL + MT)"
  echo "========================================"
  (
    cd "${L2O_DIR}"
    python train_dm.py \
      --problem lasso_25_50 \
      --if_cl True --if_mt True \
      --num_mt 1 \
      --optimizers adam \
      --unroll_length 20 \
      --num_epochs "${L2O_ENHANCED_EPOCHS}" \
      --learning_rate 0.001 \
      --save_path "${DM_ENH_MODEL_DIR}"
  )

  echo ""
  echo "========================================"
  echo " Evaluating L2O-DM+ on fixed test data"
  echo "========================================"
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

  echo ""
  echo "========================================"
  echo " Training L2O-RNNProp"
  echo "========================================"
  (
    cd "${L2O_DIR}"
    python train_rnnprop.py \
      --problem lasso_25_50 \
      --if_cl False --if_mt False \
      --num_steps 100 \
      --unroll_length 20 \
      --num_epochs "${L2O_EPOCHS}" \
      --learning_rate 0.001 \
      --save_path "${RNNPROP_MODEL_DIR}"
  )

  echo ""
  echo "========================================"
  echo " Evaluating L2O-RNNProp on fixed test data"
  echo "========================================"
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

fi  # SKIP_L2O

# ---- Plot ----

echo ""
echo "========================================"
echo " Plotting Figure 6"
echo "========================================"

PLOT_ARGS="--lista --lfista --alista
  --lam ${LAM}
  --data_dir ${DATA_DIR}
  --base_dir ${BASE_DIR}
  --num_layers ${NUM_LAYERS}
  --fig_out ../Figs/figure6_m${M}_n${N}_lam${LAM}.png"

if [ "${SKIP_L2O}" = false ]; then
  PLOT_ARGS="${PLOT_ARGS}
  --l2o_dm_pickle ${DM_PICKLE}
  --l2o_dm_enhanced_pickle ${DM_ENH_PICKLE}
  --l2o_rnnprop_pickle ${RNNPROP_PICKLE}"
fi

(cd "${SCRIPT_DIR}" && eval python plot_figure6.py ${PLOT_ARGS})

echo ""
echo "Done. Figure saved to: ../Figs/figure6_m${M}_n${N}_lam${LAM}.png"
