## L2O-Jacobian (HALO)

### Overview

[HALO: Hardware-Aware Learning to Optimize](https://dl.acm.org/doi/abs/10.1007/978-3-030-58545-7_29) (ECCV 2020). Extends `L2O-Scale`/`L2O-Entropy`'s `HierarchicalRNN` learned optimizer with a **hardware-aware sparse-update mechanism**: the optimizer is meta-trained to converge even when only a random (or top-k) subset of parameter coordinates receives an RNN-driven update each step, mimicking a resource-constrained accelerator that can't afford dense per-step updates. Also includes an auxiliary convex regularizer (a synthetic convex sub-task riding piggyback on the real network parameters) and a pretrained-model warm-start mechanism for class-incremental adaptation evaluation.

This directory unifies what used to be `l2o-scale-regularize-train/` and `l2o-scale-regularize-test/` (now migrated to TF2/Keras 3 — see `migration_changes.txt`).

### Code Overview

Same layout as `L2O-Scale/`/`L2O-Entropy/` (`optimizer/`, `problems/`, `metaopt.py`, `train.py`, `evaluate.py`), plus:

- `optimizer/hierarchical_rnn.py`'s `HierarchicalRNN` gains `random_sparse_method`/`random_sparse_prob` (the sparse-update mechanism, `_compute_updates`) and `set_grad_flag_list()` (for coordinating with `problems/problem_generator.py`'s own gradient masking during adaptation evaluation).
- `optimizer/trainable_optimizer.py` gains `use_aux_convex_l1`/`use_aux_convex_l2`/`aux_convex_dim`/`convex_l1_ratio`/`aux_convex_ratio` (auxiliary convex regularization) and `pretrained_model_path` (warm-start).
- `problems/problem_generator.py`'s `Problem.gradients()` gains per-tensor gradient masking (independent of the optimizer's own mechanism above — this is what lets baseline optimizers like SGD/Adam also be evaluated under the same sparsity pattern); `ConvNet` gains the aux-convex extra parameter and pretrained-checkpoint loading.
- `problems/datasets.py` gains the adapt/pretrain class-incremental split (`convert_dataset`: odd digits/classes → "adapt", relabeled 0..4; even → "pretrain").
- `metaopt.py`'s `test_optimizer()` gains a warm-start + continue-training + periodically-test-accuracy epoch loop (`test_dataset`/`num_epochs`/`pretrained_model_path` args) on top of its existing flat drop-in-optimizer loop.
- `metaopt_wearable.py`/`evaluate_wearable.py` are separate, forked files (matching the original's own copy-paste fork) differing only in recording test accuracy at finer within-epoch intervals (`record_num`) rather than once per epoch.
- `train_pretrain.py` (new entry point): produces the pretrained-parameter pickles `--pretrained_model_path` consumes.

### Environment

- TensorFlow >= 2.10, Keras 3 (`pip install -r requirements.txt`)

### Step 1: pretrain

```shell
python train_pretrain.py --train_dir=pretrain --test_optimizer=Adam --lr=1e-3 \
    --num_testing_itrs=100 --include_mnist_conv_problems --custom_flag=mnist_conv
```

### Step 2: meta-train (baseline vs. HALO)

```shell
# baseline (no hardware-aware regularization)
python train.py --train_dir=runs --optimizer=HierarchicalRNN --adapt_mnist_conv_problems \
    --num_problems=1 --num_meta_iterations=100 --fix_unroll --fix_unroll_length=20 \
    --evaluation_period=1 --evaluation_epochs=5 --use_second_derivatives=False

# HALO (Jacobian regularization + sparse update)
python train.py --train_dir=runs --optimizer=HierarchicalRNN --adapt_mnist_conv_problems \
    --num_problems=1 --num_meta_iterations=100 --fix_unroll --fix_unroll_length=20 \
    --evaluation_period=1 --evaluation_epochs=5 --use_second_derivatives=False \
    --regularize_time=none --alpha=5e-4 --reg_optimizer=True --reg_option=jacob \
    --random_sparse_method=layer_wise --random_sparse_prob="0.1 0.3 0.5" \
    --pretrained_model_path=records/mnist_conv/Adam_lr_0.001_mnist_conv/seed0_model_params.pickle
```

### Step 3: adapt and evaluate

```shell
python evaluate.py --train_dir=runs --optimizer=HierarchicalRNN --save_dir=runs_eval \
    --adapt_mnist_conv_problems --test_optimizer=L2o --num_epochs=5 \
    --random_sparse_method=layer_wise --random_sparse_prob="0.1 0.3 0.5" \
    --restore_model_name=model-final.l2o
```

`--test_optimizer` also accepts `SGD`/`Adam`/`Adagrad` for a baseline comparison under the same sparsity pattern. `evaluate_wearable.py` is the same workflow with finer within-epoch accuracy tracking (`--record_num`).

### Scope

Ported: the full sparse-update mechanism (`layer_wise` mode; `params_wise`/`fix_num` raise `NotImplementedError` — see `migration_changes.txt`), auxiliary convex regularization, pretrained-model warm-start, and the `mnist`/`cifar10` adaptation problem families, on top of everything already migrated for `L2O-Scale`/`L2O-Entropy`.

Explicitly excluded (see `migration_changes.txt` for why): Jetson TX2 energy measurement, the PIE face dataset, the SML2010 sensor dataset, preresnet's external pruning-config pickles, the LSTM/wearable-sensor-benchmark wing (UCI-HAR/WISDM), and TDD (unsplit raw data only).

### Citation

Paper: "HALO: Hardware-Aware Learning to Optimize" (Li, Chen, You, Wang, Lin — ECCV 2020).
