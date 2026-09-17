## L2O-Entropy

### Overview

"Learning to Generalize Provably in Learning to Optimize": the `L2O-Scale` codebase (`HierarchicalRNN`/`CoordinatewiseRNN` learned optimizers, see `Model_Free_L2O/L2O-Scale/`) plus Hessian/Jacobian-based flatness regularization of the learned optimizer's meta-training objective — the idea being that an optimizer trained to also prefer *flat* regions of the optimizee's loss landscape (low Hessian trace/eigenvalues/spectral density) generalizes better to unseen problems.

This directory unifies what used to be `L2O-ScalewHessian/L2O-Scale-Training/` and `L2O-ScalewHessian/L2O-Scale-Evaluation/` (now migrated to TF2/Keras 3 — see `migration_changes.txt`). Everything except the regularization pieces below is identical to (and copied verbatim from) the already-migrated `Model_Free_L2O/L2O-Scale/`.

### Code Overview

Same layout as `L2O-Scale/` (`optimizer/`, `problems/`, `metaopt.py`, `train.py`, `evaluate.py`), plus:

- `problems/problem_generator.py`'s `Problem` class gains `regularizer()` (dispatches on `reg_option`), and the underlying estimators it calls: `hessian_trace` (Hutchinson's trace estimator), `hessian_eigenvalues` (power iteration), `density` (Stochastic Lanczos Quadrature — eigenvalue spectral density), and `jacob_trace` (cheap mean-squared-gradient surrogate).
- `optimizer/trainable_optimizer.py`'s `TrainableOptimizer` gains `alpha`/`beta`/`reg_optimizer`/`reg_optimizee`/`reg_option`/`hessian_itrs` constructor args, and `partial_unroll()` gains a `jacob_switch` argument that gates whether the regularization term is accumulated for that unroll segment.
- `metaopt.py`'s `train_optimizer()` gains `regularize_time`/`reg_scale` args, which schedule `jacob_switch` per unroll segment (`'posterior'`: on late in the unroll, `'prior'`: on early, `'none'`: always off).

### Environment

- TensorFlow >= 2.10, Keras 3 (`pip install -r requirements.txt`)

### Train

Baseline (matching the paper's default config):

```shell
python train.py --train_dir=runs --optimizer=HierarchicalRNN \
    --include_mnist_mlp_problems --num_problems=1 --num_meta_iterations=100 \
    --fix_unroll --fix_unroll_length=20 --fix_num_steps=100 \
    --evaluation_period=1 --evaluation_epochs=5 --use_second_derivatives=False \
    --reg_optimizer=True --reg_option=hessian-esd
```

Regularization-specific flags: `--reg_option` (`hessian`/`hessian-ev`/`hessian-esd`/`jacob`), `--hessian_itrs`, `--alpha` (scale for `--reg_optimizer`), `--beta` (scale for `--reg_optimizee`), `--regularize_time` (`posterior`/`prior`/`none`), `--reg_scale`.

Checkpoints are written the same way as `L2O-Scale`'s `train.py` (`<train_dir>/<optimizer>_<cell_cls>_<cell_size>_<num_cells>/model-{best,iter<k>,final}.l2o`).

### Evaluate

```shell
python evaluate.py --train_dir=runs --optimizer=HierarchicalRNN \
    --save_dir=runs_eval --include_mnist_mlp_problems \
    --restore_model_name=model-final.l2o --num_testing_itrs=10000
```

Regularization is a meta-training-time-only concept (it doesn't affect `apply_gradients()`, the standalone drop-in-optimizer surface used here), so `evaluate.py` needs no regularization-specific flags.

### Scope

Ported: Hessian/Jacobian-based flatness regularization (the paper's actual contribution). Dropped (matching what was already dropped from `L2O-Scale`, both off-by-default extras in the original): model-transfer/knowledge-distillation (`mt_utils.py`/`if_mt`), curriculum learning (`if_cl`).

### Citation

Paper: "Learning to Generalize Provably in Learning to Optimize" (see the original `L2O-Entropy/README.md`'s title -- no BibTeX entry was ever filled in there, and none is fabricated here).
