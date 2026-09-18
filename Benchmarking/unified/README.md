# `Benchmarking/unified` — one implementation for the Experiment-2 methods

The adapters in `adapters/` wrap five libraries that each ship their own
`problems.py` / `networks.py` / `meta.py` and cannot even share an interpreter.
This package is the implementation-layer counterpart for the methods
Experiment 2 uses: **one** LASSO instance loader, **one** objective / gradient
oracle, **one** meta-training loop, **one** iterative-solver interface and
**one** evaluation walk, behind the same `core.method.L2OMethod` ABC and the
same config vocabulary as the adapters.

| method | backs | implementation |
|---|---|---|
| `unified-dm` | `l2o-dm` | `policies.DMPolicy` + `meta_train` |
| `unified-rnnprop` | `l2o-rnnprop` | `policies.RNNPropPolicy` + `meta_train` |
| `unified-lista`, `unified-alista` | `lista`, `alista` | `unrolled.UnrolledSolver` + `curriculum` |

Registration is opt-in (`import unified`); nothing in `adapters/` changes and
the adapter-backed names are never shadowed. TensorFlow is imported lazily
inside `train()` / `evaluate()`.

## Layout

| module | role | provenance |
|---|---|---|
| `instances.py` | `LassoInstances`: A / split / x0 / W loader, deterministic `walk()` (eval) and global-RNG `sample()` (train) | merges DM `problems.lasso_from_dataset`, Scale `pg.Lasso`, Model_Base `data_preprocessing.input_fn` |
| `objective.py` | `LassoObjective`: batch-mean / per-sample value, gradient oracle | merges the three libraries' identical formula |
| `solver.py` | `IterativeSolver.iterate(batch, T) -> x_0..x_T`; `PolicySolver` (model-free) | new |
| `nets_dm.py` | `CoordinateWiseDeepLSTM`, `RNNprop`, `LogAndSign`, index-keyed pickle save/load | **verbatim** from `L2O-DM and L2O-RNNProp/networks.py`, `preprocess.py` |
| `policies.py` | `OptimizerPolicy.step(x, grad, state, t)`: `DMPolicy`, `RNNPropPolicy`, `ClassicalAdamPolicy` | ported from the four `meta*.py` copies |
| `meta_train.py` | `meta_train(policy, objective, instances, MetaTrainSpec, save_path)` | ported from `train_{dm,rnnprop}.py` + `meta_*_train.py::unroll` |
| `unrolled.py` | `Lista`/`Alista` cells + models, `UnrolledSolver`, `load_layerwise_checkpoint` | **verbatim** from `Model_Base_L2O/models/{lista,alista,utils}.py` + new loader |
| `curriculum.py` | layer-wise 3-stage `fit()` curriculum, `LassoLoss`/`LassoObjective`/`Adam` | ported from `Model_Base_L2O/train.py`, `utils.py`, `data_preprocessing.py` |
| `evalwalk.py` | one split walk -> trajectory + fingerprint + Table-4 metrics + adapter-layout artifacts | merges `evaluate_dm.py`, `L2O-Scale/evaluate.py`, `adapters/model_based.py` |
| `methods.py` | the `L2OMethod` subclasses (config vocabulary = the adapters') | new |
| `exp2_driver.py` | Experiment 2 through the unified methods (`python -m unified.exp2_driver`) | fork of `experiments/v2/experiment2.py::run_cell` |

Verbatim copies are verbatim on purpose: Keras creates variables in
attribute-assignment order, and the checkpoint formats are that order, so the
copied classes load the adapters' checkpoints by construction.

## Parity (what "the same implementation" means here)

`tests/test_unified_parity.py`, `tests/test_unified_train.py`:

* **Evaluation**, real Experiment-2 checkpoints: RNNProp (try3) and DM (try2)
  reproduce the stored trajectories (max |diff| 1e-5 / 4e-7 over 1000 steps);
  LISTA / ALISTA (try3) reproduce `final_output.npy` and the per-layer values
  for all 11 test files. On the same machine the RNNProp evaluation is
  **bit-identical** to `evaluate_rnnprop.py` (0.0 over 1000 steps x 1280
  instances); the remaining difference to the stored GPU files is the scripts'
  own GPU/CPU float32 drift.
* **Training**: a same-seed CPU run of `unified-dm` / `unified-rnnprop` (with
  and without `last_step_loss`) is **bit-identical** to `train_dm.py` /
  `train_rnnprop.py` -- every saved tensor, `recovery.npz`, every logged loss.
  Unified-trained LISTA / ALISTA checkpoints are read by `Model_Base_L2O/train.py
  --test` (through the adapter) with identical results; the curriculum itself
  is not bit-reproducible in either implementation (`tf.data`
  `experimental_deterministic=False`).
* **Fairness**: `tests/fairness_matrix.py::lasso_tiny_unified` proves the
  unified methods score the very same instances (`b + x0 + x_true`
  fingerprint) as the adapter-backed reference.

## Faithful, not fixed

Reproduced for parity, documented here, switchable only explicitly:

* `curriculum.LassoLoss` uses `tf.nn.l2_loss` (= sum/2): LISTA/ALISTA are
  trained on `0.25*||Ax-b||^2 + lam*||x||_1` (batch sum) while everything is
  scored on `0.5*||Ax-b||^2 + lam*||x||_1`. `keras_train_loss="paper"`
  (default) keeps that; `"scoring"` trains on the scored objective.
* The trajectory is `[f(x_0) .. f(x_{T-1})]` (the scripts record the objective
  before each step / the metric at each layer's input) while `x_pred = x_T`.
* Model-free chunk averaging is a plain mean over `ceil(N/batch)` chunks (the
  last, smaller chunk weighs the same as the others), as in the scripts.

Deliberate deviations (stricter than the scripts): a missing checkpoint
**raises** instead of silently evaluating an untrained optimizer; unknown
config keys raise; the unported training flags (`if_cl`, `if_mt`, `if_scale`,
`fix_num_unrolls`, `track_training_curve`) raise `NotImplementedError`;
`profile_path` is ignored with a warning.

## What the unification found

Checking the LISTA parity against the stored try3 evaluation exposed that
`Model_Base_L2O/train.py`'s per-layer `lasso_layer{i}` values were running
means across test files (see `RESULTS_VALIDITY.md`, 2026-09-17). The unified
evaluator is what `regenerate_modelbased_trajectories.py` uses to recompute
them from the existing checkpoints without retraining.

## Not in the slice

L2O-Scale (`unified-scale`): its `optimizer/*.py` are byte-identical across
Scale/Entropy and would be copied verbatim under `unified/hrnn/`; its
evaluation surface has RNG-consuming quirks (dummy `apply_gradients`, a dead
`tf.random.normal` per step) that bit-exact parity would have to replicate.
Swarm / Entropy / Jacobian stay adapter-only.
