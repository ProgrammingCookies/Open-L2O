# `Benchmarking/unified` — one implementation of L2O methods, extensible

The adapters in `adapters/` wrap five libraries that each ship their own
`problems.py` / `networks.py` / `meta.py` and cannot even share an interpreter.
This package is the implementation-layer counterpart: **one** problem protocol,
**one** learned-optimizer protocol, **one** meta-training loop, **one**
iterative-solver interface and **one** evaluation walk, behind the same
`core.method.L2OMethod` ABC and the same config vocabulary as the adapters —
so the sweep template, the fairness tests and the experiment scripts work
unchanged, and a new method or problem is one module plus one line.

| registered method | backs | implementation |
|---|---|---|
| `unified-dm` | `l2o-dm` | `learned/dm.py` (`DMPolicy` + `DM_SPEC`) |
| `unified-rnnprop` | `l2o-rnnprop` | `learned/rnnprop.py` |
| `unified-lista`, `unified-alista` | `lista`, `alista` | `unrolled/` (`LISTA_SPEC`, `ALISTA_SPEC`) |

| registered problem (`config["problem"]`) | kind | implementation |
|---|---|---|
| `lasso_dataset` (model-free vocabulary) and `lasso` (model-based vocabulary) | LASSO on `data/lasso.py` datasets | `tasks/lasso.py` |
| `quadratic` | L2O-DM's library-generated `‖Wx−y‖²` | `tasks/quadratic.py` |

Registration is opt-in (`import unified`); nothing in `adapters/` changes and
the adapter-backed names are never shadowed. `import unified` is TensorFlow-free
(TF is imported inside `train()` / `evaluate()`), so the fast test suite never
pays for it.

## Layout

```
unified/
  problem.py      Batch, Problem, ProblemDialect, the problem registry
  policy.py       OptimizerPolicy, NetPolicy (save/restore), PolicySpec + the shared model-free schemas
  model.py        ModelSpec (unrolled networks)
  schema.py       ConfigSchema: allowed / required / defaults / unported / ignored / forbidden keys
  solver.py       IterativeSolver (x_0..x_T), PolicySolver
  meta_train.py   the meta-training loop (port of train_{dm,rnnprop}.py + meta_*_train.py::unroll)
  evalwalk.py     the evaluation walk (trajectory + x_pred; metrics/fingerprint/artifacts from the Problem)
  methods.py      UnifiedModelFreeMethod(PolicySpec), UnifiedUnrolledMethod(ModelSpec) -- no method names inside
  exp2_driver.py  Experiment 2 through the unified methods:  python -m unified.exp2_driver ...
  tasks/          problems:   lasso.py, quadratic.py
  learned/        optimizers: nets_dm.py (verbatim networks), dm.py, rnnprop.py, classical.py,
                              example_momentum.py (the worked example, not registered)
  unrolled/       LISTA-style: lista.py (verbatim models + UnrolledSolver), loader.py, curriculum.py, specs.py
```

Names were chosen so that no submodule is called `problems`, `networks`,
`optimizer`, `meta`, `util`, ... — the library module names whose collision is
why the adapters are subprocess-based (`tests/test_unified_contract.py` checks).

## The three protocols

**`Problem`** (`problem.py`) — an instance source and an objective. The loops
touch exactly two things: `batch.x0` and `problem.value_and_grad(x, batch)`.
```python
class Problem:
    batch_size: int                 # the problem's own (the library scripts fix it per problem)
    eval_reduction: str             # how the model-free scripts reduce over chunks: batch_mean | concat | sample_mean
    def sample(self) -> Batch       # training draw, in the library script's RNG order
    def walk(self) -> Iterator[Batch]   # evaluation pass, every instance once
    def num_chunks(self) -> int
    def per_sample(self, x, batch)  # (B,) objective values; objective()/value_and_grad() derive from it
    # optional hooks, default "nothing":
    def fingerprint(self, batches) -> dict | None            # core.fingerprint over the instances walked
    def recovery_metrics(self, batches, x_pred, eval_section) -> dict
    def write_eval_artifacts(self, result, output_path, tag) -> dict   # the library script's own files
    def training_recovery(self, batch, x_pred) -> dict | None          # the scripts' recovery.npz
```
A `ProblemDialect` is the `config["problem"]` name plus its config vocabulary:
`train_problem(config)`, `eval_problems(config) -> [(label, problem)]` (one per
test split/file; the first is the headline row), optional `val_problem`.

**`OptimizerPolicy`** (`policy.py`) — a learned optimizer:
`initial_state(x)`, `step(x, grad, state, t) -> (x', state')`, `save`/`restore`.
The policy owns its state layout (RNNProp keeps its Adam moments in it). Two
habits keep bit-exactness with the library scripts: create the network lazily
on the first `step` (Keras draws the initial weights then, in the scripts'
order), and use `NetPolicy`'s index-keyed pickle checkpoints.

**`ModelSpec`** (`model.py`) — an unrolled network (layer k = iterate x_k):
`build(problem, identity, create_cells)`, `load_checkpoint`, `train`,
`make_solver`, and the *identity keys* read from `config["train"]` in both
phases so an evaluation can never use a different architecture than was trained.

Both method kinds implement `IterativeSolver.iterate(batch, T) -> x_0 … x_T`;
`evalwalk.evaluate_split` is the only evaluation loop.

## Adding a learned optimizer

`learned/example_momentum.py` is the recipe, executed by
`tests/test_unified_example.py` (it is not registered; copy, rename, register):

1. a network whose variables are created on the first call;
2. an `OptimizerPolicy` (`NetPolicy` gives you save/restore) with
   `initial_state` / `step`;
3. a `PolicySpec`: name, `make_policy(section)` (import TF inside), the config
   keys the method accepts (`MODEL_FREE_TRAIN_SCHEMA.merge(ConfigSchema.of(...))`),
   the script's loop defaults;
4. one line in `METHODS` in `unified/__init__.py`.

You then get: `meta_train` on every registered problem, evaluation with the
learned policy or the classical baselines (`eval.optimizer`), the artifacts
and metrics of each problem, and — by adding a builder to
`tests/fairness_matrix.py` — the determinism and fairness tests.

## Adding a problem

Subclass `Problem` in `tasks/<name>.py`; implement `sample`, `walk`,
`num_chunks`, `per_sample`; override the hooks you have data for. Write one
`ProblemDialect` per config vocabulary (the name, `train_schema` /
`eval_schema`, `train_problem`, `eval_problems`). Register it in
`PROBLEM_DIALECTS` in `unified/__init__.py`. `tasks/quadratic.py` is the
minimal case (30 lines, no hooks); `tasks/lasso.py` the full one.

Two things decide whether parity with a library script is *bit-exact* or
merely statistical: draw random numbers in the same order and with the same
ops as the script's `build()`, and reproduce the script's reduction over
chunks (`eval_reduction`). Both are documented in each task module.

## Adding an unrolled network

Write the model and an `IterativeSolver` view of it (see `unrolled/lista.py`),
a checkpoint loader if its format differs from `unrolled/loader.py`'s, a
training procedure, and a `ModelSpec` in a TF-free module
(`unrolled/specs.py`); register in `METHODS`.

## Parity (what "the same implementation" means here)

`tests/test_unified_parity.py`, `tests/test_unified_train.py`, run on CPU:

* **Evaluation**, real Experiment-2 checkpoints: RNNProp (try3) and DM (try2)
  reproduce the stored trajectories (max |diff| 1e-5 / 4e-7 over 1000 steps);
  LISTA / ALISTA (try3) reproduce `final_output.npy` and the per-layer values
  for all 11 test files. On the same machine the RNNProp evaluation is
  **bit-identical** to `evaluate_rnnprop.py` (0.0 over 1000 steps × 1280
  instances); the remaining difference to the stored GPU files is the scripts'
  own GPU/CPU float32 drift.
* **Training**: a same-seed CPU run of `unified-dm` / `unified-rnnprop` (with
  and without `last_step_loss`) is **bit-identical** to `train_dm.py` /
  `train_rnnprop.py` on both `lasso_dataset` and `quadratic` — every saved
  tensor, `recovery.npz`, every logged loss. Unified-trained LISTA / ALISTA
  checkpoints are read by `Model_Base_L2O/train.py --test` with identical
  results (the curriculum itself is not bit-reproducible in either
  implementation: `tf.data` `experimental_deterministic=False`).
* **`quadratic`**: the Adam baseline and a tiny adapter-trained DM/RNNProp
  checkpoint evaluate bit-identically through both paths, for 1 and 3
  library-generated batches.
* **Fairness**: `tests/fairness_matrix.py` entries `lasso_tiny_unified` and
  `quadratic` prove the unified methods score the very same instances as the
  adapter-backed reference.

## Faithful, not fixed

Reproduced for parity, documented here, switchable only explicitly:

* `unrolled/curriculum.py::LassoLoss` uses `tf.nn.l2_loss` (= sum/2): LISTA/ALISTA
  are trained on `0.25*||Ax-b||^2 + lam*||x||_1` (batch sum) while everything
  is scored on `0.5*||Ax-b||^2 + lam*||x||_1`. `keras_train_loss="paper"`
  (default) keeps that; `"scoring"` trains on the scored objective.
* The trajectory is `[f(x_0) .. f(x_{T-1})]` (the scripts record the objective
  before each step / the metric at each layer's input) while `x_pred = x_T`.
* Model-free LASSO chunk averaging is a plain mean over `ceil(N/batch)` chunks
  (the last, smaller chunk weighs the same as the others), as in the scripts.

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

## Not in the framework yet

* **L2O-Scale** (`unified-scale`): its `optimizer/*.py` are byte-identical
  across Scale/Entropy and would be copied verbatim under `learned/hrnn/`; a
  `ScalePolicy` would own a `tf.Variable` the way `ClassicalAdamPolicy` does
  (Scale's `apply_gradients` needs `id()`-keyed slots on real Variables); its
  evaluation surface has RNG-consuming quirks (a dummy `apply_gradients`
  before loading, a dead `tf.random.normal` per step) that bit-exact parity
  would have to replicate; its training loop needs a `"scale_log"` meta-loss
  (`mean log(f_t/f_0)`), RMSprop and per-element clipping — all expressible as
  `MetaTrainSpec` options.
* Swarm / Entropy / Jacobian stay adapter-only.
