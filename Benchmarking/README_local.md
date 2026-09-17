# Generalized L2O Benchmarking Platform (full local snapshot)

> This is the same document as `README.md`, plus everything that only exists
> in this local working tree (not yet `git commit`-ed). Where the two
> disagree, `README.md` describes what's actually in the repo right now;
> this file describes what's on disk. Local-only additions are called out
> explicitly rather than silently merged in, so it's obvious what would
> change if/when they're committed. See the "Local-only additions" section
> near the bottom for the full list.

A thin, extensible harness for training and evaluating Learn-to-Optimize (L2O)
models through **one common interface**, so new optimizers can be dropped in and
compared without touching the existing libraries.

## Why a harness (and why *thin*)

The migrated libraries fall into two structurally different paradigms:

| Paradigm | Libraries | Training | Evaluation |
|---|---|---|---|
| **Learned meta-optimizers** | L2O-Scale, L2O-Entropy, L2O-Jacobian, L2O-DM, L2O-RNNProp, L2O-Swarm | unroll an *optimizer* over *optimizee* problems | run the trained optimizer on held-out problems, record a loss trajectory |
| **Classical optimizers** | Adam, SGD, Adagrad | nothing to train | run through a host library's eval harness on the same problem |

They also have colliding module names (each ships its own `meta.py`, `util.py`,
`problems.py`, `networks.py`), so they **cannot** be imported into one process.
The harness therefore **invokes each library's existing `train.py` / `evaluate.py`
as a subprocess** in that library's own directory and normalizes the artifacts it
writes. Nothing in the migrated code changes.

## Core vocabulary

- **method** — the thing being benchmarked: a learned optimizer (`l2o-scale`,
  `l2o-dm`, …) or a classical optimizer (`adam`, `sgd`, `adagrad`).
- **problem** — the optimizee the method optimizes (e.g. `mnist_mlp`, `simple`).
- **`BenchmarkResult`** — the standardized output: checkpoint path, training
  metrics, evaluation metrics (`final_loss`, `min_loss`, loss trajectory), and
  artifact paths. This is what `train_evaluate(...)` returns.

## Layout

```
Benchmarking/
  core/
    method.py     # L2OMethod ABC (train / evaluate)
    registry.py   # register_method / get_method / list_methods
    result.py     # TrainOutput, EvalOutput, BenchmarkResult
    shell.py      # subprocess helper (runs a lib's script in its own dir)
    runner.py     # train_evaluate() + run_suite() + leaderboard()
    config.py     # require() -- fail-fast lookup for non-defaulted config keys
    lasso_metrics.py  # FISTA reference solve + Experiment 2's 4 metrics [local: adaptive-restart/certified-gap FISTA, see below]
  adapters/
    scale_family.py   # L2O-Scale / L2O-Entropy / L2O-Jacobian (shared CLI)
    dm_rnnprop.py     # L2O-DM / L2O-RNNProp
    swarm.py          # L2O-Swarm
    model_based.py    # LISTA / ALISTA (layer-wise Keras fit(), not a trajectory optimizer)
    baselines.py      # Adam / SGD / Adagrad (delegate to a host harness)
    __init__.py       # registers every method
  data/
    lasso.py       # shared-dictionary LASSO dataset generator for Experiment 2
  experiments/      # standalone orchestrators; consumers of the platform, not part of it
    experiment1.py       # DA233X pre-study Table 1/2: model-free scalability/truncation sweep
    experiment2.py        # DA233X pre-study Table 3/4: the main 4-method LASSO sweep
    experiment2_try2.py    # Experiment 2 re-run at the primer paper's own (25,50)/noiseless scale
    primer_replication.py  # RNNProp-only correctness check against the primer paper's own protocol
    scale_diagnostic.py    # [local] cheap pre-check: does model-free even converge at Table-3 scale
  configs/        # one JSON per run; lists for suites
    scale_hrnn_mnist.json, suite_mnist.json, swarm_quadratic.json    # committed
    dm_simple.json, rnnprop_simple.json, entropy_hrnn_mnist.json,     # [local] worked
    jacobian_hrnn_mnist.json, lista_lasso.json, alista_lasso.json     # examples, one per method
  run.py          # CLI
  train_evaluate.py  # library entry point (re-exports core.runner.train_evaluate)
  results/        # created on first run
  generate_result_tables.py               # [local] runs/exp2_real/summary.json -> .tex tables
  generate_result_tables_try2_modelbased.py  # [local] same, for runs/exp2_try2 (LISTA/ALISTA subset)
  plot_gradnorm_instability.py  # [local] L2O-DM raw meta-gradient-norm instability figure
  plot_gradnorm_lista.py        # [local] LISTA per-layer gradient norm, from checkpoints
  plot_gradnorm_alista.py       # [local] ALISTA per-layer gradient norm, from checkpoints
  plot_ood_degradation.py               # [local] OOD-degradation figures, runs/exp2_real
  plot_ood_degradation_try2_modelbased.py  # [local] same, runs/exp2_try2 (LISTA/ALISTA subset)
```

## Usage

Always run from within the `thesis_updated` conda env (subprocesses inherit the
interpreter):

```bash
conda activate thesis_updated
cd Benchmarking

python run.py --list-methods
python run.py --config configs/scale_hrnn_mnist.json
python run.py --config configs/scale_hrnn_mnist.json --skip-train   # eval only
python run.py --suite  configs/suite_mnist.json                     # + leaderboard
```

Or programmatically:

```python
from train_evaluate import train_evaluate
result = train_evaluate({
    "method": "l2o-scale", "problem": "mnist_mlp",
    "train": {"num_meta_iterations": 5}, "eval": {"num_testing_itrs": 100},
})
print(result.eval_metrics["final_loss"])
```

## Config schema

```jsonc
{
  "method": "l2o-scale",     // registered method name (required)
  "problem": "mnist_mlp",    // canonical problem name (adapter-specific)
  "host": "l2o-scale",       // classical optimizers only: which harness to run through
  "run_name": "…",           // names the results/ subdirectory
  "output_dir": "results",   // base results directory
  "train": { … },            // raw flags forwarded to the lib's train script
  "eval":  { … },            // raw flags forwarded to the lib's evaluate script
  "skip_train": false,       // evaluate an existing checkpoint only
  "checkpoint": "…"          // optional explicit checkpoint to evaluate
}
```

`train` / `eval` are pass-throughs: any flag the underlying `train.py` /
`evaluate.py` accepts can be set there.

**No hidden defaults.** Adapters don't fall back to a hardcoded value for a
field that changes what gets trained/evaluated or how a checkpoint gets
found/scored -- config that matters must be explicit, or `core.config.require`
raises a `ValueError` naming exactly which key is missing. This is deliberate:
two calls (e.g. `train()` then `evaluate()`) each silently defaulting the same
way only agree by coincidence, and a coincidence can drift. Required keys per
adapter:

| Method(s) | `train` | `eval` |
|---|---|---|
| `l2o-scale`/`-entropy`/`-jacobian` | `optimizer`, `cell_cls`, `cell_size`, `num_cells`, `num_meta_iterations` | `test_optimizer`, `num_testing_itrs`, and (only when `test_optimizer == "L2o"`) `optimizer`/`cell_cls`/`cell_size`/`num_cells` again -- those four are inert for a classical baseline (`L2O-Scale/evaluate.py:172-182`), so left optional there |
| `l2o-dm`/`l2o-rnnprop` | `num_epochs` | `optimizer`, `num_steps`, and (only when `problem="lasso_dataset"`) `lasso_lam`, `num_fista_iters` |
| `l2o-swarm` | `num_epochs` | `optimizer`, `num_steps` |
| `lista`/`alista`/etc. (model-based) | `data_dir`, `num_layers`, `lasso_lam` | `data_dir`, `test_files`, `num_fista_iters` |

Plus top-level `problem` for every method above except when `run_name` is
given directly (model-based) or the method doesn't need it to build a path.
Classical optimizers (`adam`/`sgd`/`adagrad`) need top-level `host`.

## Adding a new L2O model

See `CONTRIBUTING.md` [local: not yet committed] for a full walkthrough (no
prior context assumed). Short version:

1. Write `adapters/my_model.py` with a class subclassing `core.method.L2OMethod`,
   implementing `train(config, output_dir) -> TrainOutput` and
   `evaluate(config, checkpoint, output_dir) -> EvalOutput` (usually by shelling
   out to your model's scripts via `core.shell.run_script` and reading its
   artifacts). Add `run_classical_optimizer_eval` too if its eval harness supports classical
   optimizers.
2. Add one `register_method(MyModel())` line to `adapters/__init__.py`.

That's it — `run.py --method my-model` now works and it appears on the
leaderboard alongside everything else.

## Known limitations (first cut)

- **Problem coverage.** `PROBLEM_MAP` in `scale_family.py` seeds a few canonical
  problems; the Scale-family train and eval problem *sets differ* (toy problems
  for training vs. neural-net problems for evaluation), so those are mapped
  separately. Add entries as needed, or pass raw `include_*_problems` flags.
- **Classical optimizer coverage.** The Scale family harness supports SGD/Adam/Adagrad; the
  DM/RNNProp and Swarm harnesses only ship an **Adam** classical optimizer.
- **Model-based L2O (LISTA family)** is supported via `adapters/model_based.py`
  (LISTA and ALISTA; see `SUPPORTED_MODELS` for the other 7 variants
  `Model_Base_L2O/train.py` exposes but this adapter doesn't yet map), and is
  the method family Experiment 2 uses alongside L2O-DM/RNNProp. It's a Keras
  `fit()`-based layer-wise training task, not a loss-trajectory optimizer, so
  it gets its own result mapping (each layer treated as one "iteration",
  see `model_based.py`'s module docstring) rather than reusing the model-free
  adapters' trajectory format directly.

## Seed control for fair/reproducible comparisons

A top-level `config["seed"] = N` is threaded by `runner.train_evaluate()` into
`config["eval"]["seed"]` (every adapter's `evaluate*.py` now accepts `--seed`,
including the Scale family) and recorded in `provenance["seed"]`. Every
adapter's `evaluate()` also falls back to the top-level `config["seed"]`
directly if `config["eval"]["seed"]` isn't set — so a caller that builds
`config["eval"]` itself without going through `runner.py` (e.g.
`experiments/experiment2.py`) still gets a seeded eval; an explicit
`config["eval"]["seed"]` always wins over this fallback. This gives
**problem-instance reproducibility**: same initial optimizee params / same data
draws for both methods at eval time.

For the Scale family (`l2o-scale`/`-entropy`/`-jacobian`) specifically:
- `evaluate.py --seed N` evaluates on just that one seed instead of the default
  sweep `[6, 12, 18, 24, 30]` — note this makes `num_seeds=1`, so its metrics
  aren't directly comparable to an unseeded (5-seed mean) run.

## Experiment orchestrators

`experiments/` holds standalone scripts that *consume* the platform (via
`core.registry.get_method` / the adapters) to run the pre-study's actual
experiments. They live outside `core`/`adapters` deliberately — they're one
user of the generic harness, not part of its interface:

- **`experiment1.py`** — DA233X pre-study Table 1/2: sweeps L2O-DM,
  L2O-RNNProp, and L2O-Scale across 4 LASSO dimensions x 9 unroll lengths x
  N seeds, recording a per-run table plus a seed-aggregated summary.
- **`experiment2.py`** — DA233X pre-study Table 3/4: the main 4-method
  (L2O-DM, L2O-RNNProp, LISTA, ALISTA) LASSO sweep across training-sparsity
  widths and test sparsities. Its `OrchestratorConfig` decisions are the
  subject of the table below.
- **`experiment2_try2.py`** — re-runs Experiment 2's protocol at the primer
  paper's own tested scale/noise setting ((m,n)=(25,50), noiseless) instead
  of Table 3's (210,300)/40dB, to check whether results trend differently
  there.
- **`primer_replication.py`** — a narrower correctness check: replicates the
  primer paper's own LASSO experiment as closely as practical for
  L2O-RNNProp alone (matching its exact protocol details — sparsity model,
  train size, x0 handling, loss weighting), to confirm this repo's
  reimplementation reproduces the paper's own reported behavior before
  trusting the broader sweeps' numbers.
- **`scale_diagnostic.py`** *(local, uncommitted)* — a cheap pre-check before
  committing to a full model-free hyperparameter search at Table-3 scale
  (m=210, n=300). Reuses `experiment2.py`'s `OrchestratorConfig`/`run_sweep`
  as-is, run at a sequence of (m, n) scales, `width="narrow"` only, evaluated
  in-distribution — isolates "does this method converge at all, and does the
  gap to model-based grow with dimension" from Experiment 2's actual
  width/OOD question.

## Experiment 2 hyperparameter decisions

`experiments/experiment2.py`'s `OrchestratorConfig` values that the pre-study
doesn't specify (everything beyond problem dims/λ/dataset sizes/sparsities/
seeds, which come straight from Table 3). Every value is a deliberate decision
now, not a script default nobody looked at — this table is the source for
each one, so it's auditable instead of just being literals in a dataclass.

| Parameter | Value | Source | Confidence |
|---|---|---|---|
| `model_lam` (LISTA/ALISTA) | `= lasso_lam` (0.005) | Derived from `models/lista.py:66-69`, `models/alista.py:62-63`: θ_init=model_lam/L, W_init=Aᵀ/L. Setting model_lam=lasso_lam makes the untrained network exactly 16-step ISTA on the real objective. `step_lista`'s hard `assert model_lam == lasso_lam` is corroborating evidence this coupling was intended. | High |
| ALISTA step size init | `1.0` | Already the code's own init (`models/alista.py:73`) — confirmed correct, not changed. | High |
| `base_lr` (model-based) | `5e-4` | Primer & Benchmark §4.1.1, stated verbatim. | High |
| `train_batch_size` (model-based) | `128` | Primer & Benchmark §4.1.1. (ALISTA's own repo default is 64; the Primer overrode it to 128, and it matches the model-free batch size too.) | High |
| model-based training schedule | patience-based (not a fixed step count) | Already how `Model_Base_L2O/train.py` works: 3 progressive stages/layer, each gated by `EarlyStopping(patience epochs)` (`train.py:317-369`) — the progressive-stage structure needed no change, just confirmed. `model_based_epochs=200` is a per-stage safety **cap** only; verified via the training log that `EarlyStopping` fires before the cap, not the reverse (an unbound cap would make LISTA/ALISTA's effective budgets asymmetric depending on which converges slower). | High |
| `patience` (Keras epochs, model-based) | `8` | Was `5` (the code's original, unexamined default). Checked directly against the real pilot run's per-epoch `val_lasso` trajectory: the shortest stage (11 epochs) was flat from its very first epoch (no trend being cut off), and the longest stage (108 epochs) showed a smooth asymptotic convergence curve that `patience=5`/`min_delta=0` correctly followed all the way to its true plateau rather than stopping early — no evidence patience=5 was causing premature stops. Bumped to 8 anyway as a low-cost safety margin (real model-based training is cheap regardless of patience). Not matched to the reference ALISTA repo's raw 2,000-4,000-*step* patience, since that number isn't directly comparable — the reference validates every 10 steps vs. this port's once per (250-step) epoch, so its patience corresponds to ~200-400 validation checks tolerated, not a value meaningfully translatable into this port's per-epoch checks. | Medium — empirically checked, but the exact number (8 vs. e.g. 10) is a judgment call |
| `num_layers` | `16` | Primer & Benchmark §4.1.1/§4.1.2 ("vanilla LISTA and ALISTA, both unrolled to 16 layers"); ALISTA repo's own `-T` default. | High |
| `share_W` (LISTA) | `False` (untied per-layer weights) | Not separately exposed in `OrchestratorConfig` — the code's own default. "Vanilla LISTA" (Primer & Benchmark's own phrase for what's being run here) means untied weights, so this default is already the right value; logged here so it's a recorded decision rather than an unexamined default. | High |
| `unroll_length` (model-free) | `20` | Stated in both the DM and RNNProp papers' own (non-LASSO) experiments. | High |
| `num_steps` (model-free, training) | `1000` | **Corrected** — Primer & Benchmark §4.1.2, the exact LASSO experiment being replicated, states models were "trained... for 100 epochs and 1,000 iterations per epoch." This overrides the generic 100-steps-total figure from the original DM/RNNProp papers' own (non-LASSO) experiments, which don't cover LASSO at all. | High — direct quote, exact experiment being replicated |
| `model_free_batch_size` | `128` (LASSO instances per meta-training step) | **Not paper-sourced** — LASSO isn't one of either paper's own benchmarked problems, so neither states a batch size for it. Judgment call: kept equal to model-based's batch size (which *is* paper-grounded, Primer §4.1.1) rather than picked independently, so there's no arbitrary cross-family asymmetry on a parameter neither paper actually constrains. | Medium — judgment call, not derived from a source |
| meta `learning_rate` | `1e-3` | Neither paper states a value (DM: "chosen by random search"); matches `train_dm.py`/`train_rnnprop.py`'s own argparse defaults — implementation convention, not a paper value. | Medium |
| `num_epochs` (model-free) | `100`, identical for DM and RNNProp | **Corrected** — was `10000`, sourced from the wrong experiment (Primer §4.3's MNIST/ConvNet test: "trained... from 10,000 different random initializations" — a different problem, not LASSO). The actual LASSO experiment (§4.1.2, quoted above) states 100 epochs. Forcing DM/RNNProp equal is still deliberate (Open-L2O's own recommended commands use very different budgets per method — 10000 vs 100 — which would be an unfair asymmetry here), but the number itself is now the paper's own LASSO-specific figure, not a guess. **Cost measured, not extrapolated**: a real 20-epoch run at `num_steps=1000` (Docker+GPU, RTX 2070) took 1655.7s → ~82.8s/epoch → ~2.3h per training run → **~138 GPU-hours (~5.75 days serial) for the full 60-run model-free sweep** — better than the old (wrong) config's ~360h/~15 days, but not the naive 10x improvement a linear steps-vs-epochs tradeoff would suggest (more unrolls/epoch adds overhead beyond the raw step count). That same run's training loss was also quite volatile (28424→2107→...→7→...→41005 by epoch 19) — consistent with the Primer's own documented LSTM-optimizer instability (§2.1), not a bug; it's exactly why periodic best-checkpoint selection (`evaluation_period` row below) matters here. | High — direct paper quote for the value, directly measured for the cost |
| `model_based_epochs` (safety cap) | `200` | **Verified via a real-dims pilot** (m=210,n=300, 32,000 train instances, 4-layer LISTA, Docker+GPU): all 12 stages (4 layers × 3 stages) triggered `EarlyStopping` well before the cap — longest stage used 108/200 epochs (54%), most under 25. Total wall time for 4 real layers: ~7 minutes. This also surfaced and fixed two real pre-existing bugs unrelated to this parameter (see note below the table) — the pilot's first two attempts silently trained zero parameters before those were found. Caveat: only checked at 4 layers, not the real 16; later layers' joint fine-tuning stages showed a mild upward trend (stage 11 was the longest), so headroom may shrink somewhat by layer 16, but the margin at 4 layers is large enough that 200 is very unlikely to bind. | High — measured directly, small residual extrapolation risk (4→16 layers) |
| `evaluation_period` (model-free) | `5`, identical for both | The scripts' own defaults disagree (`train_dm.py`=100, `train_rnnprop.py`=10) and both are calibrated for their native experiments' much larger epoch counts (e.g. Open-L2O's own recommended `train_dm.py --num_epochs=10000`) — at the corrected `num_epochs=100`, either default would validate the checkpoint only once (or never), defeating best-checkpoint selection entirely. Rescaled to 5 (20 validation events over the 100-epoch run) so periodic selection actually happens; forced identical for the same cross-method-fairness reason as before. Not paper-stated. | Medium — judgment call, rescaled from the "identical for both" principle |
| `evaluation_epochs` (model-free) | `20` | Already identical in both scripts' defaults; made explicit rather than left implicit. | High |
| test-set evaluation coverage | full deterministic pass over all 1,280 instances, all 4 methods | Was a single random ~128-instance bootstrap draw for DM/RNNProp (`evaluate_dm.py`/`evaluate_rnnprop.py --num_epochs` defaulting to 1) vs. LISTA/ALISTA's already-full deterministic pass — fixed in `problems.py`'s `lasso_from_dataset` (added a `deterministic` walk mode) + both `evaluate_*.py` scripts, so all four methods now score every test instance exactly once. Primer & Benchmark's own protocol (Eq. 21) is explicit that evaluation is over the entire test set. | High |
| model-free test horizon | `1000` iterations | Primer & Benchmark §4.1.2: "we set 1,000 iterations for both model-free L2O methods." **Not directly comparable to model-based's 16 layers** — the Primer found exactly this gap and worked around it by extending LISTA/ALISTA's curve with FISTA iterations; here it's instead reported as a separate matched-budget-at-k=16 table alongside the full curves, with the caveat (in the figure caption, not just body text) that one L2O-DM/RNNProp "iteration" additionally runs an LSTM per coordinate, so equal counts don't mean equal FLOPs/wall-clock. | High |
| FISTA reference solve | adaptive restart + certified duality-gap stopping (1e-10 relative), not a fixed 2000 iterations *(local, uncommitted — see below)* | Plain FISTA at 2000 iterations (the Primer's own convention) is measurably under-converged at this problem's denser test sparsities. Measured directly against this repo's own solver (`m=210,n=300`, unit-norm columns, 3 sparsities): the certified relative duality gap after 2000 plain-FISTA iterations was ~4e-15 at p=0.02 vs. ~5.6e-8 at p=0.40 — about 7 orders of magnitude worse at the dense end, not merely "worse." (An external estimate for this same problem separately suggested ~3e-4 to 6e-4 at p=0.40 for a differently-parameterized problem instance; the two aren't measuring the same setup, so don't read them as agreeing — this repo's own number is what the fix is based on.) Implemented in the **local working copy** of `core/lasso_metrics.py::fista_solve` (per-instance gradient-scheme adaptive restart + a certified-gap early-stopping check every 25 iterations, `num_fista_iters` now a safety cap of `50_000` rather than the real count); **not yet committed** — the version currently in git is a plain fixed-2000-iteration FISTA loop (`README.md` describes that committed behavior; `run.py`/`experiment2.py` as committed instead just run the fixed-iteration loop out to `num_fista_iters=50_000` every call). | High — measured directly against this repo's own solver |
| x₀ alignment | seeded, shared across all 4 methods (not zero, not independently random) | `lasso_from_dataset`'s optimizee used to start at `tf.random.normal(stddev=0.01)` (independently drawn per seed); LISTA/ALISTA's layer 0 implicitly started from `x₀=0` (no input slot for it at all). Decided: generate one seeded `x₀` per instance in `data/lasso.py` (new `<split>_x0.npy` sibling file per split, `x0_stddev=0.01`) and feed it into **every** method's first step/layer, so for a given seed all four methods start each instance from the identical point. Required real changes, not just config: `data_preprocessing.py` now concatenates `x0` onto the model input for `task='lasso'`; `ListaCell`/`AlistaCell`'s layer-0 formula now consumes it (LISTA via a **fixed**, non-trainable `(I-BA)` matrix — kept fixed so the network still has exactly `T` layers of learnable parameters, not `T+1`; ALISTA's formula generalizes with no special-casing needed). Verified: DM's first recorded eval cost matches an independently-computed mean f(x₀) over the test file exactly; the real (edited) `Lista`/`Alista` Keras classes' layer-0 output matches an independent hand-derivation of "one ISTA step from x₀" to float32 precision. | High — implemented and verified against this repo's own code |
| `beta1`/`beta2` (RNNProp only) | `0.95` / `0.95` | Decay rates for RNNProp's internal Adam-style input normalization (the m̃/g̃ features fed to its LSTM). Found during a full re-audit of every script's flag list: this was silently sitting at `train_rnnprop.py`/`evaluate_rnnprop.py`'s own default, never examined all session. Now threaded through explicitly (`OrchestratorConfig.model_free_rnnprop_beta1/2`, conditional on `method_name=="l2o-rnnprop"` since `train_dm.py` has no such flags at all) and kept identical between train and eval (a mismatch here would apply a trained checkpoint through a differently-configured normalization scheme). **Not independently verified against the original RNNProp paper's stated value** — carrying the code default forward deliberately, not claiming it's confirmed-correct. | Low/Medium — real gap, flagged not resolved |
| `ss_q_per_layer` / `ss_maxq` (ALISTA only) | `1.2` / `13.0` | ALISTA's per-layer support-selection schedule (`models/alista.py`'s `q[t] = clip((t+1)·ss_q_per_layer, 0, ss_maxq)`, used by `shrink_ss`) — found in the same re-audit, also never examined, silently at `Model_Base_L2O/train.py`'s own default. Now threaded through explicitly (`OrchestratorConfig.model_based_ss_q_per_layer/ss_maxq`), passed unconditionally in the model-based train config (harmless no-op for LISTA, which doesn't use support selection — absl flags are always defined regardless of `model_name`). **Not independently verified against the ALISTA/LISTA-CPSS papers' own stated schedule.** | Low/Medium — real gap, flagged not resolved |

**Two real, pre-existing bugs found and fixed while pilot-checking `model_based_epochs`** (both predate this session, confirmed against `git show HEAD`, not introduced by any change this session):
- **LISTA/ALISTA never actually trained under Keras 3.** `Lista`/`Alista` (and, by the same pattern, all 7 other `Model_Base_L2O/models/*.py` variants) create trainable parameters as plain `tf.Variable(...)` attributes. Keras 3 (this repo runs 3.15.0) does not auto-track bare `tf.Variable` attributes as layer weights — confirmed directly against this environment with a minimal reproduction (0 tracked weights for `tf.Variable`, correctly tracked for the same pattern using `keras.Variable`). Every prior LISTA/ALISTA `model.fit()` call, including every smoke test this whole session, optimized zero parameters; the loss values that looked like progress were just the untrained architecture's own layer-wise ISTA-style construction, not learning. **Fixed** in `lista.py`/`alista.py` by swapping every trainable-parameter `tf.Variable(...)` to `keras.Variable(...)` (same arguments; both files already imported `keras`). The other 7 model variants carry the same bug and were deliberately **not** fixed (out of Experiment 2's scope, same reasoning as the x0-alignment scope note above).
- **The custom `Adam` optimizer's `apply_gradients` override used a dropped Keras 3 signature.** `Model_Base_L2O/utils.py`'s `Adam.apply_gradients` forwarded `(grads_and_vars, name, all_reduce_sum_gradients)` to the parent class; Keras 3's `BaseOptimizer.apply_gradients` only accepts `(self, grads_and_vars)` now (the other two were TF1/TF2-era distributed-training args, removed in Keras 3). This only surfaced once the tracking bug above was fixed and real gradients started flowing. **Fixed** by dropping the two removed arguments from the `super().apply_gradients(...)` call; the per-variable gradient-scaling logic itself (the `0.3**self.var_list[v.name]` layer-age decay) is unchanged.

Both fixes verified against real training, not just unit-level: a real-dims 2-layer ALISTA run (same pilot setup) also completed cleanly post-fix — genuine convergence, `model.summary()` correctly showing 4 trainable params (theta + step_size per layer), all 6 stages finishing well under the epoch cap (max 25/200).

**Open, not yet decided:**
- **x₀ threading for the other 7 model-based variants** (`lfista`, `lamp`, `step_lista`, `lista_cp`, `lista_cpss`, `glista`, `tista`). `data_preprocessing.py` now feeds every `task='lasso'` model a wider `[b, x0]` input regardless of `model_name`; confirmed safe (they all only read `inputs[:, :M]` at layer 0, so the extra x0 tail is harmlessly ignored) but they still implicitly start from `x0=0` internally, same as before. Only `lista`/`alista` were changed since only those two run in Experiment 2 — out of scope, not an oversight.

For training-run reproducibility (weight init, unroll sampling), `train.py`
also accepts `--seed N` for the Scale family, `l2o-dm`, `l2o-rnnprop`, and
`l2o-swarm` — every learned-optimizer adapter's `train()` reads
`config["seed"]` directly and forwards it (an explicit
`config["train"]["seed"]` still wins). Verified bit-exact reproducible across
separate process invocations for all four.

**Known gap:** `--seed` selects *which* problem-instance is evaluated
deterministically, but doesn't guarantee bit-exact eval results for neural-net
problems (MNIST MLP/conv) — verified this diverges by ~0.1% across two runs of
the *same* seed even on unmodified `evaluate.py` (pre-existing, not a
regression from this change), most likely floating-point/threading noise in
TF's CPU kernels or Keras-layer weight init that isn't routed through the
`seed` argument. Quadratic/toy problems, whose init is a single
`tf.random.normal(..., seed=seed)` call, are unaffected. `train.py --seed`'s
reproducibility (verified bit-exact across processes) does not have this gap.

## Docker image: how the running one was actually built

The repo's `Dockerfile` describes a **clean** build. The image that has actually
produced every result to date was not built that way, and the difference matters
enough to record here — reading the `Dockerfile` alone will tell you the wrong
thing about what `open-l2o:latest` contains.

| tag | what it is |
|---|---|
| `open-l2o:2026-08-05` (`6c7e011c6a61`) | the original image, built from the repo `Dockerfile` on 2026-08-05. Every result in `runs/` up to 2026-09-12 came from this. |
| `open-l2o:latest` | that image **plus one apt layer** adding git 2.47.3, derived 2026-09-12. Not a rebuild. |

### Why git had to be added

`harness.py`'s `capture_provenance` (and `experiment1.py`'s copy) needs a `git`
binary to record working-tree state. Without one it degrades to reading `.git/`
directly: commit and branch yes, but `dirty: null`, no `git status --porcelain`,
and no diff sidecar — which is close to useless here, since much of what these
sweeps execute is uncommitted or entirely untracked (the v2 orchestrators
included). The `Dockerfile` had carried an uncommitted `git` line for some time,
but the image predated it, so the fix had never actually been built.

### Why it was derived rather than rebuilt — do not skip this

A plain `docker build` of the repo `Dockerfile` was attempted first and
**aborted after ~10 minutes.** `python:3.12-slim` was not in the local image
store, so BuildKit pulled *today's* tag, missed every cached layer, and began a
full ~20 GB `--no-cache-dir` reinstall of both venvs. `requirements-tensorflow.txt`
pins top level exactly (TF 2.21.0, Keras 3.15.0, numpy 2.5.0), but **the Debian
base and every transitive dependency are unpinned** — protobuf, h5py, ml-dtypes,
absl-py, the `nvidia-*-cu12` set, the whole torch resolve. Completing that build
would have put later cells of an in-flight sweep on a different environment from
its earlier cells and from every prior run in `runs/`.

What was done instead (~12 s, every TF/CUDA layer byte-identical to the image
that produced all prior results):

```
docker tag open-l2o:latest open-l2o:2026-08-05
docker build -t open-l2o:latest - <<'EOF'
FROM open-l2o:2026-08-05
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
EOF
```

This is the general technique for changing a live environment mid-series: derive
`FROM` the existing image, so there is no cache to miss and nothing underneath
can drift. Reuse it for any future addition while a sweep is running.

### Verified on the bind mount, not just `git --version`

Installing git is not sufficient on its own — Linux git reading a Windows
checkout can produce a junk diff. Measured inside the container against the host:
`status --porcelain -uall` = **113 paths vs 112 on the host**, `diff HEAD` =
**37,592 B**. Critically, **no `core.fileMode` inflation** (Windows bind mounts
can expose every file as 0777, making Linux git call every tracked file
mode-changed; counts were identical with and without `-c core.fileMode=false`).
So `_run_git`'s existing `-c safe.directory=* -c core.autocrlf=true` pair is
sufficient and no harness change was needed. Re-check this if the mount setup
ever changes.

### Restarting a sweep onto a new image needs a NEW container

A container is pinned to the image it was created from, so `docker stop` +
`docker start` silently keeps the old one. The sequence is stop → `docker rename
<name> <name>_nogit` (rename, never `rm` — the logs are the only record of what
that container did) → `docker run` with the original name and the same command.
`runs/exp2_try3_210x300` was relaunched this way at 0 completed cells, so its
`provenance.jsonl` holds two records: record 1 degraded (`dirty: null`) from the
aborted start, record 2 complete. **Read record 2.**

### The `Dockerfile`'s trailing `RUN`, and the clean-build caveat

`git` sits in its own `RUN` at the *end* of the `Dockerfile` rather than in the
`build-essential` one at the top. That first `RUN`'s text is the cache key for
the two multi-GB pip layers below it, so folding `git` in there invalidates both
by definition. The trailing placement makes future end-of-file additions cheap —
**but only when the base image is also still cached locally**, which is exactly
the condition that failed above. Necessary, not sufficient.

For a genuine from-scratch build (someone else, no prior image):

- It should work — the aborted attempt was a true cold build and got through the
  base pull and into the pip layers without error — but the last build verified
  **end-to-end** is 2026-08-05. The 2026-09-12 attempt was killed partway.
- It will **not** reproduce this environment. `FROM python:3.12-slim` is a moving
  tag and the transitives float. The current image's base identity is Python
  **3.12.13**, Debian **trixie** snapshot `@1785715200` (≈ 2026-08-02); the exact
  base digest is *not* recoverable from the image, since BuildKit does not stamp
  the parent digest into the child's history.
- To close that: pin the base as `FROM python:3.12-slim@sha256:...` (digest from
  Docker Hub's tag history for early August 2026) and freeze the transitives with
  a `pip freeze` taken inside `open-l2o:2026-08-05`. Neither has been done.

## Local-only additions (not yet committed)

Everything below exists on disk in this working tree but has no commit —
`git status` shows these as untracked (or, for `core/lasso_metrics.py`,
modified). None of it is referenced from `README.md`. Listed here so a
`git commit` covering "the benchmark platform" doesn't quietly leave pieces
behind without anyone noticing.

**Code:**
- **`core/lasso_metrics.py` (modified, not new)** — adds `_certified_gap()`
  and rewrites `fista_solve()` to use per-instance adaptive restart
  (O'Donoghue & Candès, 2012) plus certified-duality-gap early stopping
  (checked every 25 iterations, `1e-10` relative tolerance), raising the
  default/safety-cap iteration count from `2000` to `50_000`. See the FISTA
  row above for the measurement this is based on.
- **`experiments/scale_diagnostic.py`** — the model-free scale-convergence
  pre-check described in "Experiment orchestrators" above.
- **`generate_result_tables.py`** — reads `runs/exp2_real/summary.json`,
  writes `experiment2_results_tables.tex` (one LaTeX table per Table-4
  metric, rows grouped by training-sparsity width, mean±std across the 10
  seeds). One-off analysis script, not imported by anything.
- **`generate_result_tables_try2_modelbased.py`** — same idea, forked (not
  parameterized — the two runs differ in scale, sparsity grid, and method
  coverage) for `runs/exp2_try2/summary.json`, scoped to the LISTA/ALISTA
  10-seed-complete subset only, writing
  `experiment2_try2_modelbased_results_tables.tex`.
- **`plot_gradnorm_instability.py`** — plots L2O-DM's raw (unclipped)
  meta-gradient norm per training epoch from a `profile.jsonl` diagnostic
  run, log scale, marking the curriculum horizon-extension epoch.
- **`plot_gradnorm_lista.py`** — recomputes and plots LISTA's per-layer
  LASSO-loss gradient norm from each layer's saved checkpoint (read-only;
  doesn't modify `Model_Base_L2O`'s training code).
- **`plot_gradnorm_alista.py`** — same for ALISTA, but reads the full
  `exp2_real` sweep's native (210,300)-scale checkpoints rather than the
  (25,50) diagnostic ones (ALISTA has no (25,50) checkpoint/`W.npy`) — not
  scale-matched to the other two gradnorm figures.
- **`plot_ood_degradation.py`** — per-method OOD-degradation figures
  (Modified Relative Loss vs. test sparsity, one line per training width)
  from `runs/exp2_real/results.jsonl`, using median/IQR rather than
  mean±std (checked directly against the per-seed distributions —
  L2O-DM's is heavily right-skewed by occasional divergent runs).
- **`plot_ood_degradation_try2_modelbased.py`** — same, forked for
  `runs/exp2_try2/results.jsonl`, scoped to LISTA/ALISTA.

**Config/docs:**
- `README.md`, `README_local.md` (this file), `CONTRIBUTING.md`,
  `.gitignore` — the docs and repo hygiene file this session produced.
- `configs/dm_simple.json`, `rnnprop_simple.json`, `entropy_hrnn_mnist.json`,
  `jacobian_hrnn_mnist.json`, `lista_lasso.json`, `alista_lasso.json` — one
  worked-example config per registered learned-optimizer method beyond the
  three already committed (`scale_hrnn_mnist.json`, `suite_mnist.json`,
  `swarm_quadratic.json`).

**Generated result artifacts** (outputs of the scripts above, not source):
`experiment2_hyperparameters.tex`, `experiment2_results_tables.tex`,
`experiment2_try2_modelbased_results_tables.tex`,
`experiment2_try2_modelbased_ood_degradation_{alista,lista}.{pdf,png}`,
`gradnorm_alista_210x300.{pdf,png}`,
`gradnorm_instability_l2o-dm_25x50.{pdf,png}`,
`gradnorm_lista_25x50.{pdf,png}` (plus their mirrors under the repo-root
`Figs/` directory). Regenerate by rerunning the corresponding script above
after `runs/`'s underlying sweep data changes — these aren't hand-edited.
