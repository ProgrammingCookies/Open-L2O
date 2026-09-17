# Contributing a new method to the Benchmarking platform

This is the doc to read if you've just found this repo and want to plug your
own L2O method (or a classical baseline) into the benchmark without first
reading every existing adapter end to end. It assumes no prior context beyond
basic Python. For the platform's own design rationale and known limitations,
see `README.md` — this doc is the "how do I add mine" walkthrough; that one
is the reference.

## 1. What this platform actually does

You have (or are about to have) a learned optimizer — code that trains a
model to produce optimizer updates, then can run that trained model on new
problems. The Benchmarking platform's job is to let that method be trained
and evaluated through **one shared interface**, so its numbers land next to
every other method's numbers in a comparable format, without you having to
touch anyone else's code and without them having to touch yours.

Concretely, three pieces:

- **A `method`** — your optimizer. You register it once under a name (e.g.
  `"my-method"`) and from then on `run.py --method my-method ...` works.
- **A `problem`** — the thing being optimized (e.g. `mnist_mlp`, `lasso`).
  Problem names are adapter-specific strings your adapter interprets however
  your library needs.
- **An adapter** — the file *you* write. It's the translation layer between
  the platform's generic `train(config) -> checkpoint` /
  `evaluate(config, checkpoint) -> metrics` calls and whatever your method's
  actual code looks like.

Nothing about "the platform" lives in one deep framework you have to learn.
It's a 7-file `core/` package (~700 lines total, of which `lasso_metrics.py`
is ~200) plus one adapter file per method. Read `core/method.py` first — it's
49 lines and it's the entire contract.

## 2. Why adapters shell out to subprocesses (and why yours doesn't have to)

The five learned-optimizer libraries already in this repo (`L2O-Scale`,
`L2O-Entropy`, `L2O-Jacobian`, `L2O-DM`, `L2O-RNNProp`, `L2O-Swarm`) each ship
their own top-level `meta.py` / `util.py` / `problems.py` / `networks.py`.
Importing two of them into the same Python process collides in
`sys.modules` and silently cross-wires the libraries. So their adapters run
each library's existing `train.py` / `evaluate.py` as a **subprocess** via
`core/shell.py`, in that library's own directory, and read back whatever
files it writes.

**This is a workaround for a naming collision, not a requirement of the
platform.** If your method is a normal, self-contained Python
package/module, your adapter's `train()`/`evaluate()` can just call your
code directly, in-process — no subprocess needed. Only reach for
`core.shell.run_script` if you're wrapping an existing script that can't be
safely imported alongside the others.

## 3. The contract you implement

Everything lives in `core/method.py`:

```python
class L2OMethod(abc.ABC):
    name: str = ""            # registry name, e.g. "my-method"
    trainable: bool = True    # False for classical optimizers with nothing to train

    def train(self, config: dict, output_dir: str) -> TrainOutput:
        # default: no-op, returns TrainOutput(checkpoint=None)
        ...

    @abc.abstractmethod
    def evaluate(self, config: dict, checkpoint: str | None, output_dir: str) -> EvalOutput:
        ...
```

`TrainOutput` and `EvalOutput` (`core/result.py`) are plain dataclasses:

```python
TrainOutput(checkpoint, metrics={}, artifacts={}, command=None)
EvalOutput(metrics={}, trajectory=[], per_seed={}, artifacts={}, command=None)
```

- **`checkpoint`** — whatever your `evaluate()` needs to restore your trained
  model. A directory, a file path, a name — it only has to mean something to
  *your* adapter.
- **`trajectory`** — a flat list of scalar objective values, one per
  optimization step, from a representative evaluation run. This is what
  makes methods comparable: `core.result.summarize_trajectory()` reduces any
  trajectory to `final_loss` / `min_loss` / `mean_loss` / `num_points`, and
  those four numbers are what shows up in the leaderboard regardless of
  which library produced them.
- **`metrics`** — anything extra worth recording (free-form dict). Put
  `summarize_trajectory(trajectory)`'s output in here, plus whatever else is
  meaningful for your method.
- **`per_seed`** — optional structured breakdown, if you have one (see
  §5 below — this field's meaning is genuinely inconsistent across the
  existing adapters, so don't pattern-match it blindly).
- **`artifacts`** — `name -> path` for anything you wrote to disk that's
  worth keeping a pointer to (logs, raw result files, plots).
- **`command`** — if you shelled out, the exact command string, for
  provenance. `None` if you ran in-process.

## 4. Writing your adapter, step by step

1. **Create `adapters/my_method.py`.**

2. **Decide what "one step" means for your method**, before writing any
   code. This is the one genuinely required design decision the interface
   doesn't make for you: your method probably has *some* natural sequence of
   values as it optimizes (a loss per gradient step, per meta-iteration, per
   unrolled layer, per generation of a population...). Pick a scalar
   sequence you're willing to call `trajectory`, understanding that
   `final_loss`/`min_loss` on the leaderboard will be compared directly
   against every other method's `trajectory` — see the caveat in §6 about
   what "comparable" actually means here.

3. **Implement `train()`.** Minimal shape:

   ```python
   from core.method import L2OMethod
   from core.result import TrainOutput

   class MyMethod(L2OMethod):
       def __init__(self, name: str):
           self.name = name

       def train(self, config, output_dir):
           # config["train"] is a free-form dict of whatever flags/hparams
           # your training code needs; config["seed"] is the top-level seed
           # if the caller set one (see §7 — don't assume it reached you).
           ckpt = my_library.train(output_dir=output_dir, **config.get("train", {}))
           return TrainOutput(checkpoint=ckpt, artifacts={...})
   ```

4. **Implement `evaluate()`.**

   ```python
       def evaluate(self, config, checkpoint, output_dir):
           trajectory = my_library.evaluate(checkpoint, **config.get("eval", {}))
           metrics = summarize_trajectory(trajectory)
           return EvalOutput(metrics=metrics, trajectory=trajectory)
   ```

5. **Write everything under `output_dir`, not somewhere derived from the
   checkpoint.** Every built-in adapter except one (`SwarmMethod`, and only
   in its L2L eval branch — see `REVIEW_NOTES`/review history) writes its
   `train.log` / `eval.log` / result artifacts under the `output_dir` it was
   given. Do the same. If eval artifacts get written next to the checkpoint
   instead, re-evaluating the same checkpoint with different settings (a
   normal, supported workflow — see `skip_train` below) silently overwrites
   the previous run's artifacts, and any earlier saved `result.json` that
   pointed at that path now resolves to different content than what actually
   produced its numbers.

6. **(Optional) support classical-optimizer baselines through your harness.**
   If your evaluation code can also run Adam/SGD/etc. on the same problem,
   add:

   ```python
       def run_classical_optimizer_eval(self, config, classical, output_dir):
           cfg = dict(config)
           cfg["eval"] = {**cfg.get("eval", {}), "optimizer": classical}
           return self.evaluate(cfg, checkpoint=None, output_dir=output_dir)
   ```

   This isn't part of the `L2OMethod` ABC — `adapters/baselines.py` looks it
   up with `hasattr()` — so skip it entirely if it doesn't apply.

7. **Register it.** One line in `adapters/__init__.py`:

   ```python
   from adapters.my_method import MyMethod
   register_method(MyMethod("my-method"))
   ```

   That's it. `python run.py --list-methods` now lists it, and
   `python run.py --config some_config.json` with `"method": "my-method"`
   works.

## 5. `per_seed` — pick a meaning, but know it isn't standardized

You'll notice existing adapters use `EvalOutput.per_seed` for three
different things: `scale_family.py` keys it by seed (`"seed6"`) mapping to a
`summarize_trajectory`-shaped dict; `model_based.py` keys it by test
filename with the same shape; `swarm.py` uses it for one unrelated key
holding a raw list. There's no enforced contract here — use whatever
structure makes sense for your method's own repeated/multi-instance
evaluations (multiple seeds, multiple test files, multiple population
members...), but don't assume downstream generic code (a leaderboard, a
plotting script someone else writes) can interpret it without knowing your
adapter's specific convention. If your method naturally produces multiple
comparable sub-results per `evaluate()` call, document the shape you chose
in your adapter's module docstring, the way the existing ones do.

## 6. What "comparable" does and doesn't mean

`summarize_trajectory` gives every method the same four numbers
(`final_loss`, `min_loss`, `mean_loss`, `num_points`), which is what makes
the leaderboard and suite comparisons possible at all. But it's comparing
*shape*, not necessarily *cost*: one entry in an L2O-Scale trajectory is one
optimizer step; one entry in the LISTA-family adapter's trajectory is one
unrolled network *layer*, which may do very different amounts of work.
Decide, and document in your adapter, what one point in your trajectory
actually costs — a reader comparing your `final_loss` against another
method's shouldn't have to guess whether the x-axes line up.

Also be aware: nothing currently stops a suite (`run.py --suite`) from
mixing methods evaluated on *different* `problem`s into one leaderboard sort
by `final_loss`. That's an apples-to-oranges comparison the platform won't
catch for you — keep suites problem-scoped in practice.

## 7. Config, seeds, and the pass-through pattern

A config is just a dict:

```jsonc
{
  "method": "my-method",
  "problem": "mnist_mlp",
  "seed": 42,
  "run_name": "my_first_run",
  "output_dir": "results",
  "train": { /* anything your train() understands */ },
  "eval":  { /* anything your evaluate() understands */ },
  "skip_train": false,       // evaluate an existing checkpoint only
  "checkpoint": "…"          // explicit checkpoint path, used with skip_train
}
```

`train`/`eval` are raw pass-throughs — your adapter decides what keys mean
anything; nothing validates them for you, so a typo'd key will silently do
nothing rather than error (or, if you're shelling out to an argparse/absl
script, will produce a subprocess crash several layers down — check your
`.log` files first if a run fails mysteriously). For any field your adapter
would otherwise default (a hyperparameter, or anything `train()`/`evaluate()`
both need to independently agree on to find/score the same checkpoint), use
`core.config.require(section, key, message)` instead of `section.get(key,
default)` — see `adapters/model_based.py` for the pattern. A default that two
separate calls happen to agree on today is a coincidence, not a guarantee.

**Seed handling, concretely:**

- If you call `train_evaluate()` (`core/runner.py`) or go through `run.py`,
  a top-level `config["seed"]` is automatically copied into
  `config["eval"]["seed"]` before your `evaluate()` is called
  (`runner.py`'s `train_evaluate`). It is **not** automatically copied into
  `config["train"]` — only adapters that explicitly read
  `config.get("seed")` themselves (see `ScaleFamilyMethod.train()`) get
  seeded training. If your training code accepts a seed, read
  `config.get("seed")` directly in your own `train()`; don't assume the
  runner does it for you.
- **If you write your own orchestrator/sweep script instead of going through
  `run.py`/`core.runner.train_evaluate`** (as the `experiments/` scripts
  do, deliberately, to stay decoupled from the platform's generic path),
  the seed-threading above does not happen automatically — you are
  responsible for putting `"seed": N` into whatever `eval` dict you build
  yourself. `experiments/sweep.py` does **not** do this for you either: it
  drives the grid, not the config (see §11). It's easy to record a seed in your results row while never
  actually having passed it to the thing you're evaluating; verify by
  checking the actual logged command (`eval.log`'s `# cmd:` line) for
  `--seed`, not just by trusting that the config said so.

## 8. Testing your adapter before trusting its numbers

1. `python run.py --list-methods` — confirms registration and that
   `adapters/__init__.py` still imports cleanly (a broken import in *any*
   adapter breaks this for everyone, since `adapters/__init__.py` imports
   all of them unconditionally).
2. Write the smallest config you can that finishes in well under a minute —
   tiny problem size, 1-2 training steps — and run
   `python run.py --config configs/my_smoke.json`. Check `result.json` and
   the `train.log`/`eval.log` it wrote under `results/<run_name>/`.
3. Re-run with `--skip-train` pointing at the checkpoint you just produced,
   to confirm evaluate-only mode works and doesn't silently clobber the
   first run's artifacts (§4 step 5).
4. If your method should be leaderboard-comparable to an existing one, run
   both through `run.py --suite` on the same `problem` and sanity-check that
   `final_loss` differences look plausible, not like an eval batch or seed
   mismatch.

## 9. Cheat sheet: the CLI vocabulary you're about to reverse-engineer

Registering an adapter is one line; understanding the script you're wrapping
is the actual work. There isn't one shared flag convention across the
libraries in this repo — there are three, and knowing which one you're
looking at up front saves a trip through unfamiliar argparse/absl source:

| Library family | Arg framework | Save/checkpoint flag | Iteration-budget flag | Notes |
|---|---|---|---|---|
| L2O-Scale / L2O-Entropy / L2O-Jacobian (`scale_family.py`) | `argparse` | `--train_dir` (train), `--save_dir` (eval) | `--num_meta_iterations` | Eval sweeps 5 fixed seeds `[6,12,18,24,30]` unless `--seed` narrows it to one. |
| L2O-DM / L2O-RNNProp / L2O-Swarm (`dm_rnnprop.py`, `swarm.py`) | `argparse` | `--save_path` | `--num_epochs` / `--num_steps` | RNNProp adds `--beta1`/`--beta2`; DM has neither. |
| Model_Base_L2O — LISTA/ALISTA family (`model_based.py`) | `absl.flags` | `--base_dir` / `--data_dir` | `--epochs` (a safety cap; real stopping is `--patience`-based `EarlyStopping`, not a fixed count) | Test mode is `--test=True` + `--test_files`, not a separate script. |

`--seed` is the one flag that *is* spelled identically across every one of
these (all six model-free `train.py`/`evaluate.py` scripts and
`Model_Base_L2O/train.py`'s absl flag) — it's safe to assume without
checking; nothing else in this table is.

Beyond this table, don't guess: `core.shell.run_script` now validates every
flag your adapter sends against a live scan of the target script's own
`add_argument`/`DEFINE_*` calls (`core/shell.py`'s `discover_script_flags`)
and raises immediately, before any subprocess launches, naming the closest
real flag if you typo one. So a wrong guess fails fast with a fix suggestion
rather than silently doing nothing or crashing several layers down — but you
still have to open the script once to learn what's *available*, this table
just tells you which corner of it to open.

Worked example configs for every registered learned-optimizer method now
live in `configs/` — `dm_simple.json`, `rnnprop_simple.json`,
`entropy_hrnn_mnist.json`, `jacobian_hrnn_mnist.json`, `lista_lasso.json`,
`alista_lasso.json`, alongside the original `scale_hrnn_mnist.json` /
`swarm_quadratic.json` — so adding your own is a copy-and-edit of the closest
existing family's example, not a build-from-the-required-keys-table exercise.

## 10. Where to look for a worked example

Read one existing adapter end to end before writing your own — don't start
from `core/method.py`'s bare docstring alone:

- `adapters/dm_rnnprop.py` — simplest full example: subprocess train +
  subprocess evaluate + read back a pickle.
- `adapters/model_based.py` — example of bending a structurally different
  training loop (per-layer Keras `fit()`) onto the same trajectory contract;
  read its module docstring for how it justifies the "layer = iteration"
  mapping.
- `adapters/baselines.py` — the whole pattern for classical-optimizer
  support via `run_classical_optimizer_eval`, in ~30 lines.

## 11. Running a sweep of your method: `experiments/sweep.py`

Everything above is about making one `(method, problem)` run work. Once it
does, you usually want many of them — across seeds, problem sizes,
hyperparameters — with results in one table. That is a different job, and it
has its own shared piece: `experiments/sweep.py`.

**Copy `experiments/experiment_template.py` and edit it.** It is a real,
runnable sweep (LISTA on a small LASSO dataset, ~2 min on CPU) with four
marked places to change:

1. your config dataclass — your knobs, and *why* each value is what it is;
2. a `prepare` hook — setup shared by many cells, e.g. dataset generation;
3. `build_config` — your hyperparameters translated into your adapter's flags
   (the one irreducibly per-library part; see §9);
4. `run_cell` — what one grid point does, and what row(s) it returns.

You get, without writing any of it:

- **Resume.** Cells already in `results.jsonl` are skipped, so a sweep that
  dies at hour 30 of 40 does not restart from zero.
- **Error isolation.** A cell that raises lands in `errors.jsonl` with its
  traceback and the sweep continues. Pass `stop_on_error=True` while
  developing so you see the failure immediately instead of afterwards.
- **Provenance.** Commit, branch, working-tree status, a diff sidecar, and a
  sha256 of every untracked `.py`. Most of what a sweep here runs is
  uncommitted or untracked, so a commit hash alone does not identify the code
  that produced a number.
- **Correct axis fields.** The sweep driver stamps `seed`/your axis values into
  every row itself, so a table can never record a seed the run did not use.
- **Summaries.** Every numeric field in your rows is aggregated across seeds
  into `summary.json`; add a metric to a row and it appears there
  automatically. The statistics are a parameter, not a fixture — the default
  is mean and population std (`<field>_seed_mean` / `<field>_seed_std`), and
  you extend it rather than replace it:

  ```python
  from sweep import DEFAULT_AGGREGATORS
  run_sweep(..., aggregators={**DEFAULT_AGGREGATORS,
                              "seed_median": np.median, "seed_max": np.max})
  ```

  The dict key is the field-name suffix, so names and functions stay together
  and output order is insertion order. Spread `DEFAULT_AGGREGATORS` unless you
  really want mean/std gone — the repo's analysis scripts
  (`generate_result_tables.py` and friends) look those two up by name. An
  aggregator may return an int or an array (`np.percentile(v, [25, 75])`); it
  is coerced to something JSON can hold.
- **CLI flags derived from your dataclass**, via `add_overrides` /
  `apply_overrides` — instead of writing each tunable three times (field,
  `add_argument`, `setattr`) and having them drift apart.

The core vocabulary is small:

```python
from sweep import Axis, SweepSpec, run_sweep

spec = SweepSpec(
    results_dir=cfg.results_dir,
    axes=[Axis("seed", cfg.seeds), Axis("method", cfg.methods)],
    fanout_keys=("test_sparsity",),   # row fields that vary WITHIN one cell
    prepare=[(("seed",), make_dataset)],
    context={"experiment_config": cfg},
)
run_sweep(spec, run_cell, config_snapshot=cfg)
```

`Axis.values` reach `run_cell` unchanged (they can be tuples, e.g. `(25, 50)`);
`Axis.label` maps a value to how it appears in the row and resume key
(`"25x50"`). `fanout_keys` are for the common shape where one training run is
evaluated several ways: the sweep resumes at checkpoint granularity while the
table gets one row per evaluation.

**What the sweep driver deliberately does not own:** your config dataclass. Two
experiments holding the same value for different reasons are not duplication —
the reasons are the content. Keep configs per-experiment and keep their
comments; the sweep driver shares mechanics, not science.

For worked examples of each shape, read `experiments/v2/`: `experiment2.py`
(fan-out over test sparsities), `experiment1.py` (a tuple-valued axis with a
`label`, and a two-axis `prepare` prefix), `primer_replication.py` (train once
in an empty-prefix hook, then sweep evaluation replicates).

If anything in this doc turns out to be wrong or a built-in adapter's
behavior has changed, fix this file in the same change — it should stay a
description of the platform as it actually is, not as it was when written.

## 12. Making your adapter fairness-testable: `tests/`

`Benchmarking/tests/` is a pytest suite (`pip install pytest`; it is in
`requirements-tensorflow.txt`) that checks the two properties a benchmark
comparison rests on:

- **Determinism** (Tier A): the same `(method, problem, seed)` evaluated twice
  gives the same trajectory. Needs nothing from you beyond `EvalOutput.trajectory`.
- **Fairness** (Tier B): every method in a group was scored on the *same
  instances from the same starting points*. This needs one thing from your
  adapter: an **instances fingerprint**.

```bash
pytest Benchmarking/tests -m "not slow"          # seconds: data + contract tests
pytest Benchmarking/tests -m slow                 # minutes: live runs on a tiny fixture
pytest Benchmarking/tests --run-dir runs/<sweep>  # post hoc: check an existing sweep
```

### The fingerprint contract

In `evaluate()`, hash whatever identifies *what your script actually
evaluated*, in the order it evaluated it, and store the result under
`EvalOutput.metrics["instances_fingerprint"]`:

```python
from core.fingerprint import METRIC_KEY, fingerprint_arrays

rec = np.load(recovery_path)          # what your script wrote
metrics[METRIC_KEY] = fingerprint_arrays(b=rec["b"], x_true=rec["x_true"], x0=rec["x0"])
```

`fingerprint_arrays` takes any named arrays with a shared leading
`num_instances` dimension and returns `{"sha256", "num_instances",
"components"}`. It does not know what the arrays mean -- that is the point.
For LASSO the components are `b`, `x_true` and the starting point `x0`; for a
neural-network optimizee they might be the initial parameters and the data
batches consumed. Two adapters agree only if they hash the *same component
names* to the same digest, so pick names consistently across the adapters of
one problem. Hash what your script *observed* (its own output files), not what
you think it read: the model-free LASSO adapters hash the `b/x_true/x0` arrays
the eval script saved next to `x_pred`, which is how a script that walked the
split in the wrong order would be caught.

Because every experiment spreads `eval_out.metrics` into its result rows, the
fingerprint lands in `results.jsonl` automatically; `sweep.summarize` skips it
(non-numeric).

### Adding your problem to the matrix

`tests/fairness_matrix.py` is the only file that knows about problems. Add
one `Entry`:

```python
Entry(
    name="my_problem",
    make_fixture=make_my_tiny_fixture,   # (root_dir, seed) -> Fixture, or None
    methods={"my-method": build_my_config, "some-baseline": build_baseline_config},
    seed_selects_eval_instances=False,   # True if the eval seed picks the instances
    fingerprint_required=True,
)
```

Each builder returns the config your adapter needs (a `train` section means
"train first"; none means "evaluate directly" -- the cheap path for
classical baselines). Keep the fixture tiny: the live tests run every method
in the entry twice. If your problem has a data generator, give it the same
three checks `tests/test_dataset_generation.py` gives the LASSO one: same seed
-> identical bytes, different seed -> different data, changed config on an
existing directory -> refused.

Sweeps produced before this contract existed (Experiment 2's `runs/exp2_*`)
can be checked with `backfill_fingerprints_exp2.py --run <dir>`, which
recomputes fingerprints from their artifacts into a copy of the rows.
