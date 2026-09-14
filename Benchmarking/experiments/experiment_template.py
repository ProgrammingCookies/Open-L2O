"""Copy-and-edit starting point for a new experiment.

This is a real, runnable sweep, not pseudocode: it trains LISTA on a small
LASSO dataset across a grid of (seed, num_layers) and evaluates each trained
model at several test sparsities. It is deliberately the *smallest* thing that
still exercises every part of the pipeline, so you can run it once, look at
what it wrote, and then replace the parts that are about LISTA with the parts
that are about your method.

    python experiments/experiment_template.py --results_dir runs/template_demo

Two minutes on CPU with the defaults below. Add ``--seeds 4 5 6`` for more
repetitions; kill it halfway and re-run the same command to watch it resume.

WHAT IT WRITES (all of this is the sweep driver's doing, not this file's)

    runs/template_demo/
      results.jsonl     one row per (seed, num_layers, test_sparsity)
      summary.json      mean/std across seeds, per (num_layers, test_sparsity)
      config.json       your ExperimentConfig + the exact code state, once
      provenance.jsonl  one record per invocation (resumes append)
      errors.jsonl      any cell that raised, with its traceback
      data/             generated datasets
      checkpoints/      per-cell training output

THE FOUR PIECES YOU EDIT, IN ORDER

  1. ``ExperimentConfig`` -- your knobs. Keep the comments: say where each
     value came from (a paper, a measurement, a guess), because a number
     without a source is unusable to the next person and to you in a month.
  2. ``make_dataset``   -- a ``prepare`` hook. Runs once per seed, not once
     per cell, and its return value is merged into ``context``.
  3. ``build_config``   -- your hyperparameters translated into the flags your
     method's adapter wants. This is the part that is genuinely specific to
     one library and cannot be shared; see ``CONTRIBUTING.md`` §9 for which
     flag convention each family uses.
  4. ``run_cell``       -- what one grid point does, and what it returns.

Everything else in this file is four lines of wiring in ``main``.

WHAT THE SWEEP DRIVER GIVES YOU FOR FREE

  * **Resume.** Cells already in ``results.jsonl`` are skipped, so a sweep
    that dies at hour 30 of 40 does not restart from zero.
  * **Isolation.** A cell that raises is recorded in ``errors.jsonl`` and the
    sweep continues -- one OOM must not cost the other 200 runs. While
    developing, set ``stop_on_error=True`` so you see the traceback
    immediately instead of finding it afterwards.
  * **Provenance.** Commit, branch, working-tree diff, and a sha256 of every
    untracked ``.py``. Most of what a sweep runs here is uncommitted or
    untracked, so a commit hash alone does not identify the code that produced
    a number -- this is the difference between a results directory you can
    still interpret next month and one you cannot.
  * **Correct axis fields.** The sweep driver stamps ``seed``/``num_layers`` into
    each row itself, so a row can never claim a seed the run did not use.
  * **Summary.** Every numeric field in your rows is aggregated across seeds
    automatically; add a metric to your row and it appears in the summary
    without touching anything. Mean and std are only the default -- see the
    ``aggregators`` note at the ``run_sweep`` call below.

A NOTE ON WHAT NOT TO SHARE

Resist the urge to factor your config dataclass together with another
experiment's. Two experiments holding the same value for different reasons are
not duplication, and the reasons are the part worth keeping. The sweep driver
shares mechanics; the science stays per-file.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

# Make the Benchmarking platform importable regardless of the caller's cwd.
_BENCHMARKING_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BENCHMARKING_ROOT not in sys.path:
    sys.path.insert(0, _BENCHMARKING_ROOT)

import adapters  # noqa: F401,E402  (import side effect: registers all methods)
from core.registry import get_method  # noqa: E402
from data.lasso import Experiment2Config, generate_experiment2_dataset  # noqa: E402

from sweep import Axis, SweepSpec, add_overrides, apply_overrides, run_sweep  # noqa: E402

METHOD = "lista"


# --- 1. Your knobs ---------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Say where each value came from. "Small enough to finish on CPU" is a
    perfectly good source; "unknown" is the one answer that costs you later.
    """
    results_dir: str
    seeds: List[int] = field(default_factory=lambda: [0])

    #: The axis under study. Everything else is held fixed so that a
    #: difference between rows is attributable to this and nothing else.
    num_layers_values: List[int] = field(default_factory=lambda: [2, 4])

    # Dataset. Tiny on purpose -- this is a template, not a result.
    m: int = 20
    n: int = 30
    lam: float = 0.005
    train_size: int = 2_000
    val_size: int = 128
    test_size: int = 128
    snr_db: float = 40.0
    #: Sparsities to evaluate at. Several per trained model, hence the fan-out
    #: in run_cell below: one training run, several result rows.
    test_sparsities: List[float] = field(default_factory=lambda: [0.10, 0.175, 0.30])
    #: Which training distribution to train on. data/lasso.py writes one
    #: subdirectory per width ("narrow"/"medium"/"wide") under each seed.
    width: str = "narrow"

    # Training budget.
    epochs: int = 4
    base_lr: float = 5e-4
    train_batch_size: int = 64
    #: Safety cap on the reference FISTA solve for x*, not the real iteration
    #: count -- the solver stops on a certified duality gap well before this.
    num_fista_iters: int = 2_000


# --- 2. Setup that is shared by many cells ---------------------------------

def make_dataset(cell: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Generate this seed's dataset. Returns keys merged into ``context``.

    Registered against the ``("seed",)`` prefix in ``main``, so it runs once
    per seed rather than once per (seed, num_layers) cell -- which also means
    every num_layers value is compared on identical data.
    """
    cfg: ExperimentConfig = context["experiment_config"]
    seed_dir = generate_experiment2_dataset(
        Experiment2Config(
            seed=cell["seed"], m=cfg.m, n=cfg.n, lam=cfg.lam,
            train_size=cfg.train_size, val_size=cfg.val_size, test_size=cfg.test_size,
            snr_db=cfg.snr_db, test_sparsities=cfg.test_sparsities),
        context["data_dir"])
    print("[seed {}] dataset ready: {}".format(cell["seed"], seed_dir))
    return {"data_dir_for_seed": os.path.join(seed_dir, cfg.width)}


# --- 3. Hyperparameters -> this library's flags -----------------------------

def build_config(cfg: ExperimentConfig, cell: Dict[str, Any], data_dir: str,
                 output_dir: str) -> Dict[str, Any]:
    """The one genuinely per-library part. ``train``/``eval`` are raw
    pass-throughs to the adapter -- nothing validates their *values* for you,
    though ``core/shell.py`` does check flag names against the target script
    before launching it. See CONTRIBUTING.md §7.
    """
    return {
        "method": METHOD,
        "problem": "lasso",
        "seed": cell["seed"],
        "run_name": "{}__L{}__s{}".format(METHOD, cell["num_layers"], cell["seed"]),
        "output_dir": output_dir,
        "train": {
            "data_dir": data_dir,
            "num_layers": cell["num_layers"],
            "lasso_lam": cfg.lam,
            # model_lam == lasso_lam makes layer 0 exactly one ISTA step on the
            # actual objective, so an untrained net is ISTA rather than an
            # arbitrary starting point.
            "model_lam": cfg.lam,
            "epochs": cfg.epochs,
            "base_lr": cfg.base_lr,
            "train_batch_size": cfg.train_batch_size,
        },
    }


# --- 4. One grid point ------------------------------------------------------

def run_cell(cfg: ExperimentConfig, cell: Dict[str, Any],
             context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Train once, evaluate at every test sparsity, return one row per sparsity.

    Do NOT put ``seed``/``num_layers`` in the returned rows -- the sweep driver
    stamps those, which is what guarantees the row and the run agree.
    """
    data_dir = context["data_dir_for_seed"]
    output_dir = os.path.join(context["checkpoints_dir"],
                              "s{}".format(cell["seed"]), "L{}".format(cell["num_layers"]))
    config = build_config(cfg, cell, data_dir, output_dir)

    method = get_method(METHOD)
    t0 = time.time()
    train_out = method.train(config, output_dir)
    train_sec = time.time() - t0

    test_files = ["test_sparsity_{:.2f}.npy".format(p) for p in cfg.test_sparsities]
    config["eval"] = {
        "data_dir": data_dir,
        "test_files": test_files,
        "num_fista_iters": cfg.num_fista_iters,
        # x* depends only on (A, b, lam, iters), so every cell and every method
        # pointed at the same results_dir shares one solve.
        "xstar_cache_dir": os.path.join(context["results_dir"], "xstar_cache"),
    }
    t1 = time.time()
    eval_out = method.evaluate(config, train_out.checkpoint, output_dir)
    eval_sec = time.time() - t1

    rows = []
    for p, filename in zip(cfg.test_sparsities, test_files):
        # model_based.py keys per_seed by test filename; other adapters use it
        # differently (see CONTRIBUTING.md §5), so check yours rather than
        # copying this line.
        metrics = eval_out.per_seed.get(filename, {})
        rows.append({"test_sparsity": p, "train_sec": train_sec,
                     "eval_sec": eval_sec, **metrics})
    return rows


# --- Wiring -----------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_dir", required=True)
    # List-valued fields need bespoke parsing, so they are declared by hand;
    # add_overrides below derives a flag for every scalar field automatically.
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--num_layers_values", type=int, nargs="+", default=None)
    p.add_argument("--stop_on_error", action="store_true",
                   help="Re-raise instead of writing errors.jsonl. Use while developing.")
    scalar_fields = add_overrides(p, ExperimentConfig)
    args = p.parse_args()

    cfg = ExperimentConfig(results_dir=args.results_dir)
    apply_overrides(cfg, args, scalar_fields)
    for name in ("seeds", "num_layers_values"):
        if getattr(args, name) is not None:
            setattr(cfg, name, getattr(args, name))

    spec = SweepSpec(
        results_dir=cfg.results_dir,
        # Outermost axis first. Put the expensive-to-change one outermost so
        # the prepare hooks below fire as rarely as possible.
        axes=[Axis("seed", cfg.seeds), Axis("num_layers", cfg.num_layers_values)],
        # One trained model produces several rows differing only in this, so it
        # is a row field but not part of the resume key.
        fanout_keys=("test_sparsity",),
        prepare=[(("seed",), make_dataset)],
        # Anything the hooks or run_cell need beyond the axes rides here.
        context={"experiment_config": cfg},
        stop_on_error=args.stop_on_error,
    )
    # summary.json reports mean and population std across seeds for every
    # numeric field. If the mean is the wrong summary for what you are
    # studying -- worst case across seeds, a median that ignores one diverged
    # run -- add to the defaults rather than replacing them:
    #
    #     from sweep import DEFAULT_AGGREGATORS
    #     run_sweep(..., aggregators={**DEFAULT_AGGREGATORS, "seed_max": np.max})
    run_sweep(spec, lambda cell, ctx: run_cell(cfg, cell, ctx), config_snapshot=cfg)


if __name__ == "__main__":
    main()
