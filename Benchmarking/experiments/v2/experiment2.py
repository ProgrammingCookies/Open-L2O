"""
Experiment 2 main file, ported onto ``experiments/sweep.py``.

Runs the full sweep over methods L2O-DM, L2O-RNNProp, LISTA, ALISTA. (check pre-study table)

This is a 1:1 port of ``experiments/experiment2.py``: every hyperparameter,
every flag translation, and every metric is unchanged, and a ``--smoke`` run
produces the same result rows. What changed is only the plumbing -- the grid
loop, resume, error capture, config snapshot and summarization now come from
``sweep.py`` instead of being written out here, which is what removes the
~420 lines this file used to share verbatim with ``experiment2_try2.py``.

Two visible differences from the pre-port file, both deliberate:

  * Provenance is now the full working-tree capture (status, a diff sidecar,
    sha256 of every untracked .py) rather than a bare ``git rev-parse HEAD``.
    That is the whole reason ``runs/exp2_real``'s config.json had to be
    reconstructed by hand; the sweep driver gives every experiment the good version.
  * Rows carry an extra ``cell_sec`` field (wall time for the whole
    train+eval cell). It is bookkeeping, not a metric, so it never reaches
    summary.json.

Resume is compatible with results.jsonl files written by the pre-port
``experiment2.py``: the resume key is read out of each row's own
method/width/seed fields, so a run started there can be finished here.

Usage:
    python experiments/v2/experiment2.py --results_dir runs/exp2_smoke --seeds 0 --smoke
    python experiments/v2/experiment2.py --results_dir runs/exp2 --seeds 0 1 2 3 4 5 6 7 8 9
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

_V2_DIR = os.path.dirname(os.path.abspath(__file__))
_EXPERIMENTS_DIR = os.path.dirname(_V2_DIR)
_BENCHMARKING_ROOT = os.path.dirname(_EXPERIMENTS_DIR)
# Order matters and must be enforced, not assumed. Python puts the script's own
# directory (v2/) on sys.path automatically, but a plain insert(0, ...) of the
# others pushes it back -- and since these files share their names with the
# pre-port originals one level up, `import experiment2` would then silently
# resolve to experiments/experiment2.py instead of this directory's. Removing
# each path before re-inserting makes the final order exactly V2 > experiments
# > Benchmarking, whatever was there before.
for _p in (_BENCHMARKING_ROOT, _EXPERIMENTS_DIR, _V2_DIR):
    while _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

import adapters  # noqa: F401,E402  (import side effect: registers all methods)
from core.registry import get_method  # noqa: E402
from data.lasso import (  # noqa: E402
    DEFAULT_TEST_SPARSITIES, Experiment2Config, generate_experiment2_dataset)

from sweep import Axis, SweepSpec, add_overrides, apply_overrides, run_sweep  # noqa: E402

# "l2o-scale" is part of the shared vocabulary (experiment2_try2.py imports
# ALL_METHODS/method_kind from here and does wire it up) but this file's own
# _build_train_config has no L2O-Scale branch -- see the guard there.
MODEL_FREE_METHODS = ("l2o-dm", "l2o-rnnprop", "l2o-scale")
MODEL_BASED_METHODS = ("lista", "alista")
ALL_METHODS = MODEL_FREE_METHODS + MODEL_BASED_METHODS
WIDTHS = ("narrow", "medium", "wide")


def method_kind(method_name: str) -> str:
    if method_name in MODEL_FREE_METHODS:
        return "model_free"
    if method_name in MODEL_BASED_METHODS:
        return "model_based"
    raise ValueError("unknown method {!r} (expected one of {})".format(method_name, ALL_METHODS))


def test_filename(p: float) -> str:
    return "test_sparsity_{:.2f}.npy".format(p)


@dataclass
class OrchestratorConfig:
    """Every value below is a deliberate decision, not a script default left
    untouched.
    """
    results_dir: str
    seeds: List[int]
    methods: List[str] = field(default_factory=lambda: list(ALL_METHODS))
    widths: List[str] = field(default_factory=lambda: list(WIDTHS))

    # Table 3 dataset parameters.
    m: int = 210
    n: int = 300
    lam: float = 0.005
    train_size: int = 32_000
    val_size: int = 1_024
    test_size: int = 1_280
    test_sparsities: List[float] = field(default_factory=lambda: list(DEFAULT_TEST_SPARSITIES))
    snr_db: float = 40.0

    # Model-free (L2O-DM / L2O-RNNProp) training budget.
    model_free_num_epochs: int = 100
    model_free_num_steps: int = 1_000
    model_free_unroll_length: int = 20
    model_free_batch_size: int = 128
    model_free_eval_num_steps: int = 1000
    model_free_lr: float = 1e-3
    # Periodic best-checkpoint-selection cadence during meta-training.
    model_free_evaluation_period: int = 5
    model_free_evaluation_epochs: int = 20
    # RNNProp-only: decay rates for its internal Adam-style input
    # normalization.
    model_free_rnnprop_beta1: float = 0.95
    model_free_rnnprop_beta2: float = 0.95
    # RNNProp-only: train against the final unroll step's loss only (w_T=1,
    # w_t=0 otherwise).
    model_free_rnnprop_last_step_loss: bool = False
    # Curriculum learning (--if_cl) and imitation learning (--if_mt), Chen et
    # al. 2020 "Training Stronger Baselines for Learning to Optimize.
    model_free_curriculum: bool = False
    model_free_imitation: bool = False
    # Writes a per-epoch (loss, meta-gradient-norm) trajectory to
    # <output_dir>/profile.jsonl (train_dm.py/train_rnnprop.py's own
    # profiling.RunProfiler, already used by experiment1.py -- off by
    # default here since Experiment 2 doesn't otherwise need it).
    model_free_profile: bool = False

    # Model-based (LISTA/ALISTA)
    model_based_num_layers: int = 16
    model_based_base_lr: float = 5e-4
    model_based_train_batch_size: int = 128
    model_based_epochs: int = 200
    # ALISTA support-selection schedule (models/alista.py)
    model_based_ss_q_per_layer: float = 1.2
    model_based_ss_maxq: float = 13.0

    # Reference x* solve for the recovery metrics (core/lasso_metrics.py).
    num_fista_iters: int = 50_000

    def __post_init__(self):
        for m in self.methods:
            method_kind(m)  # raises on an unknown name
        for w in self.widths:
            if w not in WIDTHS:
                raise ValueError("unknown width {!r} (expected one of {})".format(w, WIDTHS))


def _build_train_config(method_name: str, cfg: OrchestratorConfig, width_dir: str,
                        seed: int, run_name: str, output_dir: str) -> Dict[str, Any]:
    if method_name == "l2o-scale":
        raise ValueError(
            "l2o-scale is only wired into experiment2_try2.py (its train.py/evaluate.py "
            "CLI differs from DM/RNNProp's and needs the scale_* config fields defined "
            "there); run it with `experiment2_try2.py --m 210 --n 300 ...` for these dims.")
    if method_kind(method_name) == "model_free":
        train_flags: Dict[str, Any] = {
            "lasso_data_dir": width_dir,
            "lasso_lam": cfg.lam,
            "lasso_batch_size": cfg.model_free_batch_size,
            "num_epochs": cfg.model_free_num_epochs,
            "num_steps": cfg.model_free_num_steps,
            "unroll_length": cfg.model_free_unroll_length,
            "learning_rate": cfg.model_free_lr,
            "evaluation_period": cfg.model_free_evaluation_period,
            "evaluation_epochs": cfg.model_free_evaluation_epochs,
        }
        if method_name == "l2o-rnnprop":
            # train_dm.py has no --beta1/--beta2 flags at all (argparse would
            # error on an unrecognized one), so this must stay conditional --
            # unlike model-based's ss_q_per_layer/ss_maxq (absl flags, always
            # defined, harmless no-op for the model that doesn't use them).
            train_flags["beta1"] = cfg.model_free_rnnprop_beta1
            train_flags["beta2"] = cfg.model_free_rnnprop_beta2
            if cfg.model_free_rnnprop_last_step_loss:
                train_flags["last_step_loss"] = True
        if cfg.model_free_curriculum:
            train_flags["if_cl"] = True
        if cfg.model_free_imitation:
            train_flags["if_mt"] = True
        if cfg.model_free_profile:
            train_flags["profile_path"] = os.path.join(output_dir, "profile.jsonl")
        return {
            "method": method_name,
            "problem": "lasso_dataset",
            "seed": seed,
            "run_name": run_name,
            "output_dir": output_dir,
            "train": train_flags,
        }
    return {
        "method": method_name,
        "problem": "lasso",
        "seed": seed,
        "run_name": run_name,
        "output_dir": output_dir,
        "train": {
            "data_dir": width_dir,
            "num_layers": cfg.model_based_num_layers,
            "lasso_lam": cfg.lam,
            "epochs": cfg.model_based_epochs,
            "base_lr": cfg.model_based_base_lr,
            "train_batch_size": cfg.model_based_train_batch_size,
            "model_lam": cfg.lam,
            "ss_q_per_layer": cfg.model_based_ss_q_per_layer,
            "ss_maxq": cfg.model_based_ss_maxq,
        },
    }


def run_cell(cfg: OrchestratorConfig, cell: Dict[str, Any],
             context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Train once, evaluate on every test sparsity, return one row per sparsity.

    method/width/seed are stamped into the rows by the sweep driver, so they are
    deliberately absent here.
    """
    method_name, width, seed = cell["method"], cell["width"], cell["seed"]
    width_dir = os.path.join(context["seed_dir"], width)
    method = get_method(method_name)
    run_name = "{}__{}__s{}".format(method_name, width, seed)
    output_dir = os.path.join(context["checkpoints_dir"], "s{}".format(seed), width, method_name)
    train_config = _build_train_config(method_name, cfg, width_dir, seed, run_name, output_dir)

    t0 = time.time()
    train_out = method.train(train_config, output_dir)
    train_sec = time.time() - t0

    # x* (calculated with FISTA like done in primer and benchmark too) depends only on (A, b, lam,
    # num_fista_iters) so we can cache it across all test sparsities and all methods.
    xstar_cache_dir = os.path.join(context["results_dir"], "xstar_cache")

    rows: List[Dict[str, Any]] = []
    if method_kind(method_name) == "model_free":
        for p in cfg.test_sparsities:
            eval_config = dict(train_config)
            eval_config["eval"] = {
                "optimizer": "L2L",
                "lasso_data_dir": width_dir,
                "lasso_split": test_filename(p),
                "lasso_batch_size": cfg.model_free_batch_size,
                "num_steps": cfg.model_free_eval_num_steps,
                # Must match train()'s lasso_lam, one for
                # all three, so the metrics x*/f* stay comparable to all.
                "lasso_lam": cfg.lam,
                "num_fista_iters": cfg.num_fista_iters,
                "xstar_cache_dir": xstar_cache_dir,
            }
            if method_name == "l2o-rnnprop":
                # Must match train()'s beta1/beta2, a mismatch here would apply the checkpoint
                # through a differently-configured normalization scheme.
                eval_config["eval"]["beta1"] = cfg.model_free_rnnprop_beta1
                eval_config["eval"]["beta2"] = cfg.model_free_rnnprop_beta2
            eval_output_dir = os.path.join(output_dir, "eval_p{:.2f}".format(p))
            t1 = time.time()
            eval_out = method.evaluate(eval_config, train_out.checkpoint, eval_output_dir)
            eval_sec = time.time() - t1
            rows.append({"test_sparsity": p, "train_sec": train_sec, "eval_sec": eval_sec,
                         **eval_out.metrics})
    else:
        eval_config = dict(train_config)
        eval_config["eval"] = {
            "data_dir": width_dir,
            "test_files": [test_filename(p) for p in cfg.test_sparsities],
            "num_fista_iters": cfg.num_fista_iters,
            "xstar_cache_dir": xstar_cache_dir,
        }
        t1 = time.time()
        eval_out = method.evaluate(eval_config, train_out.checkpoint, output_dir)
        eval_sec = time.time() - t1
        for p in cfg.test_sparsities:
            metrics = eval_out.per_seed.get(test_filename(p), {})
            rows.append({"test_sparsity": p, "train_sec": train_sec, "eval_sec": eval_sec,
                         **metrics})
    return rows


def make_dataset(cell: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """One dataset per seed, shared by every (width, method) cell under it."""
    cfg: OrchestratorConfig = context["experiment_config"]
    data_cfg = Experiment2Config(
        seed=cell["seed"], m=cfg.m, n=cfg.n, lam=cfg.lam,
        train_size=cfg.train_size, val_size=cfg.val_size, test_size=cfg.test_size,
        snr_db=cfg.snr_db, test_sparsities=cfg.test_sparsities,
        compute_alista_w="alista" in cfg.methods,
    )
    seed_dir = generate_experiment2_dataset(data_cfg, context["data_dir"])
    print("[seed {}] dataset ready: {}".format(cell["seed"], seed_dir))
    return {"seed_dir": seed_dir}


def build_spec(cfg: OrchestratorConfig, stop_on_error: bool = False) -> SweepSpec:
    """The sweep shape. Split out so scale_diagnostic.py can reuse it."""
    return SweepSpec(
        results_dir=cfg.results_dir,
        # seed outermost so each dataset is generated once and then reused by
        # every (width, method) cell under it -- same order as the original's
        # nested loops.
        axes=[Axis("seed", cfg.seeds), Axis("width", cfg.widths),
              Axis("method", cfg.methods)],
        # One training run is evaluated at every test sparsity, so the sweep
        # resumes at checkpoint granularity while the table has one row per
        # sparsity.
        fanout_keys=("test_sparsity",),
        prepare=[(("seed",), make_dataset)],
        context={"experiment_config": cfg},
        stop_on_error=stop_on_error,
    )


def run_experiment(cfg: OrchestratorConfig, stop_on_error: bool = False) -> None:
    run_sweep(build_spec(cfg, stop_on_error),
              lambda cell, ctx: run_cell(cfg, cell, ctx),
              config_snapshot=cfg,
              # Explicit rather than the sweep driver default (which would order
              # these width-first), so summary.json keeps the pre-port
              # file's own (method, width, test_sparsity) grouping order.
              group_by=["method", "width", "test_sparsity"])


def _smoke_config(results_dir: str, seeds: List[int], methods: List[str],
                  widths: List[str]) -> OrchestratorConfig:
    return OrchestratorConfig(
        results_dir=results_dir, seeds=seeds, methods=methods, widths=widths,
        m=20, n=30, train_size=200, val_size=64, test_size=32,
        test_sparsities=[0.1, 0.3], snr_db=40.0,
        model_free_num_epochs=3, model_free_num_steps=10, model_free_unroll_length=5,
        model_free_batch_size=16, model_free_eval_num_steps=10,
        model_based_num_layers=2, model_based_epochs=2,
        num_fista_iters=50,
    )


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_dir", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--methods", nargs="+", default=list(ALL_METHODS), choices=list(ALL_METHODS))
    p.add_argument("--widths", nargs="+", default=list(WIDTHS), choices=list(WIDTHS))
    p.add_argument("--smoke", action="store_true",
                   help="Tiny dims + training budgets to sanity-check the whole "
                        "pipeline in well under a minute. NOT a real Experiment-2 run.")
    p.add_argument("--stop_on_error", action="store_true",
                   help="Re-raise a failing cell instead of recording it in "
                        "errors.jsonl and continuing. For development.")
    # Every remaining scalar field of OrchestratorConfig becomes a --flag
    # automatically. The pre-port file hand-listed these and only ever wired
    # 6 of ~25, so most --flags a reader expected did not exist.
    # Ignored under --smoke, which defines its own tiny config.
    scalar_fields = add_overrides(p, OrchestratorConfig, help_texts={
        "model_free_curriculum": "Enable train_dm.py/train_rnnprop.py's --if_cl "
                                 "(curriculum learning on unroll length).",
        "model_free_imitation": "Enable train_dm.py/train_rnnprop.py's --if_mt "
                                "(imitation learning from analytical optimizers).",
        "model_free_profile": "Write a per-epoch (loss, meta-grad-norm) trajectory "
                              "to <output_dir>/profile.jsonl.",
        "model_free_rnnprop_last_step_loss":
            "RNNProp only: train against the final unroll step's loss only "
            "(w_T=1, w_t=0 otherwise), matching Lv, Jiang & Li 2017's own stated "
            "training objective, instead of this script's prior default of summing "
            "the loss over every step (DM's convention).",
    })
    return p.parse_args(), scalar_fields


def main() -> None:
    args, scalar_fields = _parse_args()
    if args.smoke:
        cfg = _smoke_config(args.results_dir, args.seeds, args.methods, args.widths)
    else:
        cfg = OrchestratorConfig(results_dir=args.results_dir, seeds=args.seeds,
                                 methods=args.methods, widths=args.widths)
        apply_overrides(cfg, args, scalar_fields)
    run_experiment(cfg, stop_on_error=args.stop_on_error)


if __name__ == "__main__":
    main()
