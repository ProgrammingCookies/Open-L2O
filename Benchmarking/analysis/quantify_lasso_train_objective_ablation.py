"""Quantify the effect of ``Model_Base_L2O/utils.py``'s ``lasso_train_objective``
disparity on LISTA/ALISTA's actual Experiment-2 metrics.

Background (``unified/README.md``, "Faithful, not fixed"): LISTA/ALISTA's
curriculum training minimizes ``0.25*||Ax-b||^2 + lam*||x||_1`` (summed over
the batch, ``lasso_train_objective="paper"``, the default -- ``utils.LassoLoss``),
while every scoring path (recovery metrics, the FISTA x*, DM/RNNProp/Scale's
own meta-training objective) uses ``0.5*||Ax-b||^2 + lam*||x||_1`` per
instance. Minimizing ``0.25*R + lam*S`` is *exactly* equivalent to minimizing
``0.5*(0.5*R + 2*lam*S)``, i.e. the same minimizer as the true scoring
objective at ``2*lam`` -- training at "paper" is training at double the
nominal lambda, then scoring at 1x. This script measures how much that
actually moves the reported metrics.

Trains and evaluates ``lista``/``alista`` -- the SAME adapter-backed
implementation that produced ``exp2_try3_210x300``'s existing "paper" rows
(``Model_Base_L2O/train.py`` subprocess, not the ``unified`` port) -- with
``--lasso_train_objective=scoring --deterministic_data``. Only this "scoring" arm
is trained fresh here; ``exp2_try3_210x300``'s already-completed "paper" rows
for the same seeds are the other half of the comparison (read at analysis
time from ``runs/exp2_try3_210x300/results.jsonl``, not duplicated into this
results_dir). This is deliberately NOT a paired same-sweep design: using the
identical implementation for both arms (rather than unified-lista/alista, a
parity-verified but not bit-identical port) was judged the bigger source of
confound to eliminate, since it lets these rows reuse the exact code that
already produced exp2_try3_210x300 instead of introducing a second
implementation into the comparison.

``--deterministic_data`` (new flag on ``Model_Base_L2O/train.py``, off by
default elsewhere so old results stay reproducible under the old behavior)
fixes the training-batch shuffle order via an explicit seed + tf.data
deterministic mode -- removes shuffle-order as a noise source between reruns.
It does NOT achieve bit-exact reproducibility: a same-seed rerun still shows
~1e-3 magnitude weight differences from floating-point non-associativity in
TF's multi-threaded CPU kernels (oneDNN), tiny relative to the ~2x effective-
lambda effect being measured here, so not chased further.

Reuses ``experiments/thesis/experiment2_try2.py``'s ``OrchestratorConfig`` /
``_build_train_config`` / ``make_dataset`` (the exact protocol
``exp2_try3_210x300`` was run under -- (m,n)=(210,300), noiseless,
re-centred widths) -- same shape as ``unified/exp2_driver.py``'s
``run_cell``, just routed at the adapter-backed ``lista``/``alista`` methods
directly (``adapters``, not ``unified``) with two extra train-config keys.

Usage (from ``Benchmarking/``):
    python analysis/quantify_lasso_train_objective_ablation.py \\
        --results_dir runs/exp2_try3_210x300_scoringonly --smoke
    python analysis/quantify_lasso_train_objective_ablation.py \\
        --results_dir runs/exp2_try3_210x300_scoringonly \\
        --seeds 21 22 23 --widths narrow medium wide
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from typing import Any, Dict, List

_BENCHMARKING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXPERIMENTS_DIR = os.path.join(_BENCHMARKING_ROOT, "experiments")
_THESIS_DIR = os.path.join(_EXPERIMENTS_DIR, "thesis")
for _p in (_BENCHMARKING_ROOT, _EXPERIMENTS_DIR, _THESIS_DIR):
    while _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

import adapters  # noqa: E402,F401  (import side effect: registers lista/alista, matches exp2_try3_210x300)
from core.registry import get_method  # noqa: E402
from sweep import Axis, SweepSpec, run_sweep  # noqa: E402

experiment2_try2 = importlib.import_module("experiment2_try2")

METHODS = ("lista", "alista")


def run_cell(cfg, cell: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
    method_name, width, seed = cell["method"], cell["width"], cell["seed"]
    width_dir = os.path.join(context["seed_dir"], width)
    method = get_method(method_name)
    run_name = "{}__{}__s{}__scoring".format(method_name, width, seed)
    output_dir = os.path.join(context["checkpoints_dir"], "s{}".format(seed), width, method_name)
    train_config = experiment2_try2._build_train_config(method_name, cfg, width_dir, seed, run_name, output_dir)
    # The two lines this ablation exists for: everything else about
    # train_config is identical to what exp2_try3_210x300's own lista/alista
    # cells build (lasso_train_objective default "paper" there); deterministic_data
    # pairs this specific run's own noise realization as tightly as this repo
    # currently can against a repeat of itself (see module docstring).
    train_config["train"]["lasso_train_objective"] = "scoring"
    train_config["train"]["deterministic_data"] = True

    t0 = time.time()
    train_out = method.train(train_config, output_dir)
    train_sec = time.time() - t0
    xstar_cache_dir = os.path.join(context["results_dir"], "xstar_cache")

    eval_config = dict(train_config)
    eval_config["eval"] = {
        "data_dir": width_dir,
        "test_files": [experiment2_try2.test_filename(p) for p in cfg.test_sparsities],
        "num_fista_iters": cfg.num_fista_iters,
        "xstar_cache_dir": xstar_cache_dir,
    }
    t1 = time.time()
    eval_out = method.evaluate(eval_config, train_out.checkpoint, output_dir)
    eval_sec = time.time() - t1

    rows: List[Dict[str, Any]] = []
    for p in cfg.test_sparsities:
        metrics = eval_out.per_seed.get(experiment2_try2.test_filename(p), {})
        rows.append({"test_sparsity": p, "train_sec": train_sec, "eval_sec": eval_sec, **metrics})
    return rows


def run_experiment(cfg, widths: List[str], seeds: List[int], stop_on_error: bool = False) -> None:
    spec = SweepSpec(
        results_dir=cfg.results_dir,
        axes=[Axis("seed", seeds), Axis("width", widths), Axis("method", list(METHODS))],
        fanout_keys=("test_sparsity",),
        prepare=[(("seed",), experiment2_try2.make_dataset)],
        context={"experiment_config": cfg},
        stop_on_error=stop_on_error,
    )
    run_sweep(spec, lambda cell, ctx: run_cell(cfg, cell, ctx), config_snapshot=cfg,
              group_by=["method", "width", "test_sparsity"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_dir", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[21, 22, 23])
    p.add_argument("--widths", nargs="+", default=["narrow", "medium", "wide"],
                   choices=list(experiment2_try2.WIDTHS))
    p.add_argument("--smoke", action="store_true",
                   help="Tiny dims + training budget (experiment2_try2's own smoke config) to "
                        "sanity-check the pipeline, including the lasso_train_objective/deterministic_data "
                        "overrides. NOT a measurement of the effect itself.")
    p.add_argument("--stop_on_error", action="store_true")
    args = p.parse_args()

    if args.smoke:
        cfg = experiment2_try2._smoke_config(args.results_dir, args.seeds, list(METHODS), args.widths)
    else:
        cfg = experiment2_try2.OrchestratorConfig(
            results_dir=args.results_dir, seeds=args.seeds, methods=list(METHODS), widths=args.widths,
            m=210, n=300)   # noiseless is already OrchestratorConfig's own default (snr_db=inf)
    run_experiment(cfg, args.widths, args.seeds, stop_on_error=args.stop_on_error)


if __name__ == "__main__":
    main()
