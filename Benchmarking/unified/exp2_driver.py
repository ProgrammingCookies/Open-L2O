"""Experiment 2 through the unified in-process methods.

``experiments/v2/experiment2.py`` only accepts its own method names
(``argparse choices`` + ``method_kind``), and that file is frozen while sweeps
run against it. This driver reuses its config dataclass, tiny smoke config,
dataset preparation and ``_build_train_config`` unchanged, and swaps the
method name for the unified one just before ``get_method``. ``--base
experiment2_try2`` selects the sibling experiment script instead (the one
that wires L2O-Scale: ``unified-scale`` requires it).

    cd Benchmarking
    python -m unified.exp2_driver --results_dir <dir> --seeds 0 --methods unified-rnnprop --smoke
    python -m unified.exp2_driver --base experiment2_try2 --results_dir <dir> --seeds 0 --methods unified-scale --smoke

Run as a module (``-m``), never as ``python unified/exp2_driver.py``: the
latter would put ``unified/`` itself at ``sys.path[0]``.

``run_cell`` below is a deliberate fork of ``experiment2.run_cell`` (same
config shape, one row per test sparsity); keep the two in sync.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List

_BENCHMARKING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXPERIMENTS_DIR = os.path.join(_BENCHMARKING_ROOT, "experiments")
_V2_DIR = os.path.join(_EXPERIMENTS_DIR, "v2")
for _p in (_BENCHMARKING_ROOT, _EXPERIMENTS_DIR, _V2_DIR):   # same order as experiment2.py
    while _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

import importlib  # noqa: E402

import unified  # noqa: E402,F401
from core.registry import get_method  # noqa: E402
from sweep import Axis, SweepSpec, run_sweep  # noqa: E402

UNIFIED_TO_BASE = {"unified-dm": "l2o-dm", "unified-rnnprop": "l2o-rnnprop", "unified-scale": "l2o-scale",
                   "unified-lista": "lista", "unified-alista": "alista",
                   # no experiment script knows LFISTA; it takes LISTA's config shape (same keys)
                   "unified-lfista": "lista"}
BASES = ("experiment2", "experiment2_try2")
experiment2 = importlib.import_module("experiment2")   # the default base; imports adapters itself

# ``OrchestratorConfig`` validates its ``methods`` against experiment2's own
# names, so the config always carries the BASE names (and that is what the
# config snapshot records); the unified names live on the sweep's method axis.


def run_cell(cfg, cell: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
    base = context["base_module"]
    method_name, width, seed = cell["method"], cell["width"], cell["seed"]
    base_name = UNIFIED_TO_BASE[method_name]
    width_dir = os.path.join(context["seed_dir"], width)
    method = get_method(method_name)
    run_name = "{}__{}__s{}".format(method_name, width, seed)
    output_dir = os.path.join(context["checkpoints_dir"], "s{}".format(seed), width, method_name)
    train_config = base._build_train_config(base_name, cfg, width_dir, seed, run_name, output_dir)
    train_config["method"] = method_name
    if method_name == "unified-lfista":
        # The platform's fairness contract is the seeded x0; the library's LFISTA
        # starts from x = 0 (see unrolled/lfista.py), so a sweep uses the fair variant.
        train_config["train"]["x0_at_layer0"] = True

    t0 = time.time()
    train_out = method.train(train_config, output_dir)
    train_sec = time.time() - t0
    xstar_cache_dir = os.path.join(context["results_dir"], "xstar_cache")

    rows: List[Dict[str, Any]] = []
    if experiment2.method_kind(base_name) == "model_free":
        for p in cfg.test_sparsities:
            eval_config = dict(train_config)
            if base_name == "l2o-scale":     # experiment2_try2.run_cell's Scale eval dict
                eval_config["eval"] = {
                    "test_optimizer": "L2o", "num_testing_itrs": cfg.model_free_eval_num_steps,
                    "lasso_data_dir": width_dir, "lasso_split": experiment2.test_filename(p),
                    "lasso_batch_size": cfg.model_free_batch_size, "lasso_lam": cfg.lam,
                    "lasso_x0_mode": "aligned", "min_lr": cfg.scale_min_lr, "max_lr": cfg.scale_max_lr,
                    "num_fista_iters": cfg.num_fista_iters, "xstar_cache_dir": xstar_cache_dir,
                }
            else:
                eval_config["eval"] = {
                    "optimizer": "L2L", "lasso_data_dir": width_dir,
                    "lasso_split": experiment2.test_filename(p),
                    "lasso_batch_size": cfg.model_free_batch_size,
                    "num_steps": cfg.model_free_eval_num_steps, "lasso_lam": cfg.lam,
                    "num_fista_iters": cfg.num_fista_iters, "xstar_cache_dir": xstar_cache_dir,
                }
                if base_name == "l2o-rnnprop":
                    eval_config["eval"]["beta1"] = cfg.model_free_rnnprop_beta1
                    eval_config["eval"]["beta2"] = cfg.model_free_rnnprop_beta2
            t1 = time.time()
            eval_out = method.evaluate(eval_config, train_out.checkpoint,
                                       os.path.join(output_dir, "eval_p{:.2f}".format(p)))
            rows.append({"test_sparsity": p, "train_sec": train_sec, "eval_sec": time.time() - t1,
                         **eval_out.metrics})
    else:
        eval_config = dict(train_config)
        eval_config["eval"] = {
            "data_dir": width_dir,
            "test_files": [experiment2.test_filename(p) for p in cfg.test_sparsities],
            "num_fista_iters": cfg.num_fista_iters, "xstar_cache_dir": xstar_cache_dir,
        }
        t1 = time.time()
        eval_out = method.evaluate(eval_config, train_out.checkpoint, output_dir)
        eval_sec = time.time() - t1
        for p in cfg.test_sparsities:
            rows.append({"test_sparsity": p, "train_sec": train_sec, "eval_sec": eval_sec,
                         **eval_out.per_seed.get(experiment2.test_filename(p), {})})
    return rows


def run_experiment(cfg, methods: List[str], base=experiment2, stop_on_error: bool = False) -> None:
    """``cfg.methods`` holds the base names (dataset prep reads it); ``methods`` the unified ones;
    ``base`` is the experiment module whose config/builders are reused."""
    assert [UNIFIED_TO_BASE[m] for m in methods] == list(cfg.methods)
    spec = SweepSpec(
        results_dir=cfg.results_dir,
        axes=[Axis("seed", cfg.seeds), Axis("width", cfg.widths), Axis("method", methods)],
        fanout_keys=("test_sparsity",),
        prepare=[(("seed",), base.make_dataset)],
        context={"experiment_config": cfg, "base_module": base},
        stop_on_error=stop_on_error,
    )
    run_sweep(spec, lambda cell, ctx: run_cell(cfg, cell, ctx), config_snapshot=cfg,
              group_by=["method", "width", "test_sparsity"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_dir", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--methods", nargs="+", default=list(UNIFIED_TO_BASE), choices=list(UNIFIED_TO_BASE))
    p.add_argument("--widths", nargs="+", default=list(experiment2.WIDTHS), choices=list(experiment2.WIDTHS))
    p.add_argument("--smoke", action="store_true", help="the base experiment's tiny smoke config")
    p.add_argument("--stop_on_error", action="store_true")
    p.add_argument("--base", choices=BASES, default=None,
                   help="experiment script to reuse (default experiment2; experiment2_try2 is the one "
                        "that wires L2O-Scale and is selected automatically for unified-scale)")
    known, _ = p.parse_known_args()
    base_name = known.base or ("experiment2_try2" if "unified-scale" in known.methods else "experiment2")
    base = importlib.import_module(base_name)
    if "unified-scale" in known.methods and base_name != "experiment2_try2":
        p.error("unified-scale needs --base experiment2_try2 (experiment2.py has no Scale branch)")
    scalar_fields = experiment2.add_overrides(p, base.OrchestratorConfig)
    args = p.parse_args()
    base_methods = [UNIFIED_TO_BASE[m] for m in args.methods]
    if args.smoke:
        cfg = base._smoke_config(args.results_dir, args.seeds, base_methods, args.widths)
    else:
        cfg = base.OrchestratorConfig(results_dir=args.results_dir, seeds=args.seeds,
                                      methods=base_methods, widths=args.widths)
        experiment2.apply_overrides(cfg, args, scalar_fields)
    run_experiment(cfg, args.methods, base=base, stop_on_error=args.stop_on_error)


if __name__ == "__main__":
    main()
