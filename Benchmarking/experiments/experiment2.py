"""
Experiment 2 main file. (It's a standalone file so if you're trying to understand the benchmarking platform's generality, you can skip it.)

Runs the full sweep over methods L2O-DM, L2O-RNNProp, LISTA, ALISTA. (check pre-study table)

TODO: Write more detailed description of the experiment here maybe?

Usage:
    python experiment2.py --results_dir runs/exp2_smoke --seeds 0 --smoke
    python experiment2.py --results_dir runs/exp2 --seeds 0 1 2 3 4 5 6 7 8 9
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np

_BENCHMARKING_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BENCHMARKING_ROOT not in sys.path:
    sys.path.insert(0, _BENCHMARKING_ROOT)

import adapters
from core.registry import get_method
from data.lasso import DEFAULT_TEST_SPARSITIES, Experiment2Config, generate_experiment2_dataset

MODEL_FREE_METHODS = ("l2o-dm", "l2o-rnnprop")
MODEL_BASED_METHODS = ("lista", "alista")
ALL_METHODS = MODEL_FREE_METHODS + MODEL_BASED_METHODS
WIDTHS = ("narrow", "medium", "wide")

_NON_METRIC_ROW_KEYS = ("method", "width", "seed", "test_sparsity", "train_sec", "eval_sec")


def method_kind(method_name: str) -> str:
    if method_name in MODEL_FREE_METHODS:
        return "model_free"
    if method_name in MODEL_BASED_METHODS:
        return "model_based"
    raise ValueError("unknown method {!r} (expected one of {})".format(method_name, ALL_METHODS))


def _test_filename(p: float) -> str:
    return "test_sparsity_{:.2f}.npy".format(p)


@dataclass
class OrchestratorConfig:
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
    model_free_eval_num_steps: int = 200
    # Placeholder values TODO: Decide
    model_free_lr: float = 5e-4
    model_based_num_layers: int = 16
    model_based_epochs: int = 200

    # "2000 iterations of FISTA" like primer and benchmark uses.
    num_fista_iters: int = 2000

    def __post_init__(self):
        for m in self.methods:
            method_kind(m)  # raises on an unknown name
        for w in self.widths:
            if w not in WIDTHS:
                raise ValueError("unknown width {!r} (expected one of {})".format(w, WIDTHS))


def _build_train_config(method_name: str, cfg: OrchestratorConfig, width_dir: str,
                        seed: int, run_name: str, output_dir: str) -> Dict[str, Any]:
    if method_kind(method_name) == "model_free":
        return {
            "method": method_name,
            "problem": "lasso_dataset",
            "seed": seed,
            "run_name": run_name,
            "output_dir": output_dir,
            "train": {
                "lasso_data_dir": width_dir,
                "lasso_lam": cfg.lam,
                "lasso_batch_size": cfg.model_free_batch_size,
                "num_epochs": cfg.model_free_num_epochs,
                "num_steps": cfg.model_free_num_steps,
                "unroll_length": cfg.model_free_unroll_length,
                "learning_rate": cfg.model_free_lr,
            },
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
        },
    }


def _make_row(method_name: str, width: str, seed: int, sparsity: float,
             metrics: Dict[str, Any], train_sec: float, eval_sec: float) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "method": method_name, "width": width, "seed": seed, "test_sparsity": sparsity,
        "train_sec": train_sec, "eval_sec": eval_sec,
    }
    row.update(metrics)
    return row


def run_one(method_name: str, config: OrchestratorConfig, width_dir: str, seed: int,
           width: str, results_root: str) -> List[Dict[str, Any]]:
    """Train once, evaluate on every test sparsity, return one row per sparsity."""
    method = get_method(method_name)
    run_name = "{}__{}__s{}".format(method_name, width, seed)
    output_dir = os.path.join(results_root, "s{}".format(seed), width, method_name)
    config = _build_train_config(method_name, config, width_dir, seed, run_name, output_dir)

    t0 = time.time()
    train_out = method.train(config, output_dir)
    train_sec = time.time() - t0

    # x* (calculated with FISTA like done in primer and benchmark too) depends only on (A, b, lam,
    # num_fista_iters) so we can cache it across all test sparsities and all methods.
    xstar_cache_dir = os.path.join(config.results_dir, "xstar_cache")

    rows: List[Dict[str, Any]] = []
    if method_kind(method_name) == "model_free":
        for p in config.test_sparsities:
            eval_config = dict(config)
            eval_config["eval"] = {
                "optimizer": "L2L",
                "lasso_data_dir": width_dir,
                "lasso_split": _test_filename(p),
                "lasso_batch_size": config.model_free_batch_size,
                "num_steps": config.model_free_eval_num_steps,
                # Must match train()'s lasso_lam, one for
                # all three, so the metrics x*/f* stay comparable to all.
                "lasso_lam": config.lam,
                "num_fista_iters": config.num_fista_iters,
                "xstar_cache_dir": xstar_cache_dir,
            }
            eval_output_dir = os.path.join(output_dir, "eval_p{:.2f}".format(p))
            t1 = time.time()
            eval_out = method.evaluate(eval_config, train_out.checkpoint, eval_output_dir)
            eval_sec = time.time() - t1
            rows.append(_make_row(method_name, width, seed, p, eval_out.metrics, train_sec, eval_sec))
    else:
        eval_config = dict(config)
        eval_config["eval"] = {
            "data_dir": width_dir,
            "test_files": [_test_filename(p) for p in config.test_sparsities],
            "num_fista_iters": config.num_fista_iters,
            "xstar_cache_dir": xstar_cache_dir,
        }
        t1 = time.time()
        eval_out = method.evaluate(eval_config, train_out.checkpoint, output_dir)
        eval_sec = time.time() - t1
        for p in config.test_sparsities:
            metrics = eval_out.per_seed.get(_test_filename(p), {})
            rows.append(_make_row(method_name, width, seed, p, metrics, train_sec, eval_sec))
    return rows


def _load_completed(results_path: str) -> set:
    completed = set()
    if os.path.exists(results_path):
        with open(results_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                completed.add((d["method"], d["width"], d["seed"]))
    return completed


def run_sweep(cfg: OrchestratorConfig) -> None:
    cfg.results_dir = os.path.abspath(cfg.results_dir)
    os.makedirs(cfg.results_dir, exist_ok=True)
    results_path = os.path.join(cfg.results_dir, "results.jsonl")
    errors_path = os.path.join(cfg.results_dir, "errors.jsonl")
    completed = _load_completed(results_path)

    data_root = os.path.join(cfg.results_dir, "data")
    checkpoints_root = os.path.join(cfg.results_dir, "checkpoints")
    needs_alista_w = "alista" in cfg.methods

    for seed in cfg.seeds:
        data_cfg = Experiment2Config(
            seed=seed, m=cfg.m, n=cfg.n, lam=cfg.lam,
            train_size=cfg.train_size, val_size=cfg.val_size, test_size=cfg.test_size,
            snr_db=cfg.snr_db, test_sparsities=cfg.test_sparsities,
            compute_alista_w=needs_alista_w,
        )
        seed_dir = generate_experiment2_dataset(data_cfg, data_root)
        print("[seed {}] dataset ready: {}".format(seed, seed_dir))

        for width in cfg.widths:
            width_dir = os.path.join(seed_dir, width)
            for method_name in cfg.methods:
                key = (method_name, width, seed)
                if key in completed:
                    print("[skip, already done] {}".format(key))
                    continue
                print("[running] {}".format(key))
                t0 = time.time()
                try:
                    rows = run_one(method_name, cfg, width_dir, seed, width, checkpoints_root)
                except Exception as exc:  # keeps the sweep alive across a bad config/OOM/etc as the run is so long we won't actively supervise the run so 
                    # its better to go back afterwards and fix the few runs that had errors.
                    with open(errors_path, "a") as f:
                        f.write(json.dumps({
                            "method": method_name, "width": width, "seed": seed,
                            "error": str(exc), "traceback": traceback.format_exc(),
                        }) + "\n")
                    print("[FAILED] {} : {} (see {})".format(key, exc, errors_path))
                    continue
                with open(results_path, "a") as f:
                    for row in rows:
                        f.write(json.dumps(row) + "\n")
                completed.add(key)
                print("[done] {} in {:.1f}s".format(key, time.time() - t0))

    summarize(results_path, os.path.join(cfg.results_dir, "summary.json"))


def summarize(results_path: str, summary_path: str) -> None:
    """Aggregates results.jsonl into mean/std per (method, width, test_sparsity)
    across seeds -- Table 4: "Statistics: Mean, Std Dev across 10 runs"."""
    if not os.path.exists(results_path):
        return
    rows = []
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        return

    metric_keys = sorted({k for row in rows for k in row if k not in _NON_METRIC_ROW_KEYS})
    groups: Dict[Tuple[str, str, float], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (row["method"], row["width"], row["test_sparsity"])
        groups.setdefault(key, []).append(row)

    summary = []
    for (method_name, width, sparsity), group_rows in sorted(groups.items()):
        entry: Dict[str, Any] = {
            "method": method_name, "width": width, "test_sparsity": sparsity,
            "num_seeds": len(group_rows),
        }
        for k in metric_keys:
            vals = [r[k] for r in group_rows if isinstance(r.get(k), (int, float))]
            if vals:
                entry[k + "_seed_mean"] = float(np.mean(vals))
                entry[k + "_seed_std"] = float(np.std(vals))
        summary.append(entry)

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("Wrote summary ({} groups) to {}".format(len(summary), summary_path))


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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_dir", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--methods", nargs="+", default=list(ALL_METHODS), choices=list(ALL_METHODS))
    p.add_argument("--widths", nargs="+", default=list(WIDTHS), choices=list(WIDTHS))
    p.add_argument("--smoke", action="store_true",
                   help="Tiny dims + training budgets to sanity-check the whole "
                        "pipeline in well under a minute. NOT a real Experiment-2 run.")
    # Individual overrides (ignored if --smoke; --smoke defines its own tiny config).
    p.add_argument("--model_free_num_epochs", type=int, default=None)
    p.add_argument("--model_free_num_steps", type=int, default=None)
    p.add_argument("--model_free_lr", type=float, default=None)
    p.add_argument("--model_based_num_layers", type=int, default=None)
    p.add_argument("--model_based_epochs", type=int, default=None)
    p.add_argument("--num_fista_iters", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.smoke:
        cfg = _smoke_config(args.results_dir, args.seeds, args.methods, args.widths)
    else:
        cfg = OrchestratorConfig(results_dir=args.results_dir, seeds=args.seeds,
                                 methods=args.methods, widths=args.widths)
        for field_name in ("model_free_num_epochs", "model_free_num_steps", "model_free_lr",
                           "model_based_num_layers", "model_based_epochs", "num_fista_iters"):
            value = getattr(args, field_name)
            if value is not None:
                setattr(cfg, field_name, value)
    run_sweep(cfg)


if __name__ == "__main__":
    main()
