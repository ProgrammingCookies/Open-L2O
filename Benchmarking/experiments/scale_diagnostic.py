"""
Experiment-2 model-free scale-convergence diagnostic.

Not a full experiment -- a cheap pre-check before committing to a full
model-free hyperparameter search at Table 3 scale (m=210, n=300). Runs the
standard Experiment 2 orchestrator (experiment2.py's OrchestratorConfig/
run_sweep, reused as-is) at a sequence of (m, n) scales, width="narrow" only
(training sparsity fixed at p=0.1), evaluated in-distribution at that same
p=0.1. This isolates "does this method converge at all, and does the gap to
model-based grow with dimension" from Experiment 2's actual question (does
training-distribution width trade ID quality for OOD robustness, and does
that differ by method family) -- OOD/width is deliberately out of scope here.

Every other hyperparameter (train_size, num_epochs, num_steps, unroll_length,
lr, ...) stays at the real Table 3 values (OrchestratorConfig's defaults) so
scale is the only thing varying between runs -- if it were varying training
budget too, a convergence failure at large n wouldn't tell you whether n or
budget was the cause.

Usage:
    # Step 1: does l2o-dm/l2o-rnnprop converge at all, smallest scale only.
    python experiments/scale_diagnostic.py --results_dir runs/scale_diag --dims 25x50

    # Step 2: full scale sweep (run only after Step 1 looks sane).
    python experiments/scale_diagnostic.py --results_dir runs/scale_diag \
        --dims 25x50 50x100 100x150 210x300
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional, Tuple

_EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _EXPERIMENTS_DIR not in sys.path:
    sys.path.insert(0, _EXPERIMENTS_DIR)

from experiment2 import ALL_METHODS, OrchestratorConfig, run_sweep  # noqa: E402


def _parse_dims(spec: str) -> Tuple[int, int]:
    m_str, n_str = spec.lower().split("x")
    return int(m_str), int(n_str)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_dir", required=True)
    p.add_argument("--dims", nargs="+", default=["25x50", "50x100", "100x150", "210x300"],
                   help="m x n pairs, smallest first so a diagnostic-worthy "
                        "failure (or a per-run cost surprise) surfaces before "
                        "the expensive end of the sweep runs.")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                   help="Fewer than Table 3's 10 -- this is a diagnostic, not "
                        "the final protocol.")
    p.add_argument("--methods", nargs="+", default=["l2o-dm", "l2o-rnnprop", "lista"],
                   choices=list(ALL_METHODS),
                   help="alista omitted by default (extra LP-solve cost for "
                        "its W.npy, and lista alone is enough as the "
                        "model-based reference point); add it back with "
                        "--methods l2o-dm l2o-rnnprop lista alista if wanted.")
    p.add_argument("--in_distribution_p", type=float, default=0.1,
                   help="Single test sparsity, matching narrow width's fixed "
                        "training sparsity p=0.1 -- deliberately ID-only, one "
                        "point, not the full 8-sparsity test battery.")
    # Budget overrides, same escape hatch experiment2.py itself offers --
    # None means "use OrchestratorConfig's real Table-3 default", not "off".
    p.add_argument("--model_free_num_epochs", type=int, default=None)
    p.add_argument("--model_free_num_steps", type=int, default=None)
    p.add_argument("--model_based_epochs", type=int, default=None)
    p.add_argument("--model_free_curriculum", action="store_true",
                   help="Enable --if_cl (curriculum learning on unroll "
                        "length) -- Chen et al. 2020's fix for truncation-"
                        "bias instability in fixed-short-unroll training.")
    p.add_argument("--model_free_imitation", action="store_true",
                   help="Enable --if_mt (imitation learning from analytical "
                        "optimizers) -- Chen et al. 2020's second fix.")
    p.add_argument("--model_free_profile", action="store_true",
                   help="Write a per-epoch (loss, meta-grad-norm) trajectory "
                        "to each run's checkpoint dir as profile.jsonl.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    dims = [_parse_dims(d) for d in args.dims]

    for m, n in dims:
        scale_dir = os.path.join(args.results_dir, "{}x{}".format(m, n))
        print("=" * 80)
        print("[scale] m={} n={} -> {}".format(m, n, scale_dir))
        print("=" * 80)
        cfg = OrchestratorConfig(
            results_dir=scale_dir,
            seeds=args.seeds,
            methods=args.methods,
            widths=["narrow"],
            m=m, n=n,
            test_sparsities=[args.in_distribution_p],
        )
        for field_name in ("model_free_num_epochs", "model_free_num_steps", "model_based_epochs"):
            value = getattr(args, field_name)
            if value is not None:
                setattr(cfg, field_name, value)
        if args.model_free_curriculum:
            cfg.model_free_curriculum = True
        if args.model_free_imitation:
            cfg.model_free_imitation = True
        if args.model_free_profile:
            cfg.model_free_profile = True
        run_sweep(cfg)

    _print_comparison(args.results_dir, dims, args.methods)


def _print_comparison(results_dir: str, dims: List[Tuple[int, int]], methods: List[str]) -> None:
    print()
    print("=" * 80)
    print("Scale-diagnostic summary: mean min_loss across seeds "
          "(width=narrow, in-distribution test sparsity)")
    print("=" * 80)
    print("{:>12s}".format("(m,n)") + "".join("{:>16s}".format(m) for m in methods))
    for m, n in dims:
        results_path = os.path.join(results_dir, "{}x{}".format(m, n), "results.jsonl")
        label = "{}x{}".format(m, n)
        if not os.path.exists(results_path):
            print("{:>12s}  (missing: {})".format(label, results_path))
            continue
        rows = []
        with open(results_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        cells = []
        for method_name in methods:
            vals = [r["min_loss"] for r in rows if r["method"] == method_name]
            cells.append("{:16.4f}".format(sum(vals) / len(vals)) if vals
                         else "{:>16s}".format("n/a"))
        print("{:>12s}".format(label) + "".join(cells))


if __name__ == "__main__":
    main()
