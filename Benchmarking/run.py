"""CLI entry point for the generalized L2O benchmark.

Usage:

    python run.py --list-methods
    python run.py --config configs/scale_hrnn_mnist.json
    python run.py --config configs/scale_hrnn_mnist.json --skip-train   # eval only
    python run.py --suite  configs/suite_mnist.json                     # many + leaderboard
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Make ``core`` / ``adapters`` importable regardless of the caller's cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import adapters  # noqa: F401  (import side effect: registers all methods)
from core.registry import list_methods
from core.runner import leaderboard, run_suite, train_evaluate


def _load(path: str):
    with open(path) as f:
        return json.load(f)


def main() -> None:
    p = argparse.ArgumentParser(description="Generalized L2O benchmark runner.")
    p.add_argument("--config", help="Path to a single benchmark config JSON.")
    p.add_argument("--suite", help="Path to a JSON list of benchmark configs.")
    p.add_argument("--skip-train", action="store_true",
                   help="Evaluate an existing checkpoint without retraining.")
    p.add_argument("--list-methods", action="store_true",
                   help="Print registered methods and exit.")
    args = p.parse_args()

    if args.list_methods:
        print("Registered methods:")
        for name in list_methods():
            print("  - {}".format(name))
        return

    if args.config:
        cfg = _load(args.config)
        if args.skip_train:
            cfg["skip_train"] = True
        result = train_evaluate(cfg)
        print(json.dumps({
            "method": result.method,
            "problem": result.problem,
            "checkpoint": result.checkpoint,
            "eval_metrics": result.eval_metrics,
        }, indent=2, default=str))
        return

    if args.suite:
        configs = _load(args.suite)
        if args.skip_train:
            for c in configs:
                c["skip_train"] = True
        results = run_suite(configs)
        print("\n=== Leaderboard (lower final_loss is better) ===")
        for row in leaderboard(results):
            print(json.dumps(row, default=str))
        return

    p.error("one of --config, --suite, or --list-methods is required")


if __name__ == "__main__":
    main()
